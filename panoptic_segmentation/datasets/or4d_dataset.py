from collections import OrderedDict
from pathlib import Path
from typing import List, Dict

import json_tricks as json  # Allows to load integers etc. correctly
from PIL import Image
from panopticapi.utils import id2rgb
from tqdm import tqdm

from datasets.or_dataset import ORSegmentationDataset, TRACK_TO_METAINFO, sorted_classes, calculate_class_ratios, create_ground_truth_json


class OR4DSegmentationDataset(ORSegmentationDataset):
    def __init__(self, root_dir='../4D-OR_data', transforms=None, only_load_metadata=False, use_interpolated=False, max_video_length=None, overlap=0, split='train'):
        """
        Initialize the dataset.
        :param root_dir: Root directory of the 4DOR data.
        :param transforms: Optional transforms to be applied on the images.
        """
        self.split = split
        self.cache_dir = Path('4d-or_cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        # these are the file structures
        self.take_folders = ['export_holistic_take1_processed', 'export_holistic_take2_processed', 'export_holistic_take3_processed', 'export_holistic_take4_processed',
                             'export_holistic_take5_processed',
                             'export_holistic_take6_processed', 'export_holistic_take7_processed', 'export_holistic_take8_processed', 'export_holistic_take9_processed',
                             'export_holistic_take10_processed']
        # thesee are the takes that are actually used
        self.take_names = ['001_4DOR', '002_4DOR', '003_4DOR', '004_4DOR', '005_4DOR', '006_4DOR', '007_4DOR', '008_4DOR', '009_4DOR', '010_4DOR']
        self.take_name_to_folder = {'001_4DOR': 'export_holistic_take1_processed', '002_4DOR': 'export_holistic_take2_processed', '003_4DOR': 'export_holistic_take3_processed',
                                    '004_4DOR': 'export_holistic_take4_processed', '005_4DOR': 'export_holistic_take5_processed', '006_4DOR': 'export_holistic_take6_processed',
                                    '007_4DOR': 'export_holistic_take7_processed', '008_4DOR': 'export_holistic_take8_processed', '009_4DOR': 'export_holistic_take9_processed',
                                    '010_4DOR': 'export_holistic_take10_processed'}

        self.split_to_takes = {
            "train": ['001_4DOR', '003_4DOR', '005_4DOR', '007_4DOR', '009_4DOR', '010_4DOR'],
            "small_train": ['001_4DOR', '005_4DOR', '007_4DOR', '009_4DOR'],
            "mini_train": ['001_4DOR'],  # just for debugging
            "val": ['004_4DOR', '008_4DOR'],
            "test": ['002_4DOR', '006_4DOR']
        }
        assert use_interpolated == False, "Interpolated masks are not supported right now"
        self.root_dir = Path(root_dir)
        self.IMAGE_RES = (2048, 1536)
        self.use_interpolated = use_interpolated
        # only respect use_interpolated if self.split is train, otherwise default to False
        if self.split == 'train':
            self.use_interpolated = use_interpolated
        else:
            self.use_interpolated = False
        self.max_video_length = max_video_length
        self.overlap = overlap
        self.samples: Dict[str, List[Dict[str, bool]]] = OrderedDict()  # {take_camidx: [{camera_info, interpolated}]}
        for take_name in tqdm(self.take_names, desc="Loading samples"):
            if take_name not in self.split_to_takes[self.split]:
                continue
            if take_name in self.take_name_to_folder:
                take_folder = self.take_name_to_folder[take_name]
            else:
                take_folder = take_name
            take_json_path = self.root_dir / take_folder / 'timestamp_to_pcd_and_frames_list.json'
            take_path = self.root_dir / take_folder
            if not take_json_path.exists():
                print(f"Take {take_name} does not have a json file as {take_json_path}. Skipping...")
                continue
            with open(take_json_path, 'r') as f:
                data = json.load(f)
                # for azure
                for cam_idx in [1, 2, 5]:
                    take_camidx = f'{take_name}_{cam_idx}'
                    self.samples[take_camidx] = []
                    for _, camera_info in sorted(data, key=lambda x: int(x[0])):
                        color_idx_str = camera_info[f'color_{cam_idx}']
                        rgb_path = take_path / f'colorimage/camera0{cam_idx}_colorimage-{color_idx_str}.jpg'
                        mask_path = take_path / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}.png'
                        interpolated_mask_path = take_path / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}_interpolated.png'
                        if mask_path.exists():
                            self.samples[take_camidx].append({'camera_info': camera_info, 'interpolated': False})
                        elif interpolated_mask_path.exists() and self.use_interpolated:
                            self.samples[take_camidx].append({'camera_info': camera_info, 'interpolated': True})
        self.samples = self._split_video_samples(self.samples, max_video_length, overlap)
        for take_camidx in list(self.samples.keys()):
            if len(self.samples[take_camidx]) == 0:
                self.samples.pop(take_camidx)
        self.transforms = transforms
        self.categories = {i: {'id': i, 'name': name, 'isthing': 1, 'color': TRACK_TO_METAINFO[name]['color']} for i, name in enumerate(sorted_classes)}

    def _process_video_sample_helper(self, sample, take, cam_idx, root_dir):
        if take in self.take_name_to_folder:
            take_folder = self.take_name_to_folder[take]
        else:
            take_folder = take

        rgb_path = root_dir / take_folder / 'colorimage' / f'camera0{cam_idx}_colorimage-{sample["camera_info"][f"color_{cam_idx}"]}.jpg'
        if sample["interpolated"]:
            mask_path = root_dir / take_folder / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}_interpolated.png'
        else:
            mask_path = root_dir / take_folder / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}.png'
        segments_infos, panoptic_seg, panoptic_seg_for_val = self._process_panoptic_seg_mask(mask_path)

        # convert panoptic seg to the classic panoptic annotation style compatible with detectron2.
        # pan_seg_file_name (str): The full path to panoptic segmentation ground truth file. It should be an RGB image whose pixel values are integer ids encoded using the panopticapi.utils.id2rgb function. The ids are defined by segments_info. If an id does not appear in segments_info, the pixel is considered unlabeled and is usually ignored in training & evaluation.
        panoptic_seg_rgb = id2rgb(panoptic_seg)
        panoptic_seg_rgb_for_val = self.label_mask_to_rgb(panoptic_seg_for_val)
        # we save this in a designated folder, and then we return only its path
        pan_seg_file_name = mask_path.parent.parent / f'panoptic_seg_{cam_idx}' / f'{mask_path.stem}.png'
        pan_seg_file_name_for_val = mask_path.parent.parent / f'panoptic_seg_{cam_idx}_for_val' / f'{mask_path.stem}.png'
        pan_seg_file_name.parent.mkdir(exist_ok=True, parents=True)
        pan_seg_file_name_for_val.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(panoptic_seg_rgb).save(pan_seg_file_name)
        Image.fromarray(panoptic_seg_rgb_for_val).save(pan_seg_file_name_for_val)

        return str(rgb_path), segments_infos, pan_seg_file_name, pan_seg_file_name_for_val


def get_4dor_segmentation_dataset(split):
    dataset = OR4DSegmentationDataset(max_video_length=200, overlap=3, split=split, use_interpolated=False)
    dataset_dicts = []
    for i in range(len(dataset)):
        dataset_dict = dataset[i]
        dataset_dicts.append(dataset_dict)
    return dataset_dicts


def get_4dor_segmentation_dataset_train():
    return get_4dor_segmentation_dataset('train')


def get_4dor_segmentation_dataset_train_small():
    return get_4dor_segmentation_dataset('small_train')


def get_4dor_segmentation_dataset_train_mini():
    return get_4dor_segmentation_dataset('mini_train')


def get_4dor_segmentation_dataset_val():
    return get_4dor_segmentation_dataset('val')


def get_4dor_segmentation_dataset_test():
    return get_4dor_segmentation_dataset('test')


if __name__ == '__main__':
    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.register("4dor_panoptic_train", get_4dor_segmentation_dataset_train)
    DatasetCatalog.register("4dor_panoptic_train_small", get_4dor_segmentation_dataset_train_small)
    DatasetCatalog.register("4dor_panoptic_train_mini", get_4dor_segmentation_dataset_train_mini)
    DatasetCatalog.register("4dor_panoptic_val", get_4dor_segmentation_dataset_val)
    train_meta = MetadataCatalog.get("4dor_panoptic_train")
    train_small_meta = MetadataCatalog.get("4dor_panoptic_train_small")
    train_mini_meta = MetadataCatalog.get("4dor_panoptic_train_mini")
    val_meta = MetadataCatalog.get("4dor_panoptic_val")
    test_meta = MetadataCatalog.get("4dor_panoptic_test")
    for meta in [train_meta, train_small_meta, train_mini_meta, val_meta, test_meta]:
        meta.set(thing_classes=sorted_classes)
        meta.set(thing_dataset_id_to_contiguous_id={i: i for i in range(len(sorted_classes))})
        meta.set(stuff_classes=[])
        meta.set(ignore_label=255)
        meta.set(stuff_dataset_id_to_contiguous_id={})
        meta.set(panoptic_json='')
        meta.set(categories={i: {'id': i, 'name': name, 'isthing': 1, 'color': TRACK_TO_METAINFO[name]['color']} for i, name in enumerate(sorted_classes)})

    calculate_class_ratios(OR4DSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False), save_name='4dor_class_freqs.json')
    create_ground_truth_json(OR4DSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False), 'datasets/4dor_ground_truth_train.json')
    create_ground_truth_json(OR4DSegmentationDataset(max_video_length=200, overlap=3, split='small_train', use_interpolated=False), 'datasets/4dor_ground_truth_train_small.json')
    create_ground_truth_json(OR4DSegmentationDataset(max_video_length=200, overlap=3, split='mini_train', use_interpolated=False), 'datasets/4dor_ground_truth_train_mini.json')
    create_ground_truth_json(OR4DSegmentationDataset(max_video_length=200, overlap=3, split='val', use_interpolated=False), 'datasets/4dor_ground_truth_val.json')
    create_ground_truth_json(OR4DSegmentationDataset(max_video_length=200, overlap=3, split='test', use_interpolated=False), 'datasets/4dor_ground_truth_test.json')