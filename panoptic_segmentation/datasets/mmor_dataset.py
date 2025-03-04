# 1) Load mmor like in many other places (segmentation tool)
# 2) Ability to use it both as an image dataset but also as a video dataset
# 3) Return image & segmentation label
# 4) Make it compatible with detectron2 dataset and their annotation types.
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict

import json_tricks as json  # Allows to load integers etc. correctly
from PIL import Image
from panopticapi.utils import id2rgb
from tqdm import tqdm

from datasets.or_dataset import ORSegmentationDataset, sorted_classes, TRACK_TO_METAINFO, calculate_class_ratios, create_ground_truth_json


class MMORSegmentationDataset(ORSegmentationDataset):
    def __init__(self, root_dir='../MM-OR_data', transforms=None, only_load_metadata=False, use_interpolated=False, max_video_length=None, overlap=0, split='train'):
        """
        Initialize the dataset.
        :param root_dir: Root directory of the MMOR data.
        :param transforms: Optional transforms to be applied on the images.
        """
        self.split = split
        self.cache_dir = Path('mm-or_cache')
        # self.cache_dir = Path('mm-or_cache_debug')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        # these are the file structures
        self.take_folders = ['001_PKA', '002_PKA', '003_TKA', '004_PKA', '005_TKA', '006_PKA', '007_TKA', '008_PKA', '009_TKA', '010_PKA', '011_TKA', '012_1_PKA', '013_PKA', '014_PKA', '015-018_PKA',
                             '019-022_PKA', '023-032_PKA', '033_PKA', '035_PKA', '036_PKA', '037_TKA', '038_TKA']
        # thesee are the takes that are actually used
        self.take_names = ['001_PKA', '002_PKA', '003_TKA', '004_PKA', '005_TKA', '006_PKA', '007_TKA', '008_PKA', '009_TKA', '010_PKA', '011_TKA', '012_1_PKA', '012_2_PKA', '013_PKA', '014_PKA',
                           '015_PKA', '016_PKA', '017_PKA', '018_1_PKA', '018_2_PKA', '019_PKA', '020_PKA', '021_PKA', '022_PKA', '023_PKA', '024_PKA', '025_PKA', '026_PKA', '027_PKA', '028_PKA',
                           '029_PKA', '030_PKA', '031_PKA', '032_PKA', '033_PKA', '035_PKA', '036_PKA', '037_TKA', '038_TKA']
        # seperate some takes even further. Rely on the take_jsons. Make sure combined takes are getting loaded correctly
        self.take_name_to_folder = {'012_1_PKA': '012_PKA', '012_2_PKA': '012_PKA', '015_PKA': '015-018_PKA', '016_PKA': '015-018_PKA', '017_PKA': '015-018_PKA', '018_1_PKA': '015-018_PKA',
                                    '018_2_PKA': '015-018_PKA',
                                    '019_PKA': '019-022_PKA', '020_PKA': '019-022_PKA', '021_PKA': '019-022_PKA', '022_PKA': '019-022_PKA', '023_PKA': '023-032_PKA', '024_PKA': '023-032_PKA',
                                    '025_PKA': '023-032_PKA', '026_PKA': '023-032_PKA',
                                    '027_PKA': '023-032_PKA', '028_PKA': '023-032_PKA', '029_PKA': '023-032_PKA', '030_PKA': '023-032_PKA', '031_PKA': '023-032_PKA', '032_PKA': '023-032_PKA'}

        self.split_to_takes = {
            "train": ['001_PKA', '003_TKA', '005_TKA', '006_PKA', '008_PKA', '010_PKA', '012_1_PKA', '012_2_PKA', '035_PKA', '037_TKA'],
            "small_train": ['001_PKA', '003_TKA', '035_PKA', '037_TKA', '005_TKA'],
            "mini_train": ['013_PKA'],  # just for debugging
            "val": ['002_PKA', '007_TKA', '009_TKA'],
            "test": ['004_PKA', '011_TKA', '036_PKA', '038_TKA'],
            'short_clips': ['013_PKA', '014_PKA', '015_PKA', '016_PKA', '017_PKA', '018_1_PKA', '018_2_PKA', '019_PKA', '020_PKA', '021_PKA', '022_PKA', '023_PKA', '024_PKA', '025_PKA', '026_PKA',
                            '027_PKA',
                            '028_PKA', '029_PKA', '030_PKA', '031_PKA', '032_PKA', '033_PKA']  # these are not considered main takes, but can be used for some other things including anomaly detection
        }
        assert use_interpolated == False, "Interpolated masks are not supported right now"
        self.root_dir = Path(root_dir)
        self.take_jsons = self.root_dir / 'take_jsons'
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
            take_json_path = self.take_jsons / f'{take_name}.json'
            take_path = self.root_dir / take_folder
            if not take_json_path.exists():
                print(f"Take {take_name} does not have a json file as {take_json_path}. Skipping...")
                continue
            with open(take_json_path, 'r') as f:
                data = json.load(f)
                # for azure
                for cam_idx in [1, 4, 5]:
                    take_camidx = f'{take_name}_{cam_idx}'
                    self.samples[take_camidx] = []
                    for _, camera_info in sorted(data['timestamps'].items(), key=lambda x: int(x[0])):
                        rgb_path = take_path / 'colorimage' / f'camera0{cam_idx}_colorimage-{camera_info["azure"]}.jpg'
                        mask_path = take_path / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}.png'
                        interpolated_mask_path = take_path / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}_interpolated.png'
                        if mask_path.exists():
                            self.samples[take_camidx].append({'camera_info': camera_info, 'interpolated': False, 'is_simstation': False})
                        elif interpolated_mask_path.exists() and self.use_interpolated:
                            self.samples[take_camidx].append({'camera_info': camera_info, 'interpolated': True, 'is_simstation': False})
                # for simstation
                for cam_idx in [0, 2, 3]:
                    take_camidx = f'{take_name}_simstation{cam_idx}'
                    self.samples[take_camidx] = []
                    for _, camera_info in sorted(data['timestamps'].items(), key=lambda x: int(x[0])):
                        simstation_rgb_path = take_path / 'simstation' / f'camera0{cam_idx}_{camera_info["simstation"]}.jpg'
                        simstation_mask_path = take_path / f'simstation_segmentation_export_{cam_idx}' / f'{simstation_rgb_path.stem}.png'
                        simstation_interpolated_mask_path = take_path / f'simstation_segmentation_export_{cam_idx}' / f'{simstation_rgb_path.stem}_interpolated.png'
                        if simstation_mask_path.exists():
                            self.samples[take_camidx].append({'camera_info': camera_info, 'interpolated': False, 'is_simstation': True})
                        elif simstation_interpolated_mask_path.exists() and self.use_interpolated:
                            self.samples[take_camidx].append({'camera_info': camera_info, 'interpolated': True, 'is_simstation': True})

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
        is_simstation = sample['is_simstation']
        if not is_simstation:
            rgb_path = root_dir / take_folder / 'colorimage' / f'camera0{cam_idx}_colorimage-{sample["camera_info"]["azure"]}.jpg'
            if sample["interpolated"]:
                mask_path = root_dir / take_folder / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}_interpolated.png'
            else:
                mask_path = root_dir / take_folder / f'segmentation_export_{cam_idx}' / f'{rgb_path.stem}.png'
        else:
            rgb_path = root_dir / take_folder / 'simstation' / f'camera0{cam_idx.replace("simstation", "")}_{sample["camera_info"]["simstation"]}.jpg'
            if sample["interpolated"]:
                mask_path = root_dir / take_folder / f'simstation_segmentation_export_{cam_idx.replace("simstation", "")}' / f'{rgb_path.stem}_interpolated.png'
            else:
                mask_path = root_dir / take_folder / f'simstation_segmentation_export_{cam_idx.replace("simstation", "")}' / f'{rgb_path.stem}.png'

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


def get_mmor_segmentation_dataset(split):
    dataset = MMORSegmentationDataset(max_video_length=200, overlap=3, split=split, use_interpolated=False)
    dataset_dicts = []
    for i in range(len(dataset)):
        dataset_dict = dataset[i]
        dataset_dicts.append(dataset_dict)
    return dataset_dicts


def get_mmor_segmentation_dataset_train():
    return get_mmor_segmentation_dataset('train')


def get_mmor_segmentation_dataset_train_small():
    return get_mmor_segmentation_dataset('small_train')


def get_mmor_segmentation_dataset_train_mini():
    return get_mmor_segmentation_dataset('mini_train')


def get_mmor_segmentation_dataset_val():
    return get_mmor_segmentation_dataset('val')


def get_mmor_segmentation_dataset_test():
    return get_mmor_segmentation_dataset('test')


if __name__ == '__main__':
    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.register("mmor_panoptic_train", get_mmor_segmentation_dataset_train)
    DatasetCatalog.register("mmor_panoptic_train_small", get_mmor_segmentation_dataset_train_small)
    DatasetCatalog.register("mmor_panoptic_train_mini", get_mmor_segmentation_dataset_train_mini)
    DatasetCatalog.register("mmor_panoptic_val", get_mmor_segmentation_dataset_val)
    train_meta = MetadataCatalog.get("mmor_panoptic_train")
    train_small_meta = MetadataCatalog.get("mmor_panoptic_train_small")
    train_mini_meta = MetadataCatalog.get("mmor_panoptic_train_mini")
    val_meta = MetadataCatalog.get("mmor_panoptic_val")
    test_meta = MetadataCatalog.get("mmor_panoptic_test")
    for meta in [train_meta, train_small_meta, train_mini_meta, val_meta, test_meta]:
        meta.set(thing_classes=sorted_classes)
        meta.set(thing_dataset_id_to_contiguous_id={i: i for i in range(len(sorted_classes))})
        meta.set(stuff_classes=[])
        meta.set(ignore_label=255)
        meta.set(stuff_dataset_id_to_contiguous_id={})
        meta.set(panoptic_json='')
        meta.set(categories={i: {'id': i, 'name': name, 'isthing': 1, 'color': TRACK_TO_METAINFO[name]['color']} for i, name in enumerate(sorted_classes)})

    calculate_class_ratios(MMORSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False), save_name='mmor_class_freqs.json')
    create_ground_truth_json(MMORSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False), 'datasets/mmor_ground_truth_train.json')
    create_ground_truth_json(MMORSegmentationDataset(max_video_length=200, overlap=3, split='small_train', use_interpolated=False), 'datasets/mmor_ground_truth_train_small.json')
    create_ground_truth_json(MMORSegmentationDataset(max_video_length=200, overlap=3, split='mini_train', use_interpolated=False), 'datasets/mmor_ground_truth_train_mini.json')
    create_ground_truth_json(MMORSegmentationDataset(max_video_length=200, overlap=3, split='val', use_interpolated=False), 'datasets/mmor_ground_truth_val.json')
    create_ground_truth_json(MMORSegmentationDataset(max_video_length=200, overlap=3, split='test', use_interpolated=False), 'datasets/mmor_ground_truth_test.json')
