from collections import OrderedDict
from typing import List, Dict

from datasets.mmor_dataset import MMORSegmentationDataset
from datasets.or4d_dataset import OR4DSegmentationDataset
from datasets.or_dataset import ORSegmentationDataset, sorted_classes, TRACK_TO_METAINFO, calculate_class_ratios, create_ground_truth_json


class HybridORSegmentationDataset(ORSegmentationDataset):
    def __init__(self, datasets, transforms=None, only_load_metadata=False, use_interpolated=False, max_video_length=None, overlap=0, split='train'):
        self.split = split
        # variables such as take_folders etc. should be merged
        self.take_folders = []
        self.take_names = []
        self.take_name_to_folder = {}
        self.split_to_takes = {}
        self.datasets = datasets
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
        self.sample_to_infos = {}
        self.transforms = transforms
        self.categories = {i: {'id': i, 'name': name, 'isthing': 1, 'color': TRACK_TO_METAINFO[name]['color']} for i, name in enumerate(sorted_classes)}
        assert use_interpolated == False, "Interpolated masks are not supported right now"

        for dataset_name, dataset in datasets.items():
            self.take_folders.extend(dataset.take_folders)
            self.take_names.extend(dataset.take_names)
            for key, value in dataset.split_to_takes.items():
                self.split_to_takes[key] = self.split_to_takes.get(key, []) + value
            self.take_name_to_folder.update(dataset.take_name_to_folder)
            self.samples.update(dataset.samples)
            for key in dataset.samples.keys():
                self.sample_to_infos[key] = {'root_dir': dataset.root_dir, 'process_fn': dataset._process_video_sample_helper, 'cache_dir': dataset.cache_dir}

    def _process_video_sample_helper(self, sample, take, cam_idx, part):
        if part is not None:
            infos = self.sample_to_infos[f'{take}_{cam_idx}_{part}']
        else:
            infos = self.sample_to_infos[f'{take}_{cam_idx}']
        rootdir, process_fn = infos['root_dir'], infos['process_fn']
        return process_fn(sample, take, cam_idx, rootdir)


def get_hybridor_segmentation_dataset(split):
    dataset = HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(max_video_length=200, overlap=3, split=split, use_interpolated=False),
                                           '4dor': OR4DSegmentationDataset(max_video_length=200, overlap=3, split=split, use_interpolated=False)}, max_video_length=200, overlap=3, split=split,
                                          use_interpolated=False)
    dataset_dicts = []
    for i in range(len(dataset)):
        dataset_dict = dataset[i]
        dataset_dicts.append(dataset_dict)
    return dataset_dicts


def get_hybridor_segmentation_dataset_train():
    return get_hybridor_segmentation_dataset('train')


def get_hybridor_segmentation_dataset_train_small():
    return get_hybridor_segmentation_dataset('small_train')


def get_hybridor_segmentation_dataset_train_mini():
    return get_hybridor_segmentation_dataset('mini_train')


def get_hybridor_segmentation_dataset_val():
    return get_hybridor_segmentation_dataset('val')


def get_hybridor_segmentation_dataset_test():
    return get_hybridor_segmentation_dataset('test')


if __name__ == '__main__':
    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.register("hybridor_panoptic_train", get_hybridor_segmentation_dataset_train)
    DatasetCatalog.register("hybridor_panoptic_train_small", get_hybridor_segmentation_dataset_train_small)
    DatasetCatalog.register("hybridor_panoptic_train_mini", get_hybridor_segmentation_dataset_train_mini)
    DatasetCatalog.register("hybridor_panoptic_val", get_hybridor_segmentation_dataset_val)
    train_meta = MetadataCatalog.get("hybridor_panoptic_train")
    train_small_meta = MetadataCatalog.get("hybridor_panoptic_train_small")
    train_mini_meta = MetadataCatalog.get("hybridor_panoptic_train_mini")
    val_meta = MetadataCatalog.get("hybridor_panoptic_val")
    test_meta = MetadataCatalog.get("hybridor_panoptic_test")
    for meta in [train_meta, train_small_meta, train_mini_meta, val_meta, test_meta]:
        meta.set(thing_classes=sorted_classes)
        meta.set(thing_dataset_id_to_contiguous_id={i: i for i in range(len(sorted_classes))})
        meta.set(stuff_classes=[])
        meta.set(ignore_label=255)
        meta.set(stuff_dataset_id_to_contiguous_id={})
        meta.set(panoptic_json='')
        meta.set(categories={i: {'id': i, 'name': name, 'isthing': 1, 'color': TRACK_TO_METAINFO[name]['color']} for i, name in enumerate(sorted_classes)})

    calculate_class_ratios(HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False),
                                                        '4dor': OR4DSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False)}, max_video_length=200, overlap=3,
                                                       split='train', use_interpolated=False), save_name='hybridor_class_freqs.json')
    create_ground_truth_json(HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False),
                                                          '4dor': OR4DSegmentationDataset(max_video_length=200, overlap=3, split='train', use_interpolated=False)}, max_video_length=200, overlap=3,
                                                         split='train', use_interpolated=False), 'datasets/hybridor_ground_truth_train.json')
    create_ground_truth_json(HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(max_video_length=200, overlap=3, split='small_train', use_interpolated=False),
                                                          '4dor': OR4DSegmentationDataset(max_video_length=200, overlap=3, split='small_train', use_interpolated=False)}, max_video_length=200,
                                                         overlap=3, split='small_train', use_interpolated=False), 'datasets/hybridor_ground_truth_train_small.json')
    create_ground_truth_json(HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(max_video_length=200, overlap=3, split='mini_train', use_interpolated=False),
                                                          '4dor': OR4DSegmentationDataset(max_video_length=200, overlap=3, split='mini_train', use_interpolated=False)}, max_video_length=200,
                                                         overlap=3, split='mini_train', use_interpolated=False), 'datasets/hybridor_ground_truth_train_mini.json')
    create_ground_truth_json(HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(max_video_length=200, overlap=3, split='val', use_interpolated=False),
                                                          '4dor': OR4DSegmentationDataset(max_video_length=200, overlap=3, split='val', use_interpolated=False)}, max_video_length=200, overlap=3,
                                                         split='val', use_interpolated=False), 'datasets/hybridor_ground_truth_val.json')
    create_ground_truth_json(HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(max_video_length=200, overlap=3, split='test', use_interpolated=False),
                                                            '4dor': OR4DSegmentationDataset(max_video_length=200, overlap=3, split='test', use_interpolated=False)}, max_video_length=200, overlap=3,
                                                             split='test', use_interpolated=False), 'datasets/hybridor_ground_truth_test.json')
