import json
import multiprocessing
from abc import abstractmethod
from collections import OrderedDict

import cv2
import json_tricks as json  # Allows to load integers etc. correctly
import numpy as np
import torch
from panopticapi.utils import rgb2id
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

TRACK_TO_METAINFO = {
    'instrument_table': {'color': (255, 51, 153), 'label': 1},
    'ae': {'color': (0, 0, 255), 'label': 2},
    'ot': {'color': (255, 255, 0), 'label': 3},
    'mps_station': {'color': (133, 0, 133), 'label': 4},
    'patient': {'color': (255, 0, 0), 'label': 5},
    'drape': {'color': (183, 91, 255), 'label': 6},
    'anest': {'color': (177, 255, 110), 'label': 7},
    'circulator': {'color': (255, 128, 0), 'label': 8},
    'assistant_surgeon': {'color': (116, 166, 116), 'label': 9},
    'head_surgeon': {'color': (76, 161, 245), 'label': 10},
    'mps': {'color': (125, 100, 25), 'label': 11},
    'nurse': {'color': (128, 255, 0), 'label': 12},
    'drill': {'color': (0, 255, 128), 'label': 13},  # Changed
    'hammer': {'color': (204, 0, 0), 'label': 15},
    'saw': {'color': (0, 255, 234), 'label': 16},
    'tracker': {'color': (255, 128, 128), 'label': 17},  # Changed
    'mako_robot': {'color': (60, 75, 255), 'label': 18},  # Changed
    'monitor': {'color': (255, 255, 128), 'label': 24},  # Changed
    'c_arm': {'color': (0, 204, 128), 'label': 25},  # Changed
    'unrelated_person': {'color': (255, 255, 255), 'label': 26},
    'student': {'color': (162, 232, 108), 'label': 27},
    'secondary_table': {'color': (153, 0, 153), 'label': 28},
    'cementer': {'color': (153, 76, 0), 'label': 29},
    '__background__': {'color': (0, 0, 0), 'label': 0}
}
# sorted classes by their label
sorted_classes = sorted(TRACK_TO_METAINFO.keys(), key=lambda x: TRACK_TO_METAINFO[x]['label'])
label_to_category_id = {TRACK_TO_METAINFO[track]['label']: i for i, track in enumerate(sorted_classes)}  # 0 is background
for key, value in TRACK_TO_METAINFO.items():
    c = value['color']
    segment_id = c[0] + c[1] * 256 + c[2] * 256 * 256
    value['segment_id'] = segment_id


def clean_mask(mask, area_threshold=10):
    """
    Clean a mask by removing small connected components.

    Args:
        mask (np.ndarray): Input mask array with uint8 labels.
        area_threshold (int): Minimum area threshold for connected components to be kept.

    Returns:
        np.ndarray: Cleaned mask array.
    """
    # Find all unique labels in the mask
    unique_labels = np.unique(mask)
    cleaned_mask = mask.copy()

    # Iterate over each unique label
    for label in unique_labels:
        # Skip background label (assuming it's 0)
        if label == 0:
            continue

        # Create a binary mask for the current label
        binary_mask = (mask == label).any(-1).astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over contours
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # If the area is smaller than the threshold, fill the contour with black
            if area < area_threshold:
                cv2.drawContours(cleaned_mask, [contour], -1, 0, thickness=cv2.FILLED)
    return cleaned_mask


def mask_to_polygons(mask):
    """Convert a mask ndarray (1s and 0s) to polygons."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            polygons.append(contour)
    return polygons


def encode_panoptic(class_id, instance_id, label_divisor=10):
    """Encode class and instance ID into a single integer."""
    return class_id * label_divisor + instance_id


def decode_panoptic(panoptic_label, label_divisor=10):
    """Decode the panoptic label into class and instance ID."""
    class_id = panoptic_label // label_divisor
    instance_id = panoptic_label % label_divisor
    return class_id, instance_id


class ORSegmentationDataset(Dataset):
    @abstractmethod
    def __init__(self):
        self.samples = []
        self.IMAGE_RES = ()
        self.take_name_to_folder = {}
        self.categories = {}

    def _split_video_samples(self, samples, max_video_length, overlap):
        if max_video_length is None:
            return samples

        new_samples = OrderedDict()
        for take_camidx, frames in samples.items():
            num_frames = len(frames)
            if num_frames <= max_video_length:
                new_samples[take_camidx] = frames
                continue

            for start_idx in range(0, num_frames, max_video_length - overlap):
                end_idx = min(start_idx + max_video_length, num_frames)
                segment_frames = frames[start_idx:end_idx]
                new_take_camidx = f"{take_camidx}_part{start_idx // (max_video_length - overlap)}"
                new_samples[new_take_camidx] = segment_frames
        return new_samples

    def __len__(self):
        return len(self.samples)

    def _process_panoptic_seg_mask(self, mask_path):
        mask_img = cv2.imread(mask_path.as_posix()).astype(np.uint8)
        mask_img = clean_mask(mask_img)
        mask_img = cv2.resize(mask_img, self.IMAGE_RES, interpolation=cv2.INTER_NEAREST)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        # Convert to a panoptic segmentation dataset compatible with detectron2
        # Process the mask to extract instance segmentation
        unique_labels = np.unique(mask_img)  # Assuming each unique value is an instance
        segments_info = []
        panoptic_seg = np.zeros_like(mask_img, dtype=np.int32)
        panoptic_seg_for_val = np.zeros_like(mask_img, dtype=np.int32)
        TAKE_NAME = mask_path.parent.parent.name
        LIKELY_MISTAKES = {19: 'random_artifact/tracking_tool', 20: 'random_artifact/tracking_tool', 14: 'surgeon_hand_artifact', 22: 'random_artifact', 23: 'random_artifact'}
        for label in unique_labels:
            if label == 0:  # Assuming 0 is background ->  Actually no reason to skip background.
                continue
            instance_id = 0  # maybe in the future we will use
            try:
                category_id = label_to_category_id[label]
            except KeyError:
                if label in LIKELY_MISTAKES:
                    print(f"Unsupported Label: {label}; Likely mistake: {LIKELY_MISTAKES[label]}")
                else:
                    pass
                    # print(f"Unsupported Label: {label}; plotting image...")
                    # # In these cases we want to figure out what happened. Plot both the full mask and the unknown part of the mask on top of each other, and save it with the file name
                    # unknown_mask = (mask_img == label)
                    # image_to_save = cv2.cvtColor(mask_img * 5, cv2.COLOR_GRAY2BGR)
                    # image_to_save[unknown_mask] = [255, 0, 0]
                    # cv2.imwrite(f'debug_results/unknown_labels/unknown_label_{label}_{TAKE_NAME}_{mask_path.stem}.png', image_to_save)
                continue
            panoptic_id = encode_panoptic(category_id, instance_id)
            instance_mask = (mask_img == label).astype(np.uint8)
            color = self.categories[category_id]['color']
            segments_info.append({
                "id": panoptic_id,
                "category_id": category_id,
                "area": np.sum(instance_mask),
                "bbox": cv2.boundingRect(instance_mask),
                "isthing": 0,
                "iscrowd": 0,
                'rgb2idcolor': rgb2id(color)
            })
            panoptic_seg += instance_mask * panoptic_id
            panoptic_seg_for_val += instance_mask * category_id

        return segments_info, panoptic_seg, panoptic_seg_for_val

    def label_mask_to_rgb(self, mask):
        '''
        mask is a 2D array with the labels, we want to convert it to a 3D array with the colors
        iterate over all the classes in the mask, get their color
        '''
        panoptic_seg_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label in np.unique(mask):
            color = self.categories[label]['color']
            panoptic_seg_rgb[mask == label] = color
        return panoptic_seg_rgb

    @abstractmethod
    def _process_video_sample_helper(self, sample, take, cam_idx, root_dir):
        pass

    def _process_video_sample(self, video_sample, take, cam_idx, part=None):
        video_id = f'{take}_{cam_idx}'
        video_folder = self.take_name_to_folder[take] if take in self.take_name_to_folder else take
        # Define the cache file path
        if hasattr(self, 'cache_dir'):
            if part is not None:
                cache_file = self.cache_dir / f'{take}_{cam_idx}_{part}_{self.use_interpolated}.npz'
            else:
                cache_file = self.cache_dir / f'{take}_{cam_idx}_{self.use_interpolated}.npz'
        else:
            infos = self.sample_to_infos[f'{take}_{cam_idx}_{part}'] if part is not None else self.sample_to_infos[f'{take}_{cam_idx}']
            cache_dir = infos['cache_dir']
            if part is not None:
                cache_file = cache_dir / f'{take}_{cam_idx}_{part}_{self.use_interpolated}.npz'
            else:
                cache_file = cache_dir / f'{take}_{cam_idx}_{self.use_interpolated}.npz'

        # Check if the cache file exists
        if cache_file.exists():
            # Load cached data
            cached_data = np.load(cache_file, allow_pickle=True)
            # for debuggin purposes, load an image and plot the segmentation mask on top of it
            # original_image = Image.open(cached_data['file_names'][10]).convert("RGBA")
            # panoptic_seg = Image.open(cached_data['pan_seg_file_names'][10]).convert("RGBA")
            # # add 100 to every non-zero pixel value to make it more visible, make sure we don't clip anything and cause overflow
            # panoptic_seg = np.array(panoptic_seg).astype(int)
            # panoptic_seg[panoptic_seg != 0] = np.clip(panoptic_seg[panoptic_seg != 0] + 100, 0, 255)
            # panoptic_seg = Image.fromarray(panoptic_seg.astype(np.uint8))
            # blended_image = Image.blend(original_image, panoptic_seg.resize((original_image.size)), alpha=0.6)
            # blended_image.save(f'debug_results/segmentation_example_{take}_{cam_idx}.png')

            return {
                'file_names': cached_data['file_names'].tolist(),
                'image_ids': cached_data['image_ids'].tolist(),
                'segments_infos': cached_data['segments_infos'].tolist(),
                'video_id': f'{take}_{cam_idx}',
                'video_id_part': f'{take}_{cam_idx}_{part}' if part is not None else f'{take}_{cam_idx}',
                'pan_seg_file_names': cached_data['pan_seg_file_names'].tolist(),
                'pan_seg_file_names_for_val': cached_data['pan_seg_file_names_for_val'].tolist(),
                'video_folder': video_folder
            }
        if hasattr(self, 'root_dir'):
            # Use process_map to process samples in parallel
            n_cpus = multiprocessing.cpu_count()
            results = process_map(
                self._process_video_sample_helper,
                video_sample,
                [take] * len(video_sample),
                [cam_idx] * len(video_sample),
                [self.root_dir] * len(video_sample),
                max_workers=n_cpus,
                desc=f"Processing video sample {take}_{cam_idx}_{part}" if part is not None else f"Processing video sample {take}_{cam_idx}",
                chunksize=(len(video_sample) // n_cpus // 2) + 1
            )
        else:
            n_cpus = multiprocessing.cpu_count()
            results = process_map(
                self._process_video_sample_helper,
                video_sample,
                [take] * len(video_sample),
                [cam_idx] * len(video_sample),
                [part] * len(video_sample),
                max_workers=n_cpus,
                desc=f"Processing video sample {take}_{cam_idx}_{part}" if part is not None else f"Processing video sample {take}_{cam_idx}",
                chunksize=(len(video_sample) // n_cpus // 2) + 1
            )
        file_names = []
        all_segments_infos = []
        all_pan_seg_file_names = []
        all_pan_seg_file_names_for_val = []
        image_ids = list(range(len(video_sample)))

        for rgb_path, segments_infos, pan_seg_file_name, pan_seg_file_name_for_val in results:
            file_names.append(rgb_path)
            all_segments_infos.append(segments_infos)
            all_pan_seg_file_names.append(pan_seg_file_name)
            all_pan_seg_file_names_for_val.append(pan_seg_file_name_for_val)

        # Save results to cache file
        np.savez_compressed(
            cache_file,
            file_names=np.array(file_names),
            image_ids=np.array(image_ids),
            segments_infos=np.array(all_segments_infos, dtype=object),
            pan_seg_file_names=np.array(all_pan_seg_file_names),
            pan_seg_file_names_for_val=np.array(all_pan_seg_file_names_for_val)
        )

        return {'file_names': file_names, 'image_ids': image_ids, 'segments_infos': all_segments_infos, 'video_id': f'{take}_{cam_idx}',
                'video_id_part': f'{take}_{cam_idx}_{part}' if part is not None else f'{take}_{cam_idx}', 'pan_seg_file_names': np.array(all_pan_seg_file_names),
                'pan_seg_file_names_for_val': np.array(all_pan_seg_file_names_for_val), 'video_folder': video_folder}

    def __getitem__(self, idx):
        """
        Load an image and its annotations.
        """
        # sample a key from the self.samples, which is an OrderedDict
        take_camidx = list(self.samples.keys())[idx]
        if 'part' in take_camidx:
            take, cam_idx, part = take_camidx.rsplit('_', 2)
        else:
            take, cam_idx = take_camidx.rsplit('_', 1)
            part = None
        video_sample = self.samples[take_camidx]
        cam_idx = int(cam_idx) if cam_idx.isdigit() else cam_idx
        video_sample_dict = self._process_video_sample(video_sample, take, cam_idx, part)
        return video_sample_dict


def calculate_class_ratios(dataset, save_name):
    '''
    We want to measure how offer a class occurs, defined as: pixels of class i in the entire dataset / total annotated pixels in the entire dataset
    '''
    # iterate over the dataset with a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    class_occurrences = {i: 0 for i in range(len(sorted_classes))}

    for video_sample in tqdm(dataloader, desc="Calculating class ratios"):
        for segments_infos in video_sample[0]['segments_infos']:
            for segment_info in segments_infos:
                class_occurrences[segment_info['category_id']] += segment_info['area']

    # save into a json file
    with open(f'datasets/{save_name}', 'w') as f:
        json.dump(class_occurrences, f)


def create_ground_truth_json(dataset, output_path):
    """
    Create a ground truth JSON file in the format required for evaluation.

    Args:
        dataset: The dataset object.
        output_path: The path where the ground truth JSON file will be saved.
    """
    ground_truth = {
        "categories": [],
        "videos": [],
        "annotations": []
    }

    # Fill categories
    for i, class_name in enumerate(sorted_classes):
        ground_truth["categories"].append({
            "id": i,
            "name": class_name,
            "isthing": 1,
            "color": TRACK_TO_METAINFO[class_name]["color"]
        })

    # Process samples
    for idx in range(len(dataset)):
        video_sample_dict = dataset[idx]
        # video_id = video_sample_dict['video_id']
        video_id = video_sample_dict['video_id_part']
        video_folder = video_sample_dict['video_folder']
        video_annotations = {'video_id': video_id, 'annotations': []}

        video_info = {
            "video_id": video_id,
            "images": [],
            'video_folder': video_folder
        }

        for image_id, (file_name, segments_info) in enumerate(zip(video_sample_dict['file_names'], video_sample_dict['segments_infos'])):
            image_info = {
                "id": f"{video_id}_{image_id}",
                "file_name": file_name.split('/')[-1],
                "height": dataset.IMAGE_RES[1],
                "width": dataset.IMAGE_RES[0],
                "video_id": video_id
            }
            video_info["images"].append(image_info)
            for segment_info in segments_info:
                segment_info['area'] = int(segment_info['area'])
                segment_info['id'] = int(segment_info['rgb2idcolor'])
            image_annotation = {'image_id': f"{video_id}_{image_id}", 'file_name': file_name.split('/')[-1], 'segments_info': segments_info}
            video_annotations['annotations'].append(image_annotation)

        ground_truth["annotations"].append(video_annotations)

        ground_truth["videos"].append(video_info)

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f)
