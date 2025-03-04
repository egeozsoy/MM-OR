import json
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from helpers.configurations import MMOR_TAKE_NAMES, MMOR_DATA_ROOT_PATH, OR4D_TAKE_NAMES, OR_4D_DATA_ROOT_PATH, OR4D_TAKE_NAME_TO_FOLDER, MMOR_TAKE_NAME_TO_FOLDER

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


def downsample_mask_preserve_classes(mask, OUTPUT_RES):
    h, w = mask.shape
    scale_h = h // OUTPUT_RES
    scale_w = w // OUTPUT_RES

    # Trim the mask to make it divisible by the scaling factors
    mask = mask[:OUTPUT_RES * scale_h, :OUTPUT_RES * scale_w]

    num_classes = mask.max() + 1  # Assuming class labels start from 0

    # Compute class frequencies
    class_frequencies = np.bincount(mask.flatten(), minlength=num_classes)
    # Define class priorities (inverse of frequencies so that rarer classes have higher priority)
    class_priorities = 1 / (class_frequencies + 1e-6)  # Add a small epsilon to avoid division by zero

    # Initialize an array to hold downsampled masks for each class
    masks_downsampled = np.zeros((OUTPUT_RES, OUTPUT_RES, num_classes), dtype=np.uint8)

    for c in range(num_classes):
        # Create a binary mask for the class
        class_mask = (mask == c).astype(np.uint8)
        # Reshape to apply max pooling
        class_mask = class_mask.reshape(OUTPUT_RES, scale_h, OUTPUT_RES, scale_w)
        # Perform max pooling
        class_mask_downsampled = class_mask.max(axis=(1, 3))
        masks_downsampled[:, :, c] = class_mask_downsampled

    # Multiply masks by class priorities to get priority scores
    priority_scores = masks_downsampled * class_priorities[np.newaxis, np.newaxis, :]

    # Assign the class with the highest priority score to each pixel
    downsampled_mask = np.argmax(priority_scores, axis=2).astype(np.uint8)

    return downsampled_mask


def get_rgb_mask(mask):
    # Initialize mask_rgb directly with BGR format
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for track, metainfo in TRACK_TO_METAINFO.items():
        mask_rgb[mask == metainfo['label']] = metainfo['color']
    # Combine the mask_rgb with the original image
    return mask_rgb


def _segmask_helper(take_name, model_name, USE_GT=False, OUTPUT_RES=32):
    dataset_type = '4DOR' if '4DOR' in take_name else 'MMOR'
    if dataset_type == '4DOR':
        output_folder = OR_4D_DATA_ROOT_PATH / 'take_segmasks_per_timepoint' / take_name
        output_folder.mkdir(parents=True, exist_ok=True)
        take_folder = OR4D_TAKE_NAME_TO_FOLDER.get(take_name, take_name)
        take_path = OR_4D_DATA_ROOT_PATH / take_folder
    else:
        output_folder = MMOR_DATA_ROOT_PATH / 'take_segmasks_per_timepoint' / take_name
        output_folder.mkdir(parents=True, exist_ok=True)
        take_folder = MMOR_TAKE_NAME_TO_FOLDER.get(take_name, take_name)
        take_path = MMOR_DATA_ROOT_PATH / take_folder

    if dataset_type == "MMOR":
        json_path = Path('MM-OR_data') / 'take_jsons' / f'{take_name}.json'
        # Read MMOR/Simstation JSON file for timestamps and image paths
        with json_path.open() as f:
            take_json = json.load(f)
            timestamps = take_json['timestamps']
            timestamps = {int(k): v for k, v in timestamps.items()}
            timestamps = sorted(timestamps.items())
    elif dataset_type == "4DOR":
        internal_take_name = f'export_holistic_take{int(take_name.replace("_4DOR", ""))}_processed'
        json_path = Path('4D-OR_data') / internal_take_name / 'timestamp_to_pcd_and_frames_list.json'
        # Read 4D-OR JSON file for timestamps and image paths
        with json_path.open() as f:
            take_json = json.load(f)
            timestamps = sorted(take_json)

    # Iterate through timestamps and generate multiview frames
    for timestamp, image_files in tqdm(timestamps):
        timestamp_str = str(timestamp).zfill(6)
        segmasks = []

        if dataset_type == "MMOR":
            for c_idx in [1, 4, 5]:
                if USE_GT:
                    rgb_path = take_path / 'colorimage' / f'camera0{c_idx}_colorimage-{image_files["azure"]}.jpg'
                    mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}.png'
                    interpolated_mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}_interpolated.png'
                    if mask_path.exists():
                        segmask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                        # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                        segmasks.append(segmask)

                    elif interpolated_mask_path.exists():
                        segmask = cv2.imread(str(interpolated_mask_path), cv2.IMREAD_GRAYSCALE)
                        segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                        # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                        segmasks.append(segmask)
                else:
                    all_camera_parts = (model_name / 'inference' / 'pan_pred').glob(f'{take_name}_{c_idx}_*')
                    for all_camera_part in all_camera_parts:
                        image_path = all_camera_part / f'camera0{c_idx}_colorimage-{image_files["azure"]}.png'
                        if image_path.exists():  # this is already the rgb_mask. First map it back to label representation.
                            colored_mask = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                            segmask = np.zeros((colored_mask.shape[0], colored_mask.shape[1]), dtype=np.uint8)
                            for track, metainfo in TRACK_TO_METAINFO.items():
                                segmask[(colored_mask == metainfo['color']).all(axis=2)] = metainfo['label']
                            segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                            # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                            segmasks.append(segmask)
                            break
            if len(segmasks) == 0:  # assume azure is not available, use simstation instead
                for c_idx in [0, 2, 3]:
                    if USE_GT:
                        simstation_rgb_path = take_path / 'simstation' / f'camera0{c_idx}_{image_files["simstation"]}.jpg'
                        simstation_mask_path = take_path / f'simstation_segmentation_export_{c_idx}' / f'{simstation_rgb_path.stem}.png'
                        simstation_interpolated_mask_path = take_path / f'simstation_segmentation_export_{c_idx}' / f'{simstation_rgb_path.stem}_interpolated.png'
                        if simstation_mask_path.exists():
                            segmask = cv2.imread(str(simstation_mask_path), cv2.IMREAD_GRAYSCALE)
                            segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                            # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                            segmasks.append(segmask)
                        elif simstation_interpolated_mask_path.exists():
                            segmask = cv2.imread(str(simstation_interpolated_mask_path), cv2.IMREAD_GRAYSCALE)
                            segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                            # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                            segmasks.append(segmask)
                    else:
                        all_camera_parts = (model_name / 'inference' / 'pan_pred').glob(f'{take_name}_simstation{c_idx}_*')
                        for all_camera_part in all_camera_parts:
                            image_path = all_camera_part / f'camera0{c_idx}_{image_files["simstation"]}.png'
                            if image_path.exists():
                                colored_mask = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                                segmask = np.zeros((colored_mask.shape[0], colored_mask.shape[1]), dtype=np.uint8)
                                for track, metainfo in TRACK_TO_METAINFO.items():
                                    segmask[(colored_mask == metainfo['color']).all(axis=2)] = metainfo['label']
                                segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                                # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                                segmasks.append(segmask)
        elif dataset_type == "4DOR":
            # 4D-OR uses different camera indices (1, 2, 5)
            for c_idx in [1, 2, 5]:
                if USE_GT:
                    color_idx_str = image_files[f'color_{c_idx}']
                    rgb_path = take_path / f'colorimage/camera0{c_idx}_colorimage-{color_idx_str}.jpg'
                    mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}.png'
                    interpolated_mask_path = take_path / f'segmentation_export_{c_idx}' / f'{rgb_path.stem}_interpolated.png'
                    if mask_path.exists():
                        segmask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                        # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                        segmasks.append(segmask)
                    elif interpolated_mask_path.exists():
                        segmask = cv2.imread(str(interpolated_mask_path), cv2.IMREAD_GRAYSCALE)
                        segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                        # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                        segmasks.append(segmask)
                else:
                    all_camera_parts = (model_name / 'inference' / 'pan_pred').glob(f'{take_name}_{c_idx}_*')
                    for all_camera_part in all_camera_parts:
                        image_path = all_camera_part / f'camera0{c_idx}_colorimage-{image_files[f"color_{c_idx}"]}.png'
                        if image_path.exists():
                            colored_mask = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                            segmask = np.zeros((colored_mask.shape[0], colored_mask.shape[1]), dtype=np.uint8)
                            for track, metainfo in TRACK_TO_METAINFO.items():
                                segmask[(colored_mask == metainfo['color']).all(axis=2)] = metainfo['label']
                            segmask = downsample_mask_preserve_classes(segmask, OUTPUT_RES)
                            # segmask = cv2.cvtColor(get_rgb_mask(segmask), cv2.COLOR_BGR2RGB)
                            segmasks.append(segmask)

        # keep processing
        for i in range(len(segmasks)):
            segmask = segmasks[i]
            # save it to disk
            output_path = output_folder / f'{timestamp_str}_{i}_GT{USE_GT}.png'
            cv2.imwrite(str(output_path), segmask)


def main():
    seg_path = Path('panoptic_segmentation/output_CTVIS_R50_HybridOR_withsimstation')
    USE_GT = False
    OUTPUT_RES = 32
    # 4dor
    process_map(partial(_segmask_helper, model_name=seg_path, USE_GT=USE_GT, OUTPUT_RES=OUTPUT_RES), OR4D_TAKE_NAMES, max_workers=24, chunksize=1)
    # mmor
    process_map(partial(_segmask_helper, model_name=seg_path, USE_GT=USE_GT, OUTPUT_RES=OUTPUT_RES), MMOR_TAKE_NAMES, max_workers=24, chunksize=1)


if __name__ == '__main__':
    main()
