import copy
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import wandb
from PIL import Image
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from panopticapi.utils import rgb2id

from utils.eval_vpq_vspw import vpq_compute_parallel


# Assuming vpq_compute_parallel and other required functions are defined as in your provided script


class VPSEvaluator(DatasetEvaluator):
    """
    Only for save the prediction results in VIPSeg format
    """

    def __init__(
            self,
            dataset_name,
            tasks=None,
            distributed=True,
            output_dir=None,
            *,
            use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        thing_dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        stuff_dataset_id_to_contiguous_id = self._metadata.stuff_dataset_id_to_contiguous_id
        self.contiguous_id_to_thing_dataset_id = {}
        self.contiguous_id_to_stuff_dataset_id = {}
        for i, key in enumerate(thing_dataset_id_to_contiguous_id.values()):
            self.contiguous_id_to_thing_dataset_id.update({i: key})
        for i, key in enumerate(stuff_dataset_id_to_contiguous_id.values()):
            self.contiguous_id_to_stuff_dataset_id.update({i: key})
        json_file = PathManager.get_local_path(self._metadata.panoptic_json)

        self._do_evaluation = False

    def reset(self):
        self._predictions = []
        PathManager.mkdirs(self._output_dir)
        if not os.path.exists(os.path.join(self._output_dir, 'pan_pred')):
            os.makedirs(os.path.join(self._output_dir, 'pan_pred'), exist_ok=True)

    def process(self, inputs, outputs):
        assert len(inputs) == 1, "More than one inputs are loaded for inference!"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if 'segments_infos' not in outputs:
            output_height, output_width = outputs['output_height'], outputs['output_width']
            cur_masks = torch.stack(outputs['pred_masks']).to(device)
            cur_scores = torch.as_tensor(outputs['pred_scores']).to(device)
            cur_classes = torch.as_tensor(outputs['pred_labels']).to(device)
            cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(device) * cur_masks
            del cur_scores

            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=device)
            segments_infos = []
            out_ids = []
            current_segment_id = 0

            if cur_masks.shape[0] > 0:
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class < 58
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1
                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_infos.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )

            outputs = {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }

            del cur_masks, cur_classes, cur_prob_masks

        video_id = inputs[0]["video_id_part"] if 'video_id_part' in inputs[0] else inputs[0]["video_id"]
        image_names = [inputs[0]['file_names'][idx] for idx in inputs[0]["frame_idx"]]
        img_shape = outputs['image_size']
        pan_seg_result = outputs['pred_masks'].to(device)
        segments_infos = outputs['segments_infos']
        segments_infos_ = []

        pan_format = torch.zeros((pan_seg_result.shape[0], img_shape[0], img_shape[1], 3), dtype=torch.uint8, device=device)
        for segments_info in segments_infos:
            id = segments_info['id']
            is_thing = segments_info['isthing']
            sem = segments_info['category_id']
            try:
                if is_thing:
                    sem = self.contiguous_id_to_thing_dataset_id[sem]
                else:
                    sem = self.contiguous_id_to_stuff_dataset_id[sem - len(self.contiguous_id_to_thing_dataset_id)]
            except Exception as e:
                print(e)
                continue

            mask = pan_seg_result == id
            color = self._metadata.categories[sem]['color']
            # pan_format[mask] = color
            pan_format[mask] = torch.tensor(color, device=device, dtype=torch.uint8)

            dts = []
            dt_ = {"category_id": int(sem), "iscrowd": 0, "id": int(rgb2id(color))}
            for i in range(pan_format.shape[0]):
                mask_i = mask[i]
                if mask_i.sum().item() == 0:
                    dts.append(None)
                    continue

                y_indices, x_indices = torch.nonzero(mask_i, as_tuple=True)
                x_min, x_max = x_indices.min().item(), x_indices.max().item()
                y_min, y_max = y_indices.min().item(), y_indices.max().item()
                area = mask_i.sum().item()

                dt = {
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": area
                }
                dt.update(dt_)
                dts.append(dt)

            segments_infos_.append(dts)

        pan_format = pan_format.cpu().numpy()
        del pan_seg_result

        def _image_save_helper(image_name, pan_format_i, segments_info_, output_dir, video_id):
            original_image = Image.open(image_name).convert("RGBA")
            if 'simstation' in image_name:
                original_image = original_image.resize((2048, 1536))
            panoptic_seg = Image.fromarray(pan_format_i).convert("RGBA")
            panoptic_seg_resized = panoptic_seg.resize(original_image.size, resample=Image.Resampling.NEAREST)

            # Save panoptic segmentation
            panoptic_file_name = os.path.join(output_dir, 'pan_pred', video_id, image_name.split('/')[-1].split('.')[0] + '.png')
            panoptic_seg_resized.save(panoptic_file_name)  # necessary

            # Blended image
            blended_image = Image.blend(original_image, panoptic_seg_resized, alpha=0.6).convert("RGB")
            blended_file_name = os.path.join(output_dir, 'pan_pred', video_id, 'blend', image_name.split('/')[-1].split('.')[0] + '.jpg')
            blended_image.save(blended_file_name)

            # Annotations
            annotations = {"segments_info": [item for item in segments_info_ if item is not None], "file_name": image_name.split('/')[-1]}  # necessary
            return annotations

        os.makedirs(os.path.join(self._output_dir, 'pan_pred', video_id, 'blend'), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, 'pan_pred', video_id), exist_ok=True)
        annotations = []
        with ThreadPoolExecutor(max_workers=64) as executor:
            # Using map to maintain order
            results = executor.map(_image_save_helper, image_names, pan_format, list(zip(*segments_infos_)), [self._output_dir] * len(image_names), [video_id] * len(image_names))
            for result in list(results):
                annotations.append(result)

        self._predictions.append({'annotations': annotations, 'video_id': video_id})

    def _evaluate(self):
        """
        save jsons
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            file_path = os.path.join(self._output_dir, 'pred.json')
            with open(file_path, 'w') as f:
                json.dump({'annotations': predictions}, f)
        return {}

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        dataset_split = 'train'
        if '_train_small' in self._metadata.name:
            dataset_split = 'train_small'
        elif '_train_mini' in self._metadata.name:
            dataset_split = 'train_mini'
        elif '_val' in self._metadata.name:
            dataset_split = 'val'
        elif '_test' in self._metadata.name:
            dataset_split = 'test'
        if self._output_dir:
            file_path = os.path.join(self._output_dir, f'pred_{dataset_split}.json')
            with open(file_path, 'w') as f:
                json.dump({'annotations': predictions}, f)

        # Load the ground truth data
        if 'mmor' in self._metadata.name:
            with open(f'datasets/mmor_ground_truth_{dataset_split}.json', 'r') as f:
                gt_jsons = json.load(f)
        elif '4dor' in self._metadata.name:
            with open(f'datasets/4dor_ground_truth_{dataset_split}.json', 'r') as f:
                gt_jsons = json.load(f)
        elif 'hybridor' in self._metadata.name:
            with open(f'datasets/hybridor_ground_truth_{dataset_split}.json', 'r') as f:
                gt_jsons = json.load(f)
        else:
            raise Exception(f'Unknown dataset {self._metadata.name}')

        pred_annos = predictions
        pred_j = {}
        for p_a in pred_annos:
            pred_j[p_a['video_id']] = p_a['annotations']
        gt_annos = gt_jsons['annotations']
        gt_j = {}
        for g_a in gt_annos:
            gt_j[g_a['video_id']] = g_a['annotations']

        categories = gt_jsons['categories']
        categories = {el['id']: el for el in categories}

        # Prepare the data for VPQ computation
        gt_pred_split = []
        for video_data in gt_jsons['videos']:
            video_id = video_data['video_id']
            video_folder = video_data['video_folder']
            gt_images = video_data['images']

            gt_js = gt_j[video_id]
            pred_js = pred_j[video_id]

            gt_pans, pred_pans = [], []  # only as paths
            for image in gt_images:
                camera_idx = int(image['file_name'].split('_')[0].replace('camera', ''))
                if '4DOR' in video_id:
                    gt_ann_path = Path('../4D-OR_data') / video_folder / f'panoptic_seg_{camera_idx}_for_val' / image['file_name'].replace('.jpg', '.png')
                else:
                    if 'simstation' in video_id:
                        gt_ann_path = Path('../MM-OR_data') / video_folder / f'panoptic_seg_simstation{camera_idx}_for_val' / image['file_name'].replace('.jpg', '.png')
                    else:
                        gt_ann_path = Path('../MM-OR_data') / video_folder / f'panoptic_seg_{camera_idx}_for_val' / image['file_name'].replace('.jpg', '.png')
                pred_ann_path = Path(self._output_dir) / 'pan_pred' / video_id / image['file_name'].replace('.jpg', '.png')
                gt_pans.append(gt_ann_path)
                pred_pans.append(pred_ann_path)

            gt_pred_split.append(list(zip(gt_js, pred_js, gt_pans, pred_pans, gt_images)))

        # Compute VPQ
        vpq_all, vpq_thing, vpq_stuff, all_results = [], [], [], []
        # for nframes in [1]:  # faster version.
        for nframes in [4, 8]:
            gt_pred_split_ = copy.deepcopy(gt_pred_split)
            vpq_all_, vpq_thing_, vpq_stuff_, results = vpq_compute_parallel(
                gt_pred_split_, categories, nframes, self._output_dir, num_processes=12
            )  # adjust num_processes to 1 if want to debug properly
            vpq_all.append(vpq_all_)
            vpq_thing.append(vpq_thing_)
            vpq_stuff.append(vpq_stuff_)
            all_results.append(results)

            if nframes == 1 or True:
                # Log per class results to WandB
                per_class_results_with_names = {}
                for cat_id, category in categories.items():
                    cat_name = category['name']
                    cat_pq = results['per_class'][cat_id]['pq']  # between 0 and 1
                    cat_precision = results['per_class'][cat_id]['precision']
                    cat_recall = results['per_class'][cat_id]['recall']
                    per_class_results_with_names[f'{dataset_split}_VPQ/{cat_name}_PQ'] = cat_pq
                    per_class_results_with_names[f'{dataset_split}_Prec/{cat_name}'] = cat_precision
                    per_class_results_with_names[f'{dataset_split}_Rec/{cat_name}'] = cat_recall
                wandb.log(per_class_results_with_names)
                print(per_class_results_with_names)

        vpq_results = {
            f"{dataset_split}_vpq_all": sum(vpq_all) / len(vpq_all),
            f"{dataset_split}_prec": sum(results['All']['precision'] for results in all_results) / len(all_results),
            f"{dataset_split}_rec": sum(results['All']['recall'] for results in all_results) / len(all_results),
        }

        # Log overall results to WandB
        wandb.log(vpq_results)

        # Return the results
        return vpq_results
