"""Build the VPQ ground-truth json, and score saved panoptic predictions without the GPU.

`train_net_video.py --eval-only` scores with VPSEvaluator, which reads
`datasets/<dataset>_ground_truth_<split>.json`. That file is not part of the release: build it
once per split before evaluating, with the same code that loads the data:

    python eval_from_preds.py --build-gt-only --dataset hybridor_panoptic_test

The evaluator first saves the predictions (`<OUTPUT_DIR>/inference/pred_<split>.json` and
`pan_pred/`), then computes VPQ on the CPU, which takes about two hours for the HybridOR test
split while the model stays on the GPU. To score saved predictions again, or separately from
the inference:

    python eval_from_preds.py <OUTPUT_DIR>/inference --dataset hybridor_panoptic_test

Run from the panoptic_segmentation directory (the data is resolved as ../MM-OR_data and
../4D-OR_data). VPQ_NUM_PROCESSES sets the size of the scoring pool (default 12).
Results are printed and written to <OUTPUT_DIR>/inference/vpq_<split>.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb
from detectron2.data import MetadataCatalog

from datasets.hybrid_dataset import HybridORSegmentationDataset
from datasets.mmor_dataset import MMORSegmentationDataset
from datasets.or4d_dataset import OR4DSegmentationDataset
from datasets.or_dataset import TRACK_TO_METAINFO, create_ground_truth_json, sorted_classes
from dvis_Plus.data_video.vps_eval import VPSEvaluator


def build_dataset(name, split):
    kw = dict(max_video_length=200, overlap=3, split=split, use_interpolated=False)  # as get_*_segmentation_dataset()
    if name.startswith('hybridor'):
        return HybridORSegmentationDataset({'mmor': MMORSegmentationDataset(**kw), '4dor': OR4DSegmentationDataset(**kw)}, **kw)
    if name.startswith('mmor'):
        return MMORSegmentationDataset(**kw)
    if name.startswith('4dor'):
        return OR4DSegmentationDataset(**kw)
    raise ValueError(f'unknown dataset {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inference_dir', nargs='?', help='<OUTPUT_DIR>/inference of an --eval-only run')
    ap.add_argument('--dataset', default='hybridor_panoptic_test', help='{hybridor,mmor,4dor}_panoptic_{train,val,test}')
    ap.add_argument('--build-gt-only', action='store_true', help='only build the ground-truth json')
    args = ap.parse_args()
    if not args.build_gt_only and not args.inference_dir:
        ap.error('inference_dir is required unless --build-gt-only is given')

    prefix, split = args.dataset.split('_')[0], args.dataset.rsplit('_', 1)[1]
    if split not in ('train', 'val', 'test'):
        ap.error(f'unsupported dataset {args.dataset}')
    gt_path = f'datasets/{prefix}_ground_truth_{split}.json'
    if not os.path.exists(gt_path):
        print(f'building {gt_path}', flush=True)
        create_ground_truth_json(build_dataset(args.dataset, split), gt_path)
    if args.build_gt_only:
        return

    wandb.init(mode='disabled')
    meta = MetadataCatalog.get(args.dataset)  # as in train_net_video.main()
    meta.set(thing_classes=sorted_classes)
    meta.set(thing_dataset_id_to_contiguous_id={i: i for i in range(len(sorted_classes))})
    meta.set(stuff_classes=[])
    meta.set(ignore_label=255)
    meta.set(stuff_dataset_id_to_contiguous_id={})
    meta.set(panoptic_json='')
    meta.set(categories={i: {'id': i, 'name': n, 'isthing': 1, 'color': TRACK_TO_METAINFO[n]['color']} for i, n in enumerate(sorted_classes)})
    evaluator = VPSEvaluator(args.dataset, distributed=False, output_dir=args.inference_dir)
    with open(os.path.join(args.inference_dir, f'pred_{split}.json')) as f:
        evaluator._predictions = json.load(f)['annotations']
    results = evaluator.evaluate()
    print('RESULT', json.dumps(results), flush=True)
    with open(os.path.join(args.inference_dir, f'vpq_{split}.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
