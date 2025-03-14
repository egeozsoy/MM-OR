# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import tempfile

import torch.distributed as dist
import wandb

from datasets.hybrid_dataset import get_hybridor_segmentation_dataset_train, get_hybridor_segmentation_dataset_val, get_hybridor_segmentation_dataset_train_mini, \
    get_hybridor_segmentation_dataset_train_small, get_hybridor_segmentation_dataset_test
from datasets.mmor_dataset import sorted_classes, TRACK_TO_METAINFO, get_mmor_segmentation_dataset_train, get_mmor_segmentation_dataset_train_small, get_mmor_segmentation_dataset_val, \
    get_mmor_segmentation_dataset_train_mini, get_mmor_segmentation_dataset_test
from datasets.or4d_dataset import get_4dor_segmentation_dataset_train, get_4dor_segmentation_dataset_val, get_4dor_segmentation_dataset_train_mini, \
    get_4dor_segmentation_dataset_train_small, get_4dor_segmentation_dataset_test


def initialize_distributed_group_if_needed():
    if not dist.is_initialized():
        # Use a temporary file as the init_method
        with tempfile.NamedTemporaryFile() as temp_file:
            init_method = f'file://{temp_file.name}'
            dist.init_process_group(backend='gloo', init_method=init_method, rank=0, world_size=1)


# Call this at the start of your script
initialize_distributed_group_if_needed()

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
from torchinfo import summary
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    HookBase,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# Models
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from dvis_Plus import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    YTVISEvaluator,
    VPSEvaluator,
    VSSEvaluator,
    add_minvis_config,
    add_dvis_config,
    add_ctvis_config,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
)


class WandbLoggingHook(HookBase):
    def __init__(self, log_period=20):
        self.log_period = log_period

    def after_step(self):
        current_iter = self.trainer.iter
        if current_iter % self.log_period == 0:
            # Extract metrics from the trainer's storage
            metrics_dict = self.trainer.storage.latest()
            interested_metrics = ['total_loss', 'loss_dice', 'loss_ce', 'lr']
            clean_metrics = {k: v[0] for k, v in metrics_dict.items() if k in interested_metrics}
            wandb.log(clean_metrics, step=current_iter)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(WandbLoggingHook(log_period=20))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        evaluator_dict = {'vis': YTVISEvaluator, 'vss': VSSEvaluator, 'vps': VPSEvaluator}
        assert cfg.MODEL.MASK_FORMER.TEST.TASK in evaluator_dict.keys()
        return evaluator_dict[cfg.MODEL.MASK_FORMER.TEST.TASK](dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        assert len(cfg.DATASETS.DATASET_RATIO) == len(cfg.DATASETS.TRAIN) == \
               len(cfg.DATASETS.DATASET_NEED_MAP) == len(cfg.DATASETS.DATASET_TYPE)
        mappers = []
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
            'image_instance': CocoClipDatasetMapper,
        }
        for d_i, (dataset_name, dataset_type, dataset_need_map) in \
                enumerate(zip(cfg.DATASETS.TRAIN, cfg.DATASETS.DATASET_TYPE, cfg.DATASETS.DATASET_NEED_MAP)):
            if dataset_type not in mapper_dict.keys():
                raise NotImplementedError
            _mapper = mapper_dict[dataset_type]
            mappers.append(
                _mapper(cfg, is_train=True, is_tgt=not dataset_need_map, src_dataset_name=dataset_name, )
            )
        assert len(mappers) > 0, "No dataset is chosen!"

        if len(mappers) == 1:
            mapper = mappers[0]
            return build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])
        else:
            loaders = [
                build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN)
            ]
            combined_data_loader = build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)
            return combined_data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, dataset_type):
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
        }
        if dataset_type not in mapper_dict.keys():
            raise NotImplementedError
        mapper = mapper_dict[dataset_type](cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            dataset_type = cfg.DATASETS.DATASET_TYPE_TEST[idx]
            data_loader = cls.build_test_loader(cfg, dataset_name, dataset_type)

            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    add_dvis_config(cfg)
    add_ctvis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="minvis")

    wandb.init(project='MM-OR-SEG', config=cfg)
    wandb.run.name = cfg.OUTPUT_DIR.split('/')[-1]  # Set the name of the run to the output directory name
    return cfg


def main(args):
    cfg = setup(args)
    from detectron2.data import DatasetCatalog, MetadataCatalog
    if 'mmor' in cfg['DATASETS']['TRAIN'][0]:
        DatasetCatalog.register("mmor_panoptic_train", get_mmor_segmentation_dataset_train)
        DatasetCatalog.register("mmor_panoptic_train_small", get_mmor_segmentation_dataset_train_small)
        DatasetCatalog.register("mmor_panoptic_train_mini", get_mmor_segmentation_dataset_train_mini)
        DatasetCatalog.register("mmor_panoptic_val", get_mmor_segmentation_dataset_val)
        DatasetCatalog.register("mmor_panoptic_test", get_mmor_segmentation_dataset_test)
        train_meta = MetadataCatalog.get("mmor_panoptic_train")
        train_small_meta = MetadataCatalog.get("mmor_panoptic_train_small")
        train_mini_meta = MetadataCatalog.get("mmor_panoptic_train_mini")
        val_meta = MetadataCatalog.get("mmor_panoptic_val")
        test_meta = MetadataCatalog.get("mmor_panoptic_test")
    elif '4dor' in cfg['DATASETS']['TRAIN'][0]:
        DatasetCatalog.register("4dor_panoptic_train", get_4dor_segmentation_dataset_train)
        DatasetCatalog.register("4dor_panoptic_train_small", get_4dor_segmentation_dataset_train_small)
        DatasetCatalog.register("4dor_panoptic_train_mini", get_4dor_segmentation_dataset_train_mini)
        DatasetCatalog.register("4dor_panoptic_val", get_4dor_segmentation_dataset_val)
        DatasetCatalog.register("4dor_panoptic_test", get_4dor_segmentation_dataset_test)
        train_meta = MetadataCatalog.get("4dor_panoptic_train")
        train_small_meta = MetadataCatalog.get("4dor_panoptic_train_small")
        train_mini_meta = MetadataCatalog.get("4dor_panoptic_train_mini")
        val_meta = MetadataCatalog.get("4dor_panoptic_val")
        test_meta = MetadataCatalog.get("4dor_panoptic_test")
    elif 'hybridor' in cfg['DATASETS']['TRAIN'][0]:
        DatasetCatalog.register("hybridor_panoptic_train", get_hybridor_segmentation_dataset_train)
        DatasetCatalog.register("hybridor_panoptic_train_small", get_hybridor_segmentation_dataset_train_small)
        DatasetCatalog.register("hybridor_panoptic_train_mini", get_hybridor_segmentation_dataset_train_mini)
        DatasetCatalog.register("hybridor_panoptic_val", get_hybridor_segmentation_dataset_val)
        DatasetCatalog.register("hybridor_panoptic_test", get_hybridor_segmentation_dataset_test)
        train_meta = MetadataCatalog.get("hybridor_panoptic_train")
        train_small_meta = MetadataCatalog.get("hybridor_panoptic_train_small")
        train_mini_meta = MetadataCatalog.get("hybridor_panoptic_train_mini")
        val_meta = MetadataCatalog.get("hybridor_panoptic_val")
        test_meta = MetadataCatalog.get("hybridor_panoptic_test")
    else:
        raise Exception(f'Unknown dataset type: {cfg["DATASETS"]["TRAIN"][0]}')

    for meta in [train_meta, train_small_meta, train_mini_meta, val_meta, test_meta]:
        meta.set(thing_classes=sorted_classes)
        meta.set(thing_dataset_id_to_contiguous_id={i: i for i in range(len(sorted_classes))})
        meta.set(stuff_classes=[])
        meta.set(ignore_label=255)
        meta.set(stuff_dataset_id_to_contiguous_id={})
        meta.set(panoptic_json='')
        meta.set(categories={i: {'id': i, 'name': name, 'isthing': 1, 'color': TRACK_TO_METAINFO[name]['color']} for i, name in enumerate(sorted_classes)})

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    summary(trainer.model, col_names=['num_params', 'trainable'])
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
