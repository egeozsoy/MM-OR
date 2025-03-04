import os

os.environ["WANDB_DIR"] = os.path.abspath("wandb")
os.environ["TMPDIR"] = os.path.abspath("wandb")

import warnings

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset
from scene_graph_prediction.scene_graph_helpers.model.scene_graph_prediction_model import ModelWrapper


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def load_checkpoint_data(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}


def update_checkpoint_data(file_path, model_name, checkpoint_id, wandb_run_id=None):
    data = load_checkpoint_data(file_path)
    if model_name not in data:
        data[model_name] = {"checkpoints": [], "wandb_run_id": wandb_run_id}
    if checkpoint_id not in data[model_name]["checkpoints"]:
        data[model_name]["checkpoints"].append(checkpoint_id)
    if wandb_run_id:
        data[model_name]["wandb_run_id"] = wandb_run_id
    with open(file_path, 'w') as file:
        json.dump(data, file)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)
    mode = 'evaluate'  # can be evaluate/infer/eval_all # TODO switch to eval_all for evaluation of all the checkpoints, and infer for inference
    shuffle = True
    batch_size = 8
    if 'temporality' in config and config['temporality'] == 'PRED':
        print('Online temporality. Setting batch size to 1 and not shuffling')
        shuffle = False
        batch_size = 1

    name = args.config.replace('.json', '')

    if mode == 'evaluate':
        print(f'Model path: {args.model_path}')
        eval_dataset = ORDataset(config, 'test')
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
        model = ModelWrapper(config, relationNames=eval_dataset.relations, classNames=eval_dataset.classes, model_path=args.model_path)
        model.validate(eval_loader)
    elif mode == 'eval_all':
        print('Evaluating all checkpoints')

        evaluated_file = 'evaluated_checkpoints.json'
        checkpoint_data = load_checkpoint_data(evaluated_file)
        model_path = Path(args.model_path)
        model_name = model_path.name
        if 'temporality' in config and config['temporality'] == 'PRED':
            print('Modifying model name for temporality')
            model_name += '_pred_temporality'
        eval_every_n_checkpoints = 4 if not 'temporality' in config else 8  # temporality is costly to evaluate, so we do it less often
        wandb_run_id = checkpoint_data.get(model_name, {}).get("wandb_run_id", None)
        logger = pl.loggers.WandbLogger(project='mmor_evals', name=model_name, save_dir='logs', offline=False, id=wandb_run_id)
        train_dataset = ORDataset(config, 'train')
        eval_dataset = ORDataset(config, 'val')
        eval_dataset_for_train = ORDataset(config, 'train')
        # always eval last checkpoint
        checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))
        print(checkpoints)
        checkpoint_idx = 0
        while checkpoint_idx < len(checkpoints):
            checkpoint = checkpoints[checkpoint_idx]
            if checkpoint_idx % eval_every_n_checkpoints != 0 and checkpoint_idx != len(checkpoints) - 1:
                print(f'Skipping checkpoint: {checkpoint}')
                checkpoint_idx += 1
                continue
            if checkpoint_idx == 0 and 'continue' not in model_name:
                checkpoint_idx += 1
                print(f'Skipping checkpoint: {checkpoint}')
                checkpoint_idx += 1
                continue
            checkpoint_id = int(checkpoint.name.split('-')[-1])
            if model_name in checkpoint_data and checkpoint_id in checkpoint_data[model_name]["checkpoints"]:
                print(f'Checkpoint {checkpoint_id} for model {model_name} already evaluated. Skipping.')
                checkpoint_idx += 1
                continue
            print(f'Evaluating checkpoint: {checkpoint}...')
            torch.cuda.empty_cache()
            train_loader = DataLoader(eval_dataset_for_train, batch_size=batch_size, shuffle=shuffle, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
            model = ModelWrapper(config, relationNames=train_dataset.relations, classNames=train_dataset.classes, model_path=str(checkpoint))
            model.validate(train_loader, limit_val_batches=1000 // batch_size, logging_information={'split': 'train', "logger": logger, "checkpoint_id": checkpoint_id})
            model.validate(eval_loader, logging_information={'split': 'val', "logger": logger, "checkpoint_id": checkpoint_id})
            # cleanup before next run
            del model
            update_checkpoint_data(evaluated_file, model_name, checkpoint_id, logger.experiment.id)
            checkpoint_idx += 1
            checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))  # update checkpoints in case new ones were added

    elif mode == 'infer':
        print('INFER')
        print(f'Model path: {args.model_path}')
        infer_split = 'test'
        train_dataset = ORDataset(config, 'train')
        eval_dataset = ORDataset(config, infer_split)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
        model = ModelWrapper(config, relationNames=train_dataset.relations, classNames=train_dataset.classes, model_path=args.model_path)
        results = model.infer(eval_loader)
        # results should be batch scan id -> list of relations
        output_name = f'scan_relations_{name}_{infer_split}.json'
        with open(output_name, 'w') as f:
            json.dump(results, f)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    main()
