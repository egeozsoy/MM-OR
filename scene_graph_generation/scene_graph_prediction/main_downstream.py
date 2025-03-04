import os

from scene_graph_prediction.main import config_loader

os.environ["WANDB_DIR"] = os.path.abspath("wandb")
os.environ["TMPDIR"] = os.path.abspath("wandb")

import warnings

warnings.filterwarnings('ignore')
import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset
from scene_graph_prediction.scene_graph_helpers.model.downstream_prediction_model import DownstreamPredictionModelWrapper



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)
    batch_size = 8
    task = 'robot_phase' # can be one of: ['next_action', 'robot_phase', 'sterility_breach']. While the model is trained for all at the same time, we can evaluate them separately

    print(f'Model path: {args.model_path}, task: {task}')
    eval_dataset = ORDataset(config, 'test', load_4dor=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
    scene_graph_path_pred = 'scan_relations_mv_learned_temporal_pred_test.json'
    model = DownstreamPredictionModelWrapper(config, model_path=args.model_path, path_to_scene_graphs=scene_graph_path_pred, task=task)
    model.validate(eval_loader)

if __name__ == '__main__':
    main()
