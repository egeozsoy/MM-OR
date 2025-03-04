import argparse
import random
import warnings
from random import shuffle

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import transformers
from tqdm import tqdm

from helpers.configurations import MMOR_DATA_ROOT_PATH
from scene_graph_prediction.llava_helpers.generate_dataset_format_for_llava import scene_graph_to_string, config_loader
from scene_graph_prediction.llava_helpers.scene_graph_converters import llava_sg_to_surgery_sg, surgery_sg_to_memory_str
from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset

warnings.filterwarnings('ignore')


def apply_template(current_sg, all_scene_graphs_till_now, timepoint, sample_id, task, answer):
    '''
    task: ['next_action', 'robot_phase', 'sterility_breach']
    '''
    assert task in ['next_action', 'robot_phase', 'sterility_breach']
    if task == 'next_action':
        human_prompt = 'Given the following scene graph memory representation, predict the next action. Do not provide a narrative or descriptive text.'
    elif task == 'robot_phase':
        human_prompt = 'Given the following scene graph memory representation, predict the robot phase. Do not provide a narrative or descriptive text.'
    elif task == 'sterility_breach':
        human_prompt = 'Given the following scene graph memory representation, predict the sterility breach. Do not provide a narrative or descriptive text.'
    else:
        raise ValueError(f'Unknown task {task}')

    # add both the memory_str (first), and the current_sg (second). Prepend the human prompt.
    scene_graph_context_information = f'<memory_start>: {all_scene_graphs_till_now}<memory_end>. {current_sg}.'
    human_prompt = f'{scene_graph_context_information} {human_prompt}'

    sample = {'id': sample_id,
              'timepoint': timepoint,
              "conversations": [
                  {
                      "from": "human",
                      "value": f"{human_prompt}"
                  },
                  {
                      "from": "gpt",
                      "value": answer
                  },
              ]
              }

    return sample


def generate_finetuning_samples_from_dataset(dataset, split, n_permutations=1, with_temporal_aug=False):
    samples = []
    take_timestamp_to_next_action_folder_path = MMOR_DATA_ROOT_PATH / 'take_timestamp_to_next_action'
    take_timestamp_to_robot_phase_path = MMOR_DATA_ROOT_PATH / 'take_timestamp_to_robot_phase'
    take_timestamp_to_sterility_breach_path = MMOR_DATA_ROOT_PATH / 'take_timestamp_to_sterility_breach'
    jsons_cache = {}

    # Collect unique take names from samples
    all_scene_graphs = []
    for index in tqdm(range(len(dataset)), desc='Generating samples'):
        sample = dataset[index]['sample']
        relations = sample['relationships']
        take_name = sample["take_name"]
        if '4DOR' in take_name:
            continue
        sample_id = sample["sample_id"]
        frame_id = sample["frame_id"]
        timepoint = int(frame_id)
        all_scene_graphs.append({'take_name': take_name, 'frame_id': frame_id, 'sample_id': sample_id, 'timepoint_idx': timepoint, 'scene_graph': relations})

    with open(f'all_scene_graphs_{split}_GT.json', 'w') as f:
        json.dump(all_scene_graphs, f)

    take_names = set(scene_graph['take_name'].rsplit('_', 1)[0] for scene_graph in all_scene_graphs)
    take_to_all_scene_graphs = {}
    for take_name in tqdm(take_names, desc='Processing takes for temporal information'):
        # Collect all samples for this take
        take_scene_graphs = [elem for elem in all_scene_graphs if elem['take_name'].rsplit('_', 1)[0] == take_name]
        # Remove duplicates based on timepoint
        take_scene_graphs = list({elem['timepoint_idx']: elem for elem in take_scene_graphs}.values())
        # Sort by timepoint
        take_scene_graphs = sorted(take_scene_graphs, key=lambda x: x['timepoint_idx'])
        # Convert to surgery-specific triplets
        surgery_sg_triplets = llava_sg_to_surgery_sg(take_scene_graphs, entity_of_interest=None, IRRELEVANT_PREDS=['closeto', 'closeTo'])
        take_to_all_scene_graphs[take_name] = surgery_sg_triplets

    for index in tqdm(range(len(dataset)), desc='Generating samples'):
        sample = dataset[index]['sample']
        take_name = sample["take_name"]
        if '4DOR' in take_name:
            continue
        take_name = take_name.rsplit('_', 1)[0]
        timepoint = int(sample["frame_id"])
        sample_id = sample["sample_id"]
        if take_name not in jsons_cache:
            jsons_cache[take_name] = {}
            with open(take_timestamp_to_next_action_folder_path / f'{take_name}.json', 'r') as f:
                jsons_cache[take_name]["next_action"] = json.load(f)
            with open(take_timestamp_to_robot_phase_path / f'{take_name}.json', 'r') as f:
                jsons_cache[take_name]["robot_phase"] = json.load(f)
            with open(take_timestamp_to_sterility_breach_path / f'{take_name}.json', 'r') as f:
                jsons_cache[take_name]["sterility_breach"] = json.load(f)

        next_action_json = jsons_cache[take_name]["next_action"]
        robot_phase_json = jsons_cache[take_name]["robot_phase"]
        sterility_breach_json = jsons_cache[take_name]["sterility_breach"]
        next_action = next_action_json[sample['frame_id']]
        robot_phase = robot_phase_json[sample['frame_id']]
        sterility_breach = sterility_breach_json[sample['frame_id']]

        current_sg = sample['relationships']  # definetely include this
        surgery_sg_triplets = take_to_all_scene_graphs[take_name]
        surgery_sg_triplets_up_to_current = [elem for elem in surgery_sg_triplets if elem[0] < timepoint]
        memory_str_full = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint)
        for permutation_idx in range(n_permutations):
            shuffle(current_sg)  # order should be random

            task = random.choice(['next_action', 'robot_phase', 'sterility_breach'])
            # if sterility_breach is empty, it is boring to sample from it. We should automatically "consider" switching
            while task == 'sterility_breach' and len(sterility_breach) == 0:
                keep_empty_sterility_breach = random.random() < 0.01
                if keep_empty_sterility_breach:
                    break
                task = random.choice(['next_action', 'robot_phase'])

            if task == 'next_action':
                if next_action is None or len(next_action) == 0:
                    answer = 'none'
                else:
                    next_action_str, next_action_in_seconds = next_action
                    answer = f'{next_action_str}: {next_action_in_seconds}'
            elif task == 'robot_phase':
                answer = robot_phase
            elif task == 'sterility_breach':
                if len(sterility_breach) == 0:
                    answer = 'No'
                else:
                    sterility_breaches = ''
                    for breach in sterility_breach:
                        sterility_breaches += f'{breach[0]} {breach[1]} {breach[2]}; '
                    if len(sterility_breaches) > 2:
                        sterility_breaches = sterility_breaches[:-2]
                    answer = f'Yes: {sterility_breaches}'

            if with_temporal_aug:
                p = random.random()
                if p < 0.1:
                    memory_str = None
                elif p < 0.25:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='short', DROP_HISTORY=0.5)
                elif p < 0.4:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='long', DROP_HISTORY=0.5)
                else:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='longshort', DROP_HISTORY=0.5)
            else:
                memory_str = memory_str_full

            scene_graph_string = scene_graph_to_string(current_sg)
            sample = apply_template(current_sg=scene_graph_string, all_scene_graphs_till_now=memory_str, timepoint=timepoint, sample_id=sample_id, task=task, answer=answer)
            # sample['raw_relations'] = tuple([tuple(e) for e in sorted(relations)])  # merge other indicators here as well.
            samples.append(sample)

    return samples


def main():
    N_PERM = 10  # 10 for quick experiments, 30 probably already optimal. 50 we could do but likely overkill.
    WITH_TEMPORAL_AUG = True
    DROP_HISTORY = 0.5  # either False or float
    SPLIT = 'train'
    NAME = f'downstream_task_{SPLIT}_{WITH_TEMPORAL_AUG}tempaug'
    if DROP_HISTORY is not False and DROP_HISTORY > 0.01:
        NAME += f'_drophistory{DROP_HISTORY}'
    print(f'Creating samples for LLAVA dataset with name {NAME}')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='mmor.json', help='configuration file name. Relative path under given path')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'liuhaotian/llava-v1.5-7b',
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )

    dataset = ORDataset(config, SPLIT)

    samples = generate_finetuning_samples_from_dataset(dataset, split=SPLIT, n_permutations=N_PERM, with_temporal_aug=WITH_TEMPORAL_AUG)
    # randomly shuffle the samples
    shuffle(samples)
    with open(f'data/llava_samples/{NAME}.json', 'w') as f:
        json.dump(samples, f, indent=4)


if __name__ == '__main__':
    """format of json (ultimately):
    1) id
    2) image(s) paths
    3) Prompt (formulately in multiple ways)
    4) Answer (formulately in multiple ways) (multiple orders)
    5) Prompt should include knowledge about different things
    
    Optionally: include augmentations, modifications in scene graph, prompt or both etc.
    """
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
