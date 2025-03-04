import argparse
import random
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path
from random import shuffle

import json_tricks as json  # Allows to load integers etc. correctly
import numpy as np
import pytorch_lightning as pl
import transformers
from tqdm import tqdm

from helpers.configurations import TRACKER_OBJECT_MAP
from scene_graph_prediction.llava_helpers.scene_graph_converters import parse_llava_sg, llava_sg_to_surgery_sg, surgery_sg_to_memory_str
from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset

warnings.filterwarnings('ignore')


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def scene_graph_to_string(scene_graph):
    '''
    Scene graph is a list of relations in the form of (subject, relation, object)
    '''
    out = '<SG> '
    for (subject, object, relation) in scene_graph:
        subject = subject.replace('_', ' ').lower()
        object = object.replace('_', ' ').lower()
        out += f'{subject},{object},{relation}; '
    # remove the last ";" and add the end token.
    out = out.rstrip('; ') + ' </SG>'
    return out


def apply_template(image_paths, scene_graph, timepoint, sample_id, pc, audio, raw_audio, robot_metadata, tracker_metadata, speech_transcript, segmasks):
    human_prompt = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, mps, patient, student, instrument table, operating table, secondary table, anesthesia equipment, c_arm, mako_robot, monitor, mps_station, tracker, drape, drill, hammer, saw, instrument]. Predicates: [assisting, calibrating, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, scanning, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    if robot_metadata is not None:
        # this will potentially have multiple parts, but both will be added to the original sample in the human prompt
        with open(robot_metadata, 'r') as f:
            robot_metadata = json.load(f)
        robot_metadata_str = ''
        for key, value in sorted(robot_metadata.items()):
            robot_metadata_str += f'{value["type"]}: {value["template_name"]}, '
        robot_metadata_str = robot_metadata_str.rstrip(', ')
        # insert this into the human prompt
        human_prompt = human_prompt.replace('Entities: ', f'<robot_metadata_start>: {robot_metadata_str} <robot_metadata_end>. Entities: ')
    if tracker_metadata is not None:
        unique_id_dicts = tracker_metadata['unique_id_dicts']
        tracker_metadata_str = ''
        for unique_id_dict in unique_id_dicts:
            tool_name = TRACKER_OBJECT_MAP[unique_id_dict['unique_id']]
            tool_state = unique_id_dict['button_state']
            tool_translation = np.asarray(unique_id_dict['Translation']).astype(int)
            tool_translation_str = ' '.join(tool_translation.astype(str))
            tool_rotation = np.asarray(unique_id_dict['euler_rot']).astype(int)
            tool_rotation_str = ' '.join(tool_rotation.astype(str))
            tracker_metadata_str += f'{tool_name}: state {tool_state}, translation {tool_translation_str}, euler angles {tool_rotation_str}; '
        tracker_metadata_str = tracker_metadata_str.rstrip('; ')
        # insert this into the human prompt
        human_prompt = human_prompt.replace('Entities: ', f'<tracker_metadata_start>: {tracker_metadata_str} <tracker_metadata_end>. Entities: ')
    if speech_transcript is not None:
        with open(speech_transcript, 'r') as f:
            speech_transcript = json.load(f)
        speech_transcript_str = speech_transcript['text']
        # insert this into the human prompt
        human_prompt = human_prompt.replace('Entities: ', f'<speech_transcript_start>: {speech_transcript_str} <speech_transcript_end>. Entities: ')

    sample = {'id': sample_id,
              'timepoint': timepoint,
              'vis_knowledge_paths': None,
              "conversations": [
                  {
                      "from": "human",
                      "value": f"<image>\n{human_prompt}"
                  },
                  {
                      "from": "gpt",
                      "value": scene_graph
                  },
              ]
              }
    if len(image_paths) > 0:
        sample['image'] = [str(image_path.absolute()) for image_path in image_paths]
    if len(segmasks) > 0:
        sample['segmasks'] = [str(segmask.absolute()) for segmask in segmasks]
    if pc is not None:
        sample['pc'] = str(pc.absolute())
    if audio is not None:
        sample['audio'] = str(audio.absolute())
    if raw_audio is not None:
        sample['raw_audio'] = str(raw_audio.absolute())

    return sample


def generate_finetuning_samples_from_dataset(dataset, n_permutations=1, mixed_modalities=False, MAKE_SIMPLER=False):
    samples = []
    for index in tqdm(range(len(dataset)), desc='Generating samples'):
        sample = dataset[index]
        sample, multimodal_data = sample['sample'], deepcopy(sample['multimodal_data'])
        # get length of multimodal data. As multimodal data is a dictionary with multiple keys, we consider the length to be the longest list
        multimodal_data_len = max([len(multimodal_data[key]) for key in multimodal_data])
        if multimodal_data_len == 0:
            continue
        sample_id = sample['sample_id']
        frame_id = sample['frame_id']
        image_paths = []
        azure_image_paths = multimodal_data['azure']
        simstation_image_paths = multimodal_data['simstation'] if 'simstation' in multimodal_data else []
        trackercam_image_paths = multimodal_data['trackercam'] if 'trackercam' in multimodal_data else []
        dataset_type = '4D-OR' if '4DOR' in sample_id else 'MM-OR'
        azure_views_to_use = (2, 1, 3, 5) if '4DOR' in sample_id else (1, 4, 5, 2, 3)
        limited_azure_views_to_use = (2, 1, 3, 5) if '4DOR' in sample_id else (1, 4, 5)
        simstation_views_to_use = (2, 0, 1, 3)
        if len(azure_image_paths) > 0:
            azure_image_paths = [azure_image_paths[view_idx - 1] for view_idx in azure_views_to_use]
            image_paths.extend(azure_image_paths)
        if len(simstation_image_paths) > 0:
            simstation_image_paths = [simstation_image_paths[view_idx] for view_idx in simstation_views_to_use]
            image_paths.extend(simstation_image_paths)
        if len(trackercam_image_paths) > 0:
            image_paths.extend(trackercam_image_paths[:1])

        relations = sample['relationships']
        raw_audio = multimodal_data['raw_audio'][0] if 'raw_audio' in multimodal_data else None
        pc = multimodal_data['pc'][0] if 'pc' in multimodal_data else None
        segmasks = multimodal_data['segmasks'] if 'segmasks' in multimodal_data else []

        if mixed_modalities:  # we will fetch from different timepoints using the key: similar_samples
            keys = set(multimodal_data.keys()).intersection({'audio', 'robot_metadata', 'tracker', 'speech_transcript'})
            for key in keys:
                # first fetch a similar sample
                similar_sample = random.choice(sample['similar_samples']) if len(sample['similar_samples']) > 0 else None
                if similar_sample is None:
                    continue
                similar_sample_str, similar_sample_idx = similar_sample['sample_str'], similar_sample['sample_idx']
                similar_sample = dataset[similar_sample_idx]
                similar_sample, similar_multimodal_data = similar_sample['sample'], similar_sample['multimodal_data']
                # use this multimodal information
                if key in similar_multimodal_data:
                    multimodal_data[key] = similar_multimodal_data[key]

        audio = multimodal_data['audio'][0] if 'audio' in multimodal_data else None
        robot_metadata = multimodal_data['robot_metadata'][0] if 'robot_metadata' in multimodal_data else None
        tracker_metadata = multimodal_data['tracker'][0] if 'tracker' in multimodal_data else None
        speech_transcript = multimodal_data['speech_transcript'][0] if 'speech_transcript' in multimodal_data else None

        for permutation_idx in range(n_permutations):
            shuffle(relations)  # order should be random
            scene_graph_string = scene_graph_to_string(relations)
            sample = apply_template(image_paths, scene_graph_string, timepoint=int(frame_id), sample_id=sample_id, pc=pc, audio=audio, raw_audio=raw_audio, robot_metadata=robot_metadata,
                                    tracker_metadata=tracker_metadata, speech_transcript=speech_transcript, segmasks=segmasks)

            samples.append(sample)

    return samples


def main():
    N_PERM = 20  # 10 for quick experiments, 30 probably already optimal.
    ADD_TEMPORAL = False  # set to True to add temporal information, necessary for training the temporal model
    WITH_TEMPORAL_AUG = True
    DROP_HISTORY = 0.5  # either False or float
    MIXED_MODALITIES = True
    SPLIT = 'train'
    MAKE_SIMPLER = False
    NAME = f'{SPLIT}_{N_PERM}perm_{ADD_TEMPORAL}temp_{WITH_TEMPORAL_AUG}tempaug_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS'
    if DROP_HISTORY is not False and DROP_HISTORY > 0.01:
        NAME += f'_drophistory{DROP_HISTORY}'
    if MIXED_MODALITIES:
        NAME += '_MIXED'
    if MAKE_SIMPLER:
        NAME += '_SIMPLER'

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

    samples = generate_finetuning_samples_from_dataset(dataset, n_permutations=N_PERM, mixed_modalities=MIXED_MODALITIES, MAKE_SIMPLER=MAKE_SIMPLER)
    # Load the tokenizer which will be used
    # val_samples = generate_finetuning_samples_from_dataset(val_dataset)
    # Also calculate the corresponding word frequencies
    token_freq = Counter()
    longest_sample = -1
    for sample in tqdm(samples, desc='Calculating token frequencies'):
        for conversation in sample['conversations']:
            if conversation['from'] == 'gpt':
                tokenized = tokenizer.tokenize(conversation['value'])
                token_freq.update(tokenized)
                longest_sample = max(longest_sample, len(tokenized))

    # randomly shuffle the samples
    shuffle(samples)

    if ADD_TEMPORAL:
        print('Adding temporal information...')
        take_to_history = {}
        take_timepoint_to_memory_str = {}

        # Collect unique take names from samples
        take_names = set(sample['id'].rsplit('_', 1)[0] for sample in samples)
        for take_name in tqdm(take_names, desc='Processing takes for temporal information'):
            # Collect all samples for this take
            take_scene_graphs = [elem for elem in samples if elem['id'].rsplit('_', 1)[0] == take_name]
            # Remove duplicates based on timepoint
            take_scene_graphs = list({elem['timepoint']: elem for elem in take_scene_graphs}.values())
            # Sort by timepoint
            take_scene_graphs = sorted(take_scene_graphs, key=lambda x: x['timepoint'])
            take_scene_graphs_reformatted = []
            for take_scene_graph in take_scene_graphs:
                # Parse the scene graph from the assistant's response
                scene_graph = parse_llava_sg(take_scene_graph['conversations'][1]['value'])
                take_scene_graphs_reformatted.append({'timepoint_idx': take_scene_graph['timepoint'], 'scene_graph': scene_graph})
            # Convert to surgery-specific triplets
            surgery_sg_triplets = llava_sg_to_surgery_sg(take_scene_graphs_reformatted, entity_of_interest=None, IRRELEVANT_PREDS=['closeto', 'closeTo'])
            # Save the triplets for debugging or future use
            with open(f'data/llava_samples/surgery_sg_{take_name}.json', 'w') as f:
                json.dump(surgery_sg_triplets, f)
            take_to_history[take_name] = surgery_sg_triplets

        llava_scene_graphs_with_history = []
        for llava_scene_graph in tqdm(samples, desc='Augmenting samples with temporal information'):
            take_name = llava_scene_graph['id'].rsplit('_', 1)[0]
            surgery_sg_triplets = take_to_history[take_name]
            timepoint = llava_scene_graph['timepoint']
            # Filter triplets up to the current timepoint
            surgery_sg_triplets_up_to_current = [elem for elem in surgery_sg_triplets if elem[0] < timepoint]
            # Generate the memory string
            memory_str = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint)
            take_timepoint_to_memory_str[f'{take_name}_{timepoint}'] = memory_str
            input_prompt = llava_scene_graph['conversations'][0]['value']

            if WITH_TEMPORAL_AUG:
                p = random.random()
                if p < 0.5:
                    memory_str = None
                elif p < 0.666:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='short', DROP_HISTORY=DROP_HISTORY)
                elif p < 0.833:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='long', DROP_HISTORY=DROP_HISTORY)
                else:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint, TEMPORAL_STYLE='longshort', DROP_HISTORY=DROP_HISTORY)

            if memory_str is not None:
                input_prompt = input_prompt.replace('<image>\n', f'<image>\n<memory_start>: {memory_str}<memory_end>.\n')
            llava_scene_graph['conversations'][0]['value'] = input_prompt
            llava_scene_graphs_with_history.append(llava_scene_graph)

        samples = llava_scene_graphs_with_history

        with open(f'data/llava_samples/{NAME}_take_timepoint_to_memory_str.json', 'w') as f:
            json.dump(take_timepoint_to_memory_str, f)

    with open(f'data/llava_samples/{NAME}.json', 'w') as f:
        json.dump(samples, f, indent=4)

    # if SPLIT == 'train' and not ADD_TEMPORAL:
    #     with open(f'data/llava_samples/train_token_freqs_7b_{N_PERM}perm_{FREQUENCY_BETA}freqbeta.json', 'w') as f:
    #         json.dump(token_freq, f, indent=4)


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
