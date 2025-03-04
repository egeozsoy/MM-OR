# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from collections import defaultdict
from copy import deepcopy

import torch
from extract_metalabels_for_MMOR import STERILITY_BREACH_PREDICATES, STERILE_ENTITIES, UNSTERILE_ENTITIES
from sklearn.metrics import classification_report
from tqdm import tqdm

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX
from LLaVA.llava.conversation import SeparatorStyle, default_conversation
from LLaVA.llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.model.builder import load_pretrained_model
from helpers.configurations import MMOR_DATA_ROOT_PATH
from scene_graph_prediction.llava_helpers.generate_dataset_format_for_llava import scene_graph_to_string
from scene_graph_prediction.llava_helpers.scene_graph_converters import llava_sg_to_surgery_sg, surgery_sg_to_memory_str


def reformat_reference_scene_graphs(all_scene_graphs_pred):
    '''
    Predicted/Infered scene graphs look different, we will fix
    pred scene graphs: keys: 002_4DOR_000000, values: scene graph
    gt scene graphs: keys: take_name, frame_id, sample_id(same as the keys of pred scene graphs), timeppint_idx, scene graph.
    '''
    reformat_scene_graphs = []
    # start by somehow sorting the scene graphs
    all_scene_graphs_pred = sorted(all_scene_graphs_pred.items())
    for sample_id, scene_graph in all_scene_graphs_pred:
        take_name, frame_id = sample_id.rsplit('_', 1)
        timepoint_idx = int(frame_id)
        new_scene_graph = []
        for sub, pred, obj in scene_graph:
            sub = sub.replace(' ', '_')
            obj = obj.replace(' ', '_')
            new_scene_graph.append((sub, obj, pred))
        reformat_scene_graphs.append({'take_name': take_name, 'frame_id': frame_id, 'timepoint_idx': timepoint_idx, 'scene_graph': new_scene_graph})

    return reformat_scene_graphs


class DownstreamPredictionModelWrapper:
    def __init__(self, config, model_path, path_to_scene_graphs, task, model_base='liuhaotian/llava-v1.5-7b', load_8bit=False, load_4bit=False):
        self.config = config
        self.mconfig = config['MODEL']
        self.path_to_scene_graphs = path_to_scene_graphs
        self.lr = float(self.config['LR'])
        # evaluation metrics
        self.train_take_preds = defaultdict(list)
        self.train_take_gts = defaultdict(list)
        self.val_take_preds = defaultdict(list)
        self.val_take_gts = defaultdict(list)
        self.reset_metrics()

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name, load_8bit, load_4bit)
        self.model.config.mv_type = self.mconfig['mv_type']
        self.model.config.tokenizer_padding_side = "left"
        self.task = task
        assert self.task in ['next_action', 'robot_phase', 'sterility_breach']

        with open(path_to_scene_graphs, 'r') as f:
            all_scene_graphs = json.load(f)
            if not (isinstance(all_scene_graphs, list) and len(all_scene_graphs) > 0 and 'take_name' in all_scene_graphs[0].keys()):
                print('Reformatting reference scene graphs')
                all_scene_graphs = reformat_reference_scene_graphs(all_scene_graphs)

        take_names = set(scene_graph['take_name'].rsplit('_', 1)[0] for scene_graph in all_scene_graphs)
        self.take_to_all_scene_graphs = {}
        self.take_to_all_full_scene_graphs = {}
        for take_name in tqdm(take_names, desc='Processing takes for temporal information'):
            # Collect all samples for this take
            take_scene_graphs = [elem for elem in all_scene_graphs if elem['take_name'].rsplit('_', 1)[0] == take_name]
            # Remove duplicates based on timepoint
            take_scene_graphs = list({elem['timepoint_idx']: elem for elem in take_scene_graphs}.values())
            # Sort by timepoint
            take_scene_graphs = sorted(take_scene_graphs, key=lambda x: x['timepoint_idx'])
            # Convert to surgery-specific triplets
            surgery_sg_triplets = llava_sg_to_surgery_sg(take_scene_graphs, entity_of_interest=None, IRRELEVANT_PREDS=['closeto', 'closeTo'])
            self.take_to_all_scene_graphs[take_name] = surgery_sg_triplets
            self.take_to_all_full_scene_graphs[take_name] = {int(elem['frame_id']): elem['scene_graph'] for elem in take_scene_graphs}

        self.next_actions = ['bring in', 'prepare', 'clean', 'cut', 'drill', 'saw', 'hammer', 'cement', 'suture', 'scan', 'bring out', 'none']
        self.next_actions_to_idx = {action: idx for idx, action in enumerate(self.next_actions)}

        self.robot_phases = ['turn on', 'initial calibration by mps', 'dressing the robot, to make it sterile', 'install the saw by nurse', 'install base array by nurse', 'install calibration array',
                             'calibrate the robot by nurse', 'remove calibration array', 'install actual saw tip']
        self.robot_phases_to_idx = {phase: idx for idx, phase in enumerate(self.robot_phases)}

        self.sterility_breaches = ['no', 'yes']
        self.sterility_breaches_to_idx = {breach: idx for idx, breach in enumerate(self.sterility_breaches)}

        if self.task == 'next_action':
            self.downstream_classes = self.next_actions
            self.downstream_classes_to_idx = self.next_actions_to_idx
        elif self.task == 'robot_phase':
            self.downstream_classes = self.robot_phases
            self.downstream_classes_to_idx = self.robot_phases_to_idx
        elif self.task == 'sterility_breach':
            self.downstream_classes = self.sterility_breaches
            self.downstream_classes_to_idx = self.sterility_breaches_to_idx
        else:
            raise ValueError(f'Unknown task {self.task}')

    def forward(self, batch):
        batchsize = len(batch)
        all_prompts = []
        for elem in batch:
            conv = deepcopy(default_conversation)
            sample, multimodal_data = elem['sample'], elem['multimodal_data']
            take_name = sample["take_name"]
            if '4DOR' in take_name:
                continue
            take_name = take_name.rsplit('_', 1)[0]
            timepoint = int(sample["frame_id"])
            if self.task == 'next_action':
                human_prompt = 'Given the following scene graph memory representation, predict the next action. Do not provide a narrative or descriptive text.'
            elif self.task == 'robot_phase':
                human_prompt = 'Given the following scene graph memory representation, predict the robot phase. Do not provide a narrative or descriptive text.'
            elif self.task == 'sterility_breach':
                human_prompt = 'Given the following scene graph memory representation, predict the sterility breach. Do not provide a narrative or descriptive text.'
            else:
                raise ValueError(f'Unknown task {self.task}')

            surgery_sg_triplets = self.take_to_all_scene_graphs[take_name]
            current_sg = self.take_to_all_full_scene_graphs[take_name][timepoint]
            surgery_sg_triplets_up_to_current = [elem for elem in surgery_sg_triplets if elem[0] < timepoint]
            all_scene_graphs_till_now = surgery_sg_to_memory_str(surgery_sg_triplets_up_to_current, current_timepoint=timepoint)
            current_sg = scene_graph_to_string(current_sg)

            # add both the memory_str (first), and the current_sg (second). Prepend the human prompt.
            scene_graph_context_information = f'<memory_start>: {all_scene_graphs_till_now}<memory_end>. {current_sg}.'
            human_prompt = f'{scene_graph_context_information} {human_prompt}'

            conv.append_message(conv.roles[0], human_prompt)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            all_prompts.append(prompt)

        if batchsize == 1:
            input_ids = tokenizer_image_token(all_prompts[0], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        else:
            input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in all_prompts]
            # merge with left padding
            inverted_input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(inverted_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            # invert back
            input_ids = torch.flip(input_ids, dims=[1]).to(self.model.device)
            # image_tensor = torch.cat(all_images)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            # print(f'Length of input_ids: {input_ids.shape[1]}')
            output_ids = self.model.generate(
                input_ids,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria], )
        if batchsize == 1:
            outputs = [self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

        return outputs

    def reset_metrics(self, split=None):
        if split == 'train':
            self.train_take_preds = defaultdict(list)
            self.train_take_gts = defaultdict(list)
        elif split == 'val':
            self.val_take_preds = defaultdict(list)
            self.val_take_gts = defaultdict(list)
        else:
            self.train_take_preds = defaultdict(list)
            self.train_take_gts = defaultdict(list)
            self.val_take_preds = defaultdict(list)
            self.val_take_gts = defaultdict(list)

    def infer(self, dataloader):
        return self.validate(dataloader, return_raw_predictions=True)

    # def test_step(self, batch, batch_idx): # not for inference
    #     return self.validation_step(batch, batch_idx)

    def validate(self, dataloader, limit_val_batches=None, logging_information=None, return_raw_predictions=False):
        take_preds = defaultdict(list)
        take_gts = defaultdict(list)
        sample_id_to_raw_predictions = {}  # dictionary to store predicted scene graphs
        limit_counter = None
        if isinstance(limit_val_batches, int):
            limit_counter = limit_val_batches
        elif isinstance(limit_val_batches, float):
            limit_counter = int(limit_val_batches * len(dataloader))

        jsons_cache = {}
        patient_currently_sterile = False
        surgery_done = False
        current_take = ''

        for batch in tqdm(dataloader):
            if limit_counter is not None:
                if limit_counter <= 0:
                    break
                limit_counter -= 1

            outputs = self.forward(batch)
            for idx, output in enumerate(outputs):
                elem = batch[idx]
                sample, multimodal_data = elem['sample'], elem['multimodal_data']
                timepoint = int(sample["frame_id"])
                take_name = sample["take_name"]
                take_name = take_name.rsplit('_', 1)[0]
                if current_take != take_name:
                    current_take = take_name
                    patient_currently_sterile = False
                    surgery_done = False

                if self.task == 'robot_phase' and '004_PKA' in take_name:  # skip, it is simply not suitable for this task, robot already full prepared
                    continue
                current_sg = self.take_to_all_full_scene_graphs[take_name][timepoint]
                if self.task == 'next_action':
                    take_timestamp_to_next_action_folder_path = MMOR_DATA_ROOT_PATH / 'take_timestamp_to_next_action'
                    if take_name not in jsons_cache:
                        jsons_cache[take_name] = {}
                        with open(take_timestamp_to_next_action_folder_path / f'{take_name}.json', 'r') as f:
                            jsons_cache[take_name]["next_action"] = json.load(f)
                    next_action_json = jsons_cache[take_name]["next_action"]
                    downstream_gt = next_action_json[sample['frame_id']]
                    if downstream_gt is None:
                        continue
                    downstream_gt, remaining_time = downstream_gt[0], downstream_gt[1]  # we are ignoring time for now
                    # parse the prediction
                    if ":" in output:
                        output = output.split(":")[0]
                    downstream_pred = output.lower().strip()
                    if downstream_pred == 'scan patient':
                        downstream_pred = 'scan'
                elif self.task == 'robot_phase':
                    take_timestamp_to_robot_phase_path = MMOR_DATA_ROOT_PATH / 'take_timestamp_to_robot_phase'
                    if take_name not in jsons_cache:
                        jsons_cache[take_name] = {}
                        with open(take_timestamp_to_robot_phase_path / f'{take_name}.json', 'r') as f:
                            jsons_cache[take_name]["robot_phase"] = json.load(f)
                    robot_phase_json = jsons_cache[take_name]["robot_phase"]
                    downstream_gt = robot_phase_json[sample['frame_id']].lower().strip()
                    downstream_pred = output.lower().strip()
                elif self.task == 'sterility_breach':
                    take_timestamp_to_sterility_breach_path = MMOR_DATA_ROOT_PATH / 'take_timestamp_to_sterility_breach'
                    if take_name not in jsons_cache:
                        jsons_cache[take_name] = {}
                        with open(take_timestamp_to_sterility_breach_path / f'{take_name}.json', 'r') as f:
                            jsons_cache[take_name]["sterility_breach"] = json.load(f)
                    sterility_breach_json = jsons_cache[take_name]["sterility_breach"]
                    downstream_gt = sterility_breach_json[sample['frame_id']]
                    downstream_gt = 'no' if len(downstream_gt) == 0 else 'yes'
                    network_based_pred = output.lower().strip()
                    rule_based_pred = 'no'
                    for sub, obj, rel in current_sg:
                        if sub in ['head_surgeon', 'assistant_surgeon'] and obj == 'patient' and rel in ['drilling', 'cutting', 'sawing', 'hammering']:
                            patient_currently_sterile = True
                        elif rel == 'suturing':
                            patient_currently_sterile = False
                            surgery_done = True
                        if surgery_done:  # if the surgery is pretty much done, but they are cleaning up, no more sterility breaches are important
                            continue
                        if rel not in STERILITY_BREACH_PREDICATES:
                            continue
                        if sub != 'patient' and sub not in STERILE_ENTITIES and sub not in UNSTERILE_ENTITIES:  # we don't care about this then
                            continue
                        if obj != 'patient' and obj not in STERILE_ENTITIES and obj not in UNSTERILE_ENTITIES:  # we don't care about this then
                            continue
                        if (
                                sub == 'patient' or obj == 'patient') and not patient_currently_sterile:  # we don't care about this then. Only once the patient is sterile, we start caring about sterility breaches
                            continue
                        sub_sterile = sub in STERILE_ENTITIES or (sub == 'patient' and patient_currently_sterile)
                        obj_sterile = obj in STERILE_ENTITIES or (obj == 'patient' and patient_currently_sterile)
                        if sub_sterile != obj_sterile:
                            rule_based_pred = 'yes'
                    if surgery_done:
                        rule_based_pred = 'no'
                    downstream_pred = rule_based_pred
                    if downstream_pred != 'no':
                        downstream_pred = 'yes'
                else:
                    raise ValueError(f'Unknown task {self.task}')
                try:
                    downstream_gt_idx = self.downstream_classes_to_idx[downstream_gt]
                except KeyError:
                    print(f'Unknown ground truth class: {downstream_gt}')
                    continue
                downstream_pred_idx = self.downstream_classes_to_idx.get(downstream_pred, -1)
                take_preds[take_name].append(downstream_pred_idx)
                take_gts[take_name].append(downstream_gt_idx)

        self.val_take_preds, self.val_take_gts = take_preds, take_gts
        self.evaluate_predictions('val')
        self.reset_metrics(split='val')

        if return_raw_predictions:
            return sample_id_to_raw_predictions

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    def evaluate_predictions(self, split):
        if split == 'train':
            take_rel_preds = self.train_take_preds
            take_rel_gts = self.train_take_gts
        elif split == 'val':
            take_rel_preds = self.val_take_preds
            take_rel_gts = self.val_take_gts
        else:
            raise NotImplementedError()

        all_rel_gts = []
        all_rel_preds = []

        for take_name in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_name]
            rel_gts = take_rel_gts[take_name]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.downstream_classes))),
                                               target_names=self.downstream_classes)
            print(f'\nTake {take_name}\n')
            print(cls_report)

        print(f'{split} Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.downstream_classes))),
                                           target_names=self.downstream_classes)
        print(cls_report)
