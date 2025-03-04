# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm

from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN
from LLaVA.llava.conversation import SeparatorStyle, default_conversation
from LLaVA.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.model.builder import load_pretrained_model
from helpers.configurations import TRACKER_OBJECT_MAP
from scene_graph_prediction.llava_helpers.scene_graph_converters import llava_sg_to_surgery_sg, surgery_sg_to_memory_str
from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import map_scene_graph_name_to_vocab_idx, map_vocab_idx_to_scene_graph_name, reversed_role_synonyms


class ModelWrapper:
    def __init__(self, config, relationNames, classNames, model_path, model_base='liuhaotian/llava-v1.5-7b', load_8bit=False, load_4bit=False):
        self.config = config
        self.mconfig = config['MODEL']
        self.n_object_types = 6
        self.relationNames = relationNames
        self.classNames = classNames
        self.relation_names_lower_case = [relation.lower() for relation in self.relationNames]
        self.lr = float(self.config['LR'])
        # evaluation metrics
        self.train_take_rel_preds = defaultdict(list)
        self.train_take_rel_gts = defaultdict(list)
        self.val_take_rel_preds = defaultdict(list)
        self.val_take_rel_gts = defaultdict(list)

        self.train_take_rel_binary_interaction_preds = defaultdict(list)
        self.train_take_rel_binary_interaction_gts = defaultdict(list)
        self.val_take_rel_binary_interaction_preds = defaultdict(list)
        self.val_take_rel_binary_interaction_gts = defaultdict(list)

        self.train_take_entity_preds = defaultdict(list)
        self.train_take_entity_gts = defaultdict(list)
        self.val_take_entity_preds = defaultdict(list)
        self.val_take_entity_gts = defaultdict(list)

        self.reset_metrics()

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name, load_8bit, load_4bit)
        self.model.config.mv_type = self.mconfig['mv_type']
        self.model.config.tokenizer_padding_side = "left"
        self.temporal_online_prediction = False
        if 'temporality' in config and config['temporality'] == 'PRED':
            print('Preparing temporality PRED')
            self.take_to_history = defaultdict(list)
            self.temporal_online_prediction = True

    def forward(self, batch):
        batchsize = len(batch)
        all_images = []
        all_prompts = []
        pcs = []
        audios = []
        segmasks = []
        all_dataset_types = []
        for elem in batch:
            conv = deepcopy(default_conversation)
            images = []
            pc = None
            audio = None
            segmask = None
            sample, multimodal_data = elem['sample'], elem['multimodal_data']
            azure_image_paths = multimodal_data['azure']
            # load images
            if '4DOR' in sample['take_name']:
                all_dataset_types.append('4D-OR')
                for cam_idx in self.config['CAMERAS_4DOR']:
                    try:
                        img = Image.open(azure_image_paths[cam_idx - 1]).convert('RGB')
                    except Exception as e:
                        print(f'Warning: not enough images for 4DOR ({len(azure_image_paths)} < {max(self.config["CAMERAS_4DOR"])}): {e}')
                        img = Image.new('RGB', (2048, 1536), (0, 0, 0))
                    images.append(img)
            else:
                all_dataset_types.append('MM-OR')
                simstation_image_paths = multimodal_data['simstation'] if 'simstation' in multimodal_data else None
                trackercam_image_paths = multimodal_data['trackercam'] if 'trackercam' in multimodal_data else None
                # we have 7 image slots. We will use them like this. We prefer to use azure images(5) + robot monitor screen(1) + trackercam(1). If azure is not available, we will use simstation instead (3)
                # load the room cameras here
                if len(azure_image_paths) > 0:
                    for cam_idx in self.config['CAMERAS_MMOR']:
                        img = Image.open(azure_image_paths[cam_idx - 1]).convert('RGB')
                        images.append(img)
                elif simstation_image_paths is not None and len(simstation_image_paths) > 0:
                    for cam_idx in (2, 0, 3):
                        img = Image.open(simstation_image_paths[cam_idx]).convert('RGB')
                        images.append(img)
                else:  # ideally we should never be here
                    print('Defaulting to black images...')
                    for _ in self.config['CAMERAS_MMOR']:
                        img = Image.new('RGB', (2048, 1536), (0, 0, 0))
                        images.append(img)

                # load the robot screen, which is the 1. image in simstation
                if simstation_image_paths is not None and len(simstation_image_paths) > 0:
                    img = Image.open(simstation_image_paths[1]).convert('RGB')
                    images.append(img)
                # load the trackercam
                if trackercam_image_paths is not None and len(trackercam_image_paths) > 0:
                    img = Image.open(trackercam_image_paths[0]).convert('RGB')
                    images.append(img)

            # Similar operation in model_worker.py
            image_tensor = process_images(images, self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.bfloat16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.bfloat16)
            all_images.append(image_tensor)

            # Load pc
            if 'pc' in multimodal_data:
                pc = o3d.io.read_point_cloud(str(multimodal_data['pc'][0]))
                pc = torch.from_numpy(np.concatenate([np.asarray(pc.points) / 1000, np.asarray(pc.colors)], axis=-1)).float()
            pcs.append(pc)

            # load audio
            if 'audio' in multimodal_data:
                audio = torch.load(multimodal_data['audio'][0], map_location='cpu')
            audios.append(audio)

            if 'segmasks' in multimodal_data:
                # they are one channel images, read them accordingly
                segmask = [torch.from_numpy(np.array(Image.open(segmask_path).convert('L'))) for segmask_path in multimodal_data['segmasks']]
            segmasks.append(segmask)

            inp = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, mps, patient, student, instrument table, operating table, secondary table, anesthesia equipment, c_arm, mako_robot, monitor, mps_station, tracker, drape, drill, hammer, saw, instrument]. Predicates: [assisting, calibrating, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, scanning, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'

            # Load robot metadata
            if 'robot_metadata' in multimodal_data:
                robot_metadata = multimodal_data['robot_metadata'][0]
                with open(robot_metadata, 'r') as f:
                    robot_metadata = json.load(f)
                robot_metadata_str = ''
                for key, value in sorted(robot_metadata.items()):
                    robot_metadata_str += f'{value["type"]}: {value["template_name"]}, '
                robot_metadata_str = robot_metadata_str.rstrip(', ')
                # insert this into the human prompt
                inp = inp.replace('Entities: ', f'<robot_metadata_start>: {robot_metadata_str} <robot_metadata_end>. Entities: ')

            if 'tracker' in multimodal_data:
                tracker_metadata = multimodal_data['tracker'][0]
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
                inp = inp.replace('Entities: ', f'<tracker_metadata_start>: {tracker_metadata_str} <tracker_metadata_end>. Entities: ')

            if 'speech_transcript' in multimodal_data:
                speech_transcript = multimodal_data['speech_transcript'][0]
                with open(speech_transcript, 'r') as f:
                    speech_transcript = json.load(f)
                speech_transcript_str = speech_transcript['text']
                inp = inp.replace('Entities: ', f'<speech_transcript_start>: {speech_transcript_str} <speech_transcript_end>. Entities: ')

            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            if 'temporality' in self.config:
                if self.config['temporality'] == 'PRED':
                    take_name = elem['sample']['take_name']
                    timepoint_idx = int(elem['sample']['frame_id'])
                    raw_triplets = self.take_to_history[take_name]
                    surgery_sg_triplets = llava_sg_to_surgery_sg(raw_triplets, entity_of_interest=None, IRRELEVANT_PREDS=['closeto', 'closeTo'])
                    surgery_sg_triplets = [elem for elem in surgery_sg_triplets if elem[0] < timepoint_idx]
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint_idx, TEMPORAL_STYLE='longshort')
                else:
                    raise NotImplementedError()
                if len(memory_str) > 5000:
                    print(f'Warning: memory string is too long ({len(memory_str)} chars)')
                    memory_str = '...' + memory_str[-5000:]
                inp = inp.replace(f'{DEFAULT_IMAGE_TOKEN}\n', f'{DEFAULT_IMAGE_TOKEN}\n<memory_start>: {memory_str}<memory_end>.\n')
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            all_prompts.append(prompt)

        if batchsize == 1:
            input_ids = tokenizer_image_token(all_prompts[0], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            image_tensor = all_images[0]

        else:
            input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in all_prompts]
            # merge with left padding
            inverted_input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(inverted_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            # invert back
            input_ids = torch.flip(input_ids, dims=[1]).to(self.model.device)
            # image_tensor = torch.cat(all_images)
            image_tensor = [img[0] for img in all_images]  # still keep it as a list instead of trying to convert to a tensor

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            # print(f'Length of input_ids: {input_ids.shape[1]}')
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria],
                pc=pcs if any(pc is not None for pc in pcs) else None,
                audio=audios if any(audio is not None for audio in audios) else None,
                segmasks=segmasks if any(segmask is not None for segmask in segmasks) else None
            )
        if batchsize == 1:
            outputs = [self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

        return outputs

    def reset_metrics(self, split=None):
        if split == 'train':
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)

            self.train_take_rel_binary_interaction_preds = defaultdict(list)
            self.train_take_rel_binary_interaction_gts = defaultdict(list)

            self.train_take_entity_preds = defaultdict(list)
            self.train_take_entity_gts = defaultdict(list)
        elif split == 'val':
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)

            self.val_take_rel_binary_interaction_preds = defaultdict(list)
            self.val_take_rel_binary_interaction_gts = defaultdict(list)

            self.val_take_entity_preds = defaultdict(list)
            self.val_take_entity_gts = defaultdict(list)
        else:
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)

            self.train_take_rel_binary_interaction_preds = defaultdict(list)
            self.train_take_rel_binary_interaction_gts = defaultdict(list)
            self.val_take_rel_binary_interaction_preds = defaultdict(list)
            self.val_take_rel_binary_interaction_gts = defaultdict(list)

            self.train_take_entity_preds = defaultdict(list)
            self.train_take_entity_gts = defaultdict(list)
            self.val_take_entity_preds = defaultdict(list)
            self.val_take_entity_gts = defaultdict(list)

    def infer(self, dataloader):
        return self.validate(dataloader, return_raw_predictions=True)

    # def test_step(self, batch, batch_idx): # not for inference
    #     return self.validation_step(batch, batch_idx)

    def validate(self, dataloader, limit_val_batches=None, logging_information=None, return_raw_predictions=False):
        take_rel_preds = defaultdict(list)
        take_rel_gts = defaultdict(list)
        take_rel_binary_interaction_preds = defaultdict(list)
        take_rel_binary_interaction_gts = defaultdict(list)
        take_entity_preds = defaultdict(list)
        take_entity_gts = defaultdict(list)

        sample_id_to_raw_predictions = {}  # dictionary to store predicted scene graphs
        limit_counter = None
        if isinstance(limit_val_batches, int):
            limit_counter = limit_val_batches
        elif isinstance(limit_val_batches, float):
            limit_counter = int(limit_val_batches * len(dataloader))

        for batch in tqdm(dataloader):
            if limit_counter is not None:
                if limit_counter <= 0:
                    break
                limit_counter -= 1

            assert len(batch) == 1 or not self.temporal_online_prediction

            outputs = self.forward(batch)
            for idx, output in enumerate(outputs):
                elem = batch[idx]
                sample, multimodal_data = elem['sample'], elem['multimodal_data']
                timepoint = int(sample['frame_id'])
                triplets = []
                raw_triplets = []
                # remove everything between the first """ and the last """ using regex. This is used for chain of thought
                output = re.sub(r'""".*?"""', '', output, flags=re.DOTALL)
                if '<SG>' in output and '</SG>' in output and output.index('<SG>') < output.index('</SG>'):
                    triplet_str = output.split('<SG>')[1].split('</SG>')[0].strip().split(';')
                else:
                    triplet_str = output.split(';')

                for triplet in triplet_str:
                    triplet = triplet.replace('.', '').replace('</s>', '').replace('<s>', '').strip()
                    if triplet == '':
                        continue
                    triplet = triplet.split(',')
                    triplet = [elem.strip() for elem in triplet]
                    if len(triplet) != 3:
                        continue
                    sub, obj, pred = triplet
                    raw_triplets.append((sub, pred, obj))
                    if sub in reversed_role_synonyms:
                        sub = reversed_role_synonyms[sub]
                    if obj in reversed_role_synonyms:
                        obj = reversed_role_synonyms[obj]
                    triplets.append((sub, pred, obj))
                # these have to be mapped. First to human names, also the predicates
                sample_id_to_raw_predictions[sample['sample_id']] = raw_triplets
                if self.temporal_online_prediction:
                    self.take_to_history[sample['take_name']].append({'timepoint_idx': timepoint, 'scene_graph': raw_triplets})
                rel_preds = []
                for (sub, pred, obj) in triplets:
                    try:
                        sub = map_scene_graph_name_to_vocab_idx(sub.replace(' ', '_'))
                        obj = map_scene_graph_name_to_vocab_idx(obj.replace(' ', '_'))
                        pred = map_scene_graph_name_to_vocab_idx(pred)
                        rel_preds.append((sub, pred, obj))
                    except Exception as e:
                        print(e)
                        continue
                rel_labels = torch.tensor([(map_scene_graph_name_to_vocab_idx(sub), map_scene_graph_name_to_vocab_idx(rel), map_scene_graph_name_to_vocab_idx(obj)) for
                                           (sub, obj, rel) in sample['relationships']])
                human_readable_pred = [(map_vocab_idx_to_scene_graph_name(sub), map_vocab_idx_to_scene_graph_name(pred), map_vocab_idx_to_scene_graph_name(obj))
                                       for sub, pred, obj in rel_preds]
                human_readable_gt = [(map_vocab_idx_to_scene_graph_name(sub), map_vocab_idx_to_scene_graph_name(pred), map_vocab_idx_to_scene_graph_name(obj))
                                     for sub, pred, obj in rel_labels.tolist()]
                if len(rel_labels) == 0:
                    all_gt_objects = []
                else:
                    all_gt_objects = sorted(set(rel_labels[:, [0, 2]].flatten().tolist()))
                # Search for all possible relationships between objects, those that don't have any should be labeled 'none', otherwise the correct relation is asked for
                all_pred_objects = sorted(set([sub for sub, _, _ in rel_preds] + [obj for _, _, obj in rel_preds]))

                for gt_obj1 in all_gt_objects:
                    # add this object to ground truth entities
                    take_entity_gts[sample['take_name']].append(self.classNames.index(map_vocab_idx_to_scene_graph_name(gt_obj1)))
                    # if this object is part of the predicted entities, add it to the predicted entities
                    if gt_obj1 in all_pred_objects:
                        take_entity_preds[sample['take_name']].append(self.classNames.index(map_vocab_idx_to_scene_graph_name(gt_obj1)))
                    else:
                        take_entity_preds[sample['take_name']].append(-1)
                    for gt_obj2 in all_gt_objects:
                        if gt_obj1 == gt_obj2:
                            continue
                        for gt_sub, gt_rel, gt_obj in rel_labels:
                            if gt_sub == gt_obj1 and gt_obj == gt_obj2:
                                take_rel_gts[sample['take_name']].append(self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(gt_rel.item())))
                                take_rel_binary_interaction_gts[sample['take_name']].append(1)
                                break
                        else:
                            take_rel_gts[sample['take_name']].append(self.relation_names_lower_case.index('none'))
                            take_rel_binary_interaction_gts[sample['take_name']].append(0)
                        for pred_sub, pred_rel, pred_obj in rel_preds:
                            if pred_sub == gt_obj1 and pred_obj == gt_obj2:
                                try:
                                    pred_rel_id = self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(pred_rel))
                                    take_rel_binary_interaction_preds[sample['take_name']].append(1)
                                except Exception as e:  # if a   none sense relation was predicted ignore
                                    pred_rel_id = self.relation_names_lower_case.index('none')
                                    take_rel_binary_interaction_preds[sample['take_name']].append(0)
                                take_rel_preds[sample['take_name']].append(pred_rel_id)
                                break
                        else:
                            take_rel_preds[sample['take_name']].append(self.relation_names_lower_case.index('none'))
                            take_rel_binary_interaction_preds[sample['take_name']].append(0)

        self.val_take_rel_preds, self.val_take_rel_gts = take_rel_preds, take_rel_gts
        self.val_take_rel_binary_interaction_preds, self.val_take_rel_binary_interaction_gts = take_rel_binary_interaction_preds, take_rel_binary_interaction_gts
        self.val_take_entity_preds, self.val_take_entity_gts = take_entity_preds, take_entity_gts
        self.evaluate_predictions(None, 'val', logging_information=logging_information)
        self.reset_metrics(split='val')

        if return_raw_predictions:
            return sample_id_to_raw_predictions

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    def evaluate_predictions(self, epoch_loss, split, logging_information=None):
        if split == 'train':
            take_rel_preds = self.train_take_rel_preds
            take_rel_gts = self.train_take_rel_gts
            take_rel_binary_interaction_preds = self.train_take_rel_binary_interaction_preds
            take_rel_binary_interaction_gts = self.train_take_rel_binary_interaction_gts
            take_entity_preds = self.train_take_entity_preds
            take_entity_gts = self.train_take_entity_gts
        elif split == 'val':
            take_rel_preds = self.val_take_rel_preds
            take_rel_gts = self.val_take_rel_gts
            take_rel_binary_interaction_preds = self.val_take_rel_binary_interaction_preds
            take_rel_binary_interaction_gts = self.val_take_rel_binary_interaction_gts
            take_entity_preds = self.val_take_entity_preds
            take_entity_gts = self.val_take_entity_gts
        else:
            raise NotImplementedError()

        all_rel_gts = []
        all_rel_preds = []
        all_rel_binary_interaction_gts = []
        all_rel_binary_interaction_preds = []
        all_entity_gts = []
        all_entity_preds = []
        data_type_rel_preds = defaultdict(list)
        data_type_rel_gts = defaultdict(list)
        data_type_rel_binary_interaction_preds = defaultdict(list)
        data_type_rel_binary_interaction_gts = defaultdict(list)
        data_type_entity_preds = defaultdict(list)
        data_type_entity_gts = defaultdict(list)

        for take_name in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_name]
            rel_gts = take_rel_gts[take_name]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)

            rel_binary_interaction_preds = take_rel_binary_interaction_preds[take_name]
            rel_binary_interaction_gts = take_rel_binary_interaction_gts[take_name]
            all_rel_binary_interaction_gts.extend(rel_binary_interaction_gts)
            all_rel_binary_interaction_preds.extend(rel_binary_interaction_preds)

            entity_preds = take_entity_preds[take_name]
            entity_gts = take_entity_gts[take_name]
            all_entity_gts.extend(entity_gts)
            all_entity_preds.extend(entity_preds)

            # Determine data type based on take_idx or take name
            if '4DOR' in take_name:
                data_type = '4DOR'
            else:
                data_type = 'MMOR'
            # Accumulate predictions and ground truths per data type
            data_type_rel_preds[data_type].extend(rel_preds)
            data_type_rel_gts[data_type].extend(rel_gts)
            data_type_rel_binary_interaction_preds[data_type].extend(rel_binary_interaction_preds)
            data_type_rel_binary_interaction_gts[data_type].extend(rel_binary_interaction_gts)
            data_type_entity_preds[data_type].extend(entity_preds)
            data_type_entity_gts[data_type].extend(entity_gts)
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames, output_dict=True, digits=4)  # non existing relations will be counted as True
            real_macro_values = {'precision': [], 'recall': [], 'f1-score': []}
            for rel_name in self.relationNames:
                if cls_report[rel_name]['support'] == 0:
                    continue
                for score_type in ['precision', 'recall', 'f1-score']:
                    real_macro_values[score_type].append(cls_report[rel_name][score_type])
                    # self.log(f'{rel_name}/{take_idx}_{score_type[:2].upper()}', cls_report[rel_name][score_type], rank_zero_only=True)
                    if logging_information is not None:
                        logging_information['logger'].log_metrics({f'{rel_name}/{take_name}_{score_type[:2].upper()}': cls_report[rel_name][score_type]}, step=logging_information['checkpoint_id'])
            real_macro_values = {score_type: np.mean(values) for score_type, values in real_macro_values.items()}
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames, digits=4)  # non existing relations will be counted as True
            print(f'\nTake {take_name}\n')
            print(cls_report)
            print(f'Macro Precision: {real_macro_values["precision"]:.3f}, Macro Recall: {real_macro_values["recall"]:.3f}, Macro F1: {real_macro_values["f1-score"]:.3f}')

            # a less granular report for binary_interaction and entity detection
            binary_interaction_cls_report = classification_report(rel_binary_interaction_gts, rel_binary_interaction_preds, labels=[0, 1],
                                                                  target_names=['no_interaction', 'interaction'], output_dict=False, digits=4)
            entity_cls_report = classification_report(entity_gts, entity_preds, labels=list(range(len(self.classNames))),
                                                      target_names=self.classNames, output_dict=False, digits=4)
            print(f'\nBinary Interaction Classification Report for Take {take_name}\n')
            print(binary_interaction_cls_report)
            print(f'\nEntity Classification Report for Take {take_name}\n')
            print(entity_cls_report)

        # Compute and print classification reports per data type
        for data_type in data_type_rel_preds.keys():
            rel_preds = data_type_rel_preds[data_type]
            rel_gts = data_type_rel_gts[data_type]
            rel_binary_interaction_preds = data_type_rel_binary_interaction_preds[data_type]
            rel_binary_interaction_gts = data_type_rel_binary_interaction_gts[data_type]
            entity_preds = data_type_entity_preds[data_type]
            entity_gts = data_type_entity_gts[data_type]

            cls_report = classification_report(
                rel_gts, rel_preds,
                labels=list(range(len(self.relationNames))),
                target_names=self.relationNames,
                output_dict=True,
                digits=4  # non existing relations will be counted as True
            )
            real_macro_values = {'precision': [], 'recall': [], 'f1-score': []}
            for rel_name in self.relationNames:
                if cls_report[rel_name]['support'] == 0:
                    continue
                for score_type in ['precision', 'recall', 'f1-score']:
                    real_macro_values[score_type].append(cls_report[rel_name][score_type])
            real_macro_values = {score_type: np.mean(values) for score_type, values in real_macro_values.items()}

            # Log per-data-type metrics if needed
            if logging_information is not None:
                logging_information['logger'].log_metrics({f'{logging_information["split"]}_{data_type}_precision': cls_report['macro avg']['precision']}, step=logging_information['checkpoint_id'])
                logging_information['logger'].log_metrics({f'{logging_information["split"]}_{data_type}_recall': cls_report['macro avg']['recall']}, step=logging_information['checkpoint_id'])
                logging_information['logger'].log_metrics({f'{logging_information["split"]}_{data_type}_macro_f1': cls_report['macro avg']['f1-score']}, step=logging_information['checkpoint_id'])

            # Print per-data-type classification report
            print(f'\nData Type: {data_type}\n')
            cls_report_str = classification_report(
                rel_gts, rel_preds,
                labels=list(range(len(self.relationNames))),
                target_names=self.relationNames,
                digits=4  # non existing relations will be counted as True
            )
            print(cls_report_str)
            print(f'Macro Precision: {real_macro_values["precision"]:.3f}, Macro Recall: {real_macro_values["recall"]:.3f}, Macro F1: {real_macro_values["f1-score"]:.3f}')

            # Print per-data-type binary interaction and entity classification reports
            binary_interaction_cls_report = classification_report(
                rel_binary_interaction_gts, rel_binary_interaction_preds,
                labels=[0, 1],
                target_names=['no_interaction', 'interaction'],
                output_dict=False,
                digits=4
            )
            entity_cls_report = classification_report(
                entity_gts, entity_preds,
                labels=list(range(len(self.classNames))),
                target_names=self.classNames,
                output_dict=False,
                digits=4
            )
            print(f'\nBinary Interaction Classification Report for Data Type {data_type}\n')
            print(binary_interaction_cls_report)
            print(f'\nEntity Classification Report for Data Type {data_type}\n')
            print(entity_cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True, digits=4)
        old_macro_f1 = results['macro avg']['f1-score']
        real_macro_values = {'precision': [], 'recall': [], 'f1-score': []}
        for rel_name in self.relationNames:
            if results[rel_name]['support'] == 0:
                continue
            for score_type in ['precision', 'recall', 'f1-score']:
                real_macro_values[score_type].append(results[rel_name][score_type])
        real_macro_values = {score_type: np.mean(values) for score_type, values in real_macro_values.items()}
        macro_f1 = real_macro_values['f1-score']
        if logging_information is not None:
            # logging_information will have a key use it to log to wandb. It will also have a checkpoint int, which we also want to log (similar to epoch). Also we want to use the split to log as train or val
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_precision': results['macro avg']['precision']}, step=logging_information['checkpoint_id'])
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_recall': results['macro avg']['recall']}, step=logging_information['checkpoint_id'])
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_macro_f1': results['macro avg']['f1-score']}, step=logging_information['checkpoint_id'])

        print(f'{split} Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                           target_names=self.relationNames, digits=4)  # non existing relations will be counted as True
        print(cls_report)
        print(f'Macro Precision: {real_macro_values["precision"]:.3f}, Macro Recall: {real_macro_values["recall"]:.3f}, Macro F1: {real_macro_values["f1-score"]:.3f}')

        # a less granular report for binary_interaction and entity detection
        binary_interaction_cls_report = classification_report(all_rel_binary_interaction_gts, all_rel_binary_interaction_preds, labels=[0, 1],
                                                              target_names=['no_interaction', 'interaction'], output_dict=False, digits=4)
        entity_cls_report = classification_report(all_entity_gts, all_entity_preds, labels=list(range(len(self.classNames))), target_names=self.classNames, output_dict=False, digits=4)
        print(f'\nBinary Interaction Classification Report for {split}\n')
        print(binary_interaction_cls_report)

        print(f'\nEntity Classification Report for {split}\n')
        print(entity_cls_report)

        return macro_f1
