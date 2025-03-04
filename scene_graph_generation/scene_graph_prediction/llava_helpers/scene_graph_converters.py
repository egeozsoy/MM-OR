import random
import re
from collections import Counter
from random import shuffle

PRED_COUNTER = Counter()


def collapse_sgs(sgs):
    '''
    We only take note of the changes. Given a timepoint, we want to collapse all the changes and know what the current status is. As an example if the we know from 100 timepoints ago that head surgeon is cutting patient and this did not change
    then we assume that this is still the case. Handle the case of stopping
    '''
    sub_obj_to_pred = {}  # key: (sub, obj), value: pred
    for timepoint_idx, (sub, pred, obj) in sgs:
        if pred.startswith('not '):
            if (sub, obj) in sub_obj_to_pred:
                del sub_obj_to_pred[(sub, obj)]
        else:
            sub_obj_to_pred[(sub, obj)] = pred

    return sub_obj_to_pred


def find_related_entities(scene_graph, entity_of_interest, multi_hop_n):
    def _find_related(current_entity, current_hop, visited):
        if current_hop > multi_hop_n:
            return set()

        visited.add(current_entity)
        related_entities = set()

        if current_hop == 0:
            related_entities.add(current_entity)  # Add the entity of interest at the start

        for sub, pred, obj in scene_graph:
            if sub == current_entity and obj not in visited:
                if current_hop < multi_hop_n:
                    related_entities.add(obj)
                    related_entities.update(_find_related(obj, current_hop + 1, visited.copy()))
            elif obj == current_entity and sub not in visited:
                if current_hop < multi_hop_n:
                    related_entities.add(sub)
                    related_entities.update(_find_related(sub, current_hop + 1, visited.copy()))

        return related_entities

    # Start with the entity of interest and an empty set for visited entities
    return _find_related(entity_of_interest, 0, set())


def llava_sg_to_surgery_sg(llava_sgs, entity_of_interest=None, IRRELEVANT_PREDS=None):
    '''
    Modifies the original function to only include changes that concern the specified entity.
    entity_of_interest: The entity to focus on (e.g., 'head surgeon').
    '''
    surgery_sg_triplets = []  # Records the timepoints as well as the corresponding change log.

    for elem in llava_sgs:
        sg = elem['scene_graph']
        timepoint = elem['timepoint_idx']
        prev_sg = collapse_sgs(surgery_sg_triplets)
        if entity_of_interest is None and IRRELEVANT_PREDS is None:
            current_sg = {(sub, obj): pred for (sub, pred, obj) in sg if sub != 'none' and obj != 'none'}
        elif entity_of_interest is None:
            current_sg = {(sub, obj): pred for (sub, pred, obj) in sg if pred not in IRRELEVANT_PREDS and sub != 'none' and obj != 'none'}
        else:
            related_entities = find_related_entities(sg, entity_of_interest, multi_hop_n=0)  # 0 only entity of interest, 1 also related entities, 2 also related entities of related entities, etc.
            # Filter scene graph for changes involving the entity of interest
            current_sg = {(sub, obj): pred for (sub, pred, obj) in sg if
                          pred not in IRRELEVANT_PREDS and (sub == entity_of_interest or obj == entity_of_interest or sub in related_entities or obj in related_entities)}
        # Compare current_sg with previous (collapsed) sg. If there is a difference, add it to the modifications.
        additions = []
        removals = []
        for (sub, obj), pred in current_sg.items():
            if (sub, obj) not in prev_sg:
                additions.append((sub, pred, obj))
        for (sub, obj), pred in prev_sg.items():
            if (sub, obj) not in current_sg:
                removals.append((sub, pred, obj))
        modifications = []
        for sub, pred, obj in additions:
            PRED_COUNTER[pred] += 1
            modifications.append((timepoint, (sub, pred, obj)))
        for sub, pred, obj in removals:
            modifications.append((timepoint, (sub, f'not {pred}', obj)))
        shuffle(modifications)
        surgery_sg_triplets.extend(modifications)
    return surgery_sg_triplets


def extract_take_int_from_image_path(image_path):
    return int(re.findall('take(\d+)', str(image_path))[0])


def parse_llava_sg(llava_sg):
    if '<SG>' in llava_sg and '</SG>' in llava_sg and llava_sg.index('<SG>') < llava_sg.index('</SG>'):
        triplet_str = llava_sg.split('<SG>')[1].split('</SG>')[0].strip().split(';')
    else:
        triplet_str = llava_sg.split(';')
    triplets = []
    for triplet in triplet_str:
        triplet = triplet.replace('.', '').replace('</s>', '').replace('<s>', '').strip()
        if triplet == '':
            continue
        triplet = triplet.split(',')
        triplet = [elem.strip() for elem in triplet]
        if len(triplet) != 3:
            continue
        sub, obj, pred = triplet
        triplets.append((sub, pred, obj))
    return triplets


def surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint, TEMPORAL_STYLE='longshort', DROP_HISTORY=False):
    '''
    Temporal style can be all, long, short, longshort
    '''

    def _get_memory_str(sub, obj, pred, timepoint, last_reltimepoint):
        rel_timepoint = current_timepoint - timepoint
        if rel_timepoint == last_reltimepoint:  # add without timepoint
            m_str = f'{sub},{obj},{pred}; '
        else:
            m_str = f'{sub},{obj},{pred}; '
            last_reltimepoint = rel_timepoint
        return m_str, last_reltimepoint

    memory_str = ''
    last_reltimepoint = -1
    if TEMPORAL_STYLE == 'short':
        # Only include the most recent 5 changes, formatted as short term memory.
        memory_str += 'Short: '
        for timepoint, (sub, pred, obj) in surgery_sg_triplets[-5:]:
            m_str, last_reltimepoint = _get_memory_str(sub, obj, pred, timepoint, last_reltimepoint)
            if DROP_HISTORY is not False and random.random() < DROP_HISTORY:
                continue
            memory_str += m_str
    elif TEMPORAL_STYLE == 'long':
        # Only include long term memory, formatted in the long term manner.
        memory_str += 'Long: '
        occurrenced_triplets = set()
        for timepoint, (sub, pred, obj) in surgery_sg_triplets[:-5]:
            # simplified representation: Only the first occurance of every action is logged. "not" actions are also skipped.
            if (sub, obj, pred) not in occurrenced_triplets and not pred.startswith('not '):
                occurrenced_triplets.add((sub, obj, pred))
                m_str, last_reltimepoint = _get_memory_str(sub, obj, pred, timepoint, last_reltimepoint)
                if DROP_HISTORY is not False and random.random() < DROP_HISTORY:
                    continue
                memory_str += m_str

    elif TEMPORAL_STYLE == 'longshort':
        # include both short and long term memory, formatted in the a mix of the two styles.
        memory_str += 'Long: '
        occurrenced_triplets = set()
        for timepoint, (sub, pred, obj) in surgery_sg_triplets[:-5]:
            # simplified representation: Only the first occurance of every action is logged. "not" actions are also skipped.
            if (sub, obj, pred) not in occurrenced_triplets and not pred.startswith('not '):
                occurrenced_triplets.add((sub, obj, pred))
                m_str, last_reltimepoint = _get_memory_str(sub, obj, pred, timepoint, last_reltimepoint)
                if DROP_HISTORY is not False and random.random() < DROP_HISTORY:
                    continue
                memory_str += m_str
        memory_str += 'Short: '
        for timepoint, (sub, pred, obj) in surgery_sg_triplets[-5:]:
            # full representation: All actions are logged. "not" actions are also logged.
            m_str, last_reltimepoint = _get_memory_str(sub, obj, pred, timepoint, last_reltimepoint)
            if DROP_HISTORY is not False and random.random() < DROP_HISTORY:
                continue
            memory_str += m_str

    if memory_str == '':
        return ''
    return memory_str[:-2]
