import json
from pathlib import Path

from tqdm import tqdm

from helpers.configurations import MMOR_DATA_ROOT_PATH, OR_4D_DATA_ROOT_PATH, MMOR_TAKE_NAMES, MMOR_SPLIT_TO_TAKES, MMOR_TAKE_NAME_TO_FOLDER, OR4D_TAKE_SPLIT


def filter_rels_by(scan_relationships, sub=None, obj=None, pred=None):
    filtered_rels = []
    for (s, o, p) in scan_relationships:
        if sub is not None and s != sub:
            continue
        if obj is not None and o != obj:
            continue
        if pred is not None and p != pred:
            continue
        filtered_rels.append((s, o, p))

    return filtered_rels


def infer_lyingon(scan_objects, scan_relationships):
    # if patient in scan_objects, infer patient lying on operating_table
    if 'patient' in scan_objects or 'operating_table' in scan_objects:
        scan_objects.add('patient')
        scan_objects.add('operating_table')
        scan_relationships.add((('patient', 'operating_table', 'lyingOn')))


def infer_holding_instrument(scan_objects, scan_relationships):
    new_rels = set()
    for (sub, obj, pred) in scan_relationships:
        if pred in ['cutting', 'drilling', 'sawing', 'suturing', 'hammering', 'cementing', 'cleaning']:
            scan_objects.add('instrument')
            new_rels.add((sub, 'instrument', 'holding'))

    scan_relationships.update(new_rels)


def infer_operating_table_rels(scan_objects, scan_relationships):
    # if patient has any relations with an object, assume that object is close to the operating table
    new_rels = set()
    for s in scan_objects:
        if s == 'operating_table':
            continue

        rels1 = filter_rels_by(scan_relationships, sub=s, obj='patient')
        rels2 = filter_rels_by(scan_relationships, sub='patient', obj=s)
        if len(rels1) + len(rels2) > 0:
            existing_operating_table_rel = filter_rels_by(scan_relationships, sub=s, obj='operating_table')
            if len(existing_operating_table_rel) == 0:
                new_rels.add((s, 'operating_table', 'closeTo'))

    scan_relationships.update(new_rels)


def infer_closeto(scan_objects, scan_relationships):
    raise Exception('We have decided that this does not make much sense and is only spamming relationships')
    # if any two objects have any relationship with each other, assume closeto in the other direction
    new_rels = set()
    for s1 in scan_objects:
        for s2 in scan_objects:
            rels1 = filter_rels_by(scan_relationships, sub=s1, obj=s2)
            rels2 = filter_rels_by(scan_relationships, sub=s2, obj=s1)
            if len(rels1) == 0 and len(rels2) == 0:
                # Both directions no relations, skip
                continue
            elif len(rels1) > 0 and len(rels2) > 0:
                # Both directions with relation, skip
                continue
            elif len(rels1) > 0 and len(rels2) == 0:
                # Relation only from s1 to s2, infer reverse closeTo relation
                new_rels.add((s2, s1, 'closeTo'))
            elif len(rels1) == 0 and len(rels2) > 0:
                # Relation only from s2 to s1, infer reverse closeTo relation
                new_rels.add((s1, s2, 'closeTo'))

    scan_relationships.update(new_rels)


def check_unique_relation(scan_objects, scan_relationships, relation_json_path, take):
    for s1 in scan_objects:
        for s2 in scan_objects:
            counter = 0
            for (sub, obj, pred) in scan_relationships:
                if (sub == s1 and obj == s2):
                    counter += 1

            assert counter <= 1
            # if counter > 1:
            #     print(f'Error: {take}-{relation_json_path.name} has multiple relations between {s1} and {s2}')


def main():
    INCLUDE_MMOR = True
    INCLUDE_4D_OR = True
    print(f'INCLUDE_MMOR: {INCLUDE_MMOR}, INCLUDE_4D_OR: {INCLUDE_4D_OR}')
    save_path = Path('data')

    samples = []
    all_classes = set()
    all_relations = set()
    save_classes_path = save_path / 'classes.txt'
    save_relationships_path = save_path / 'relationships.txt'
    save_train_json_path = save_path / 'relationships_train.json'
    save_val_json_path = save_path / 'relationships_validation.json'
    save_test_json_path = save_path / 'relationships_test.json'

    # start with 4D-OR
    if INCLUDE_4D_OR:
        for take in tqdm(range(12), desc='Loading 4D-OR'):
            root_path = OR_4D_DATA_ROOT_PATH / f'export_holistic_take{take}_processed'
            relations_path = root_path / 'relation_labels'
            for relation_json_path in sorted(list(relations_path.glob('*.json'))):
                frame_id = relation_json_path.name.replace('.json', '')
                scan_objects = set()
                scan_relationships = set()
                with relation_json_path.open() as f:
                    info_json = json.load(f)
                    relation_json = info_json['rel_annotations']
                    human_name_json = info_json['human_name_annotations']
                    # delete humans with none keys
                    human_name_json = {k: v for k, v in human_name_json.items() if v != 'none'}
                for sub, pred, obj, sub_body_part, obj_body_part in relation_json:  # There are informations about human role
                    if sub in ['human_7', 'human_8'] or obj in ['human_7', 'human_8']:
                        continue
                    if 'human_' in sub:
                        sub = human_name_json.get(sub, 'circulator').replace('circulating-nurse', 'circulator')  # default to circulator
                    if 'human_' in obj:
                        obj = human_name_json.get(obj, 'circulator').replace('circulating-nurse', 'circulator')  # default to circulator
                    sub = sub.lower().replace('-', '_')
                    obj = obj.lower().replace('-', '_')
                    # Convert first letter of predicate to lower case
                    pred = pred[0].lower() + pred[1:]
                    if pred == 'operating':
                        pred = 'manipulating'
                    scan_objects.add(sub)
                    scan_objects.add(obj)
                    all_relations.add(pred)
                    scan_relationships.add((sub, obj, pred))

                infer_lyingon(scan_objects, scan_relationships)
                infer_operating_table_rels(scan_objects, scan_relationships)
                infer_holding_instrument(scan_objects, scan_relationships)
                # infer_closeto(scan_objects, scan_relationships)
                check_unique_relation(scan_objects, scan_relationships, relation_json_path, take)
                scan_objects.add('instrument')
                all_classes.update(scan_objects)
                take_name = f'{str(take).zfill(3)}_4DOR'
                samples.append(
                    {'take_name': take_name, 'frame_id': frame_id, 'relationships': sorted(scan_relationships)})

    if INCLUDE_MMOR:
        for take in tqdm(MMOR_TAKE_NAMES, desc='Loading MMOR'):
            root_path = MMOR_DATA_ROOT_PATH / MMOR_TAKE_NAME_TO_FOLDER.get(take, take)
            relations_path = root_path / 'relation_labels'
            if not relations_path.exists():
                # try to see if it exists with a suffix
                relations_path = root_path / f'relation_labels_{take}'
                if not relations_path.exists():
                    print(f'Still not found for {take}')
                    continue
            sample_every = 1
            if take == '007_TKA':  # take 007 is labeled very densely and clearly dominates the validation dataset if present. We want to only take every 3. element to keep it reasonable
                sample_every = 3
            all_jsons_list = sorted(list(relations_path.glob('*.json')))
            print(f'{len(all_jsons_list)} jsons found for {take}')
            for j_idx, relation_json_path in enumerate(all_jsons_list):
                if j_idx % sample_every != 0:
                    continue
                frame_id = relation_json_path.name.replace('.json', '')
                scan_objects = set()
                scan_relationships = set()
                with relation_json_path.open() as f:
                    info_json = json.load(f)
                relation_json = info_json['rel_annotations']
                for sub, pred, obj in relation_json:
                    sub = sub.lower().replace('-', '_')
                    obj = obj.lower().replace('-', '_')
                    sub = 'operating_table' if sub == 'ot' else sub
                    obj = 'operating_table' if obj == 'ot' else obj
                    sub = 'anesthesia_equipment' if sub == 'ae' else sub
                    obj = 'anesthesia_equipment' if obj == 'ae' else obj
                    sub = 'anaesthetist' if sub == 'anest' else sub
                    obj = 'anaesthetist' if obj == 'anest' else obj
                    # Convert first letter of predicate to lower case
                    pred = pred[0].lower() + pred[1:]
                    if pred == 'operating':
                        pred = 'manipulating'
                    scan_objects.add(sub)
                    scan_objects.add(obj)
                    all_relations.add(pred)
                    scan_relationships.add((sub, obj, pred))

                infer_lyingon(scan_objects, scan_relationships)
                infer_operating_table_rels(scan_objects, scan_relationships)
                # infer_closeto(scan_objects, scan_relationships)
                check_unique_relation(scan_objects, scan_relationships, relation_json_path, take)
                all_classes.update(scan_objects)
                take_name = f'{take}_MMOR'
                samples.append(
                    {'take_name': take_name, 'frame_id': frame_id, 'relationships': sorted(scan_relationships)})

    all_relations = sorted(all_relations)
    train_relationship_json = []
    val_relationship_json = []
    test_relationship_json = []

    for scan in samples:
        if '_4DOR' in scan['take_name']:
            take_idx = int(scan['take_name'].replace('_4DOR', ''))
            if take_idx in OR4D_TAKE_SPLIT['train']:
                train_relationship_json.append(scan)
            elif take_idx in OR4D_TAKE_SPLIT['val']:
                val_relationship_json.append(scan)
            elif take_idx in OR4D_TAKE_SPLIT['test']:
                test_relationship_json.append(scan)
            else:
                raise Exception('TAKE SPLIT UNKNOWN')
        elif '_MMOR' in scan['take_name']:
            take_idx = scan['take_name'].replace('_MMOR', '')
            if take_idx in MMOR_SPLIT_TO_TAKES['train']:
                train_relationship_json.append(scan)
            elif take_idx in MMOR_SPLIT_TO_TAKES['val']:
                val_relationship_json.append(scan)
            elif take_idx in MMOR_SPLIT_TO_TAKES['test']:
                test_relationship_json.append(scan)
            elif take_idx in MMOR_SPLIT_TO_TAKES['short_clips']:
                continue
            else:
                raise Exception('TAKE SPLIT UNKNOWN')
        else:
            raise Exception(f'Unknown take_name: {scan["take_name"]}')

    with save_classes_path.open('w') as f:
        f.write('\n'.join(sorted(all_classes)))

    with save_relationships_path.open('w') as f:
        f.write('\n'.join(sorted(all_relations)))

    with save_train_json_path.open('w') as f:
        json.dump(train_relationship_json, f)

    with save_val_json_path.open('w') as f:
        json.dump(val_relationship_json, f)

    with save_test_json_path.open('w') as f:
        json.dump(test_relationship_json, f)

    # report how many classes we have, how many relations we have, how many train, val, test takes we have
    print(f'Number of classes: {len(all_classes)}')
    print(f'Number of relations: {len(all_relations)}')
    print(f'Number of train takes: {len(train_relationship_json)}')
    print(f'Number of val takes: {len(val_relationship_json)}')
    print(f'Number of test takes: {len(test_relationship_json)}')


if __name__ == '__main__':
    main()
