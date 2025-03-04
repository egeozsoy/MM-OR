scene_graph_name_to_vocab_idx = {
    'anaesthetist': 0,
    'anesthesia_equipment': 1,
    'assistant_surgeon': 2,
    'c_arm': 3,
    'circulator': 4,
    'drape': 5,
    'drill': 6,
    'hammer': 7,
    'head_surgeon': 8,
    'instrument': 9,
    'instrument_table': 10,
    'mako_robot': 11,
    'monitor': 12,
    'mps': 13,
    'mps_station': 14,
    'nurse': 15,
    'operating_table': 16,
    'patient': 17,
    'saw': 18,
    'secondary_table': 19,
    'student': 20,
    'tracker': 21,
    'unrelated_person': 22,
    'assisting': 23,
    'calibrating': 24,
    'cementing': 25,
    'cleaning': 26,
    'closeto': 27,
    'cutting': 28,
    'drilling': 29,
    'hammering': 30,
    'holding': 31,
    'lyingon': 32,
    'manipulating': 33,
    'preparing': 34,
    'sawing': 35,
    'scanning': 36,
    'suturing': 37,
    'touching': 38,
}
vocab_idx_to_scene_graph_name = {v: k for k, v in scene_graph_name_to_vocab_idx.items()}

synonyms = {
    'anesthesia_equipment': ['anaesthesia_equipment', 'anesthesia equipment', 'anaesthetist_station'],
    'closeto': ['close', 'close to'],
    'instrument': ['tool'],
    'operating_table': ['opertating_table'],
}

role_synonyms = {
    'head_surgeon': ['head_surgent'],
    'anaesthetist': ['anesthetist'],
}


# Reverse synonym mapping
def reverse_synonym_mapping(synonyms_dict):
    reversed_dict = {}
    for key, synonyms_list in synonyms_dict.items():
        for synonym in synonyms_list:
            reversed_dict[synonym] = key
    return reversed_dict


# Applying the function
reversed_synonyms = reverse_synonym_mapping(synonyms)
reversed_role_synonyms = reverse_synonym_mapping(role_synonyms)


def map_scene_graph_name_to_vocab_idx(name):
    name = name.lower()
    # Synonym mapping
    if name in reversed_synonyms:
        name = reversed_synonyms[name]
    return scene_graph_name_to_vocab_idx[name]


def map_vocab_idx_to_scene_graph_name(vocab_idx):
    return vocab_idx_to_scene_graph_name[vocab_idx]
