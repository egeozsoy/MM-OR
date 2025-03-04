from pathlib import Path

OR4D_TAKE_SPLIT = {'train': [1, 3, 5, 7, 9, 10], 'val': [4, 8], 'test': [2, 6]}
OR4D_TAKE_FOLDERS = ['export_holistic_take1_processed', 'export_holistic_take2_processed', 'export_holistic_take3_processed', 'export_holistic_take4_processed',
                     'export_holistic_take5_processed',
                     'export_holistic_take6_processed', 'export_holistic_take7_processed', 'export_holistic_take8_processed', 'export_holistic_take9_processed',
                     'export_holistic_take10_processed']
# thesee are the takes that are actually used
OR4D_TAKE_NAMES = ['001_4DOR', '002_4DOR', '003_4DOR', '004_4DOR', '005_4DOR', '006_4DOR', '007_4DOR', '008_4DOR', '009_4DOR', '010_4DOR']
OR4D_TAKE_NAME_TO_FOLDER = {'001_4DOR': 'export_holistic_take1_processed', '002_4DOR': 'export_holistic_take2_processed', '003_4DOR': 'export_holistic_take3_processed',
                            '004_4DOR': 'export_holistic_take4_processed', '005_4DOR': 'export_holistic_take5_processed', '006_4DOR': 'export_holistic_take6_processed',
                            '007_4DOR': 'export_holistic_take7_processed', '008_4DOR': 'export_holistic_take8_processed', '009_4DOR': 'export_holistic_take9_processed',
                            '010_4DOR': 'export_holistic_take10_processed'}

OR4D_SPLIT_TO_TAKES = {
    "train": ['001_4DOR', '003_4DOR', '005_4DOR', '007_4DOR', '009_4DOR', '010_4DOR'],
    "small_train": ['001_4DOR', '005_4DOR', '007_4DOR', '009_4DOR'],
    "mini_train": ['001_4DOR'],  # just for debugging
    "val": ['004_4DOR', '008_4DOR'],
    "test": ['002_4DOR', '006_4DOR']
}

OBJECT_COLOR_MAP = {
    'anesthesia_equipment': (0.96, 0.576, 0.65),
    'operating_table': (0.2, 0.83, 0.72),
    'instrument_table': (0.93, 0.65, 0.93),
    'secondary_table': (0.90, 0.30, 0.63),
    'instrument': (1.0, 0.811, 0.129),
    'object': (0.61, 0.48, 0.04),
    'Patient': (0, 1., 0),
    'human_0': (1., 0., 0),
    'human_1': (0.9, 0., 0),
    'human_2': (0.85, 0., 0),
    'human_3': (0.8, 0., 0),
    'human_4': (0.75, 0., 0),
    'human_5': (0.7, 0., 0),
    'human_6': (0.65, 0., 0),
    'human_7': (0.6, 0., 0)
    # 'drill': (0.90, 0.30, 0.30),
    # 'saw': (1.0, 0.811, 0.129),
    # '': (0.90, 0.30, 0.30),
    # 'c-arm': (1.0, 0.811, 0.129),
    # 'c-arm_base': (0.61, 0.48, 0.04),
    # 'unidentified_1': (0.34, 0.65, 0.36),
    # 'unidentified_2': (0.22, 0.3, 0.83)
}

OBJECT_LABEL_MAP = {
    'anesthesia_equipment': 0,
    'operating_table': 1,
    'instrument_table': 2,
    'secondary_table': 3,
    'instrument': 4,
    'object': 5,
    'Patient': 9,
    'human_0': 10,
    'human_1': 11,
    'human_2': 12,
    'human_3': 13,
    'human_4': 14,
    'human_5': 15,
    'human_6': 16,
    'human_7': 17
}

OR_4D_DATA_ROOT_PATH = Path('../4D-OR_data')
EXPORT_HOLISTICS_PATHS = list(OR_4D_DATA_ROOT_PATH.glob('export_holistic_take*'))

MMOR_TAKE_FOLDERS = ['001_PKA', '002_PKA', '003_TKA', '004_PKA', '005_TKA', '006_PKA', '007_TKA', '008_PKA', '009_TKA', '010_PKA', '011_TKA', '012_1_PKA', '013_PKA', '014_PKA', '015-018_PKA',
                     '019-022_PKA', '023-032_PKA', '033_PKA', '035_PKA', '036_PKA', '037_TKA', '038_TKA']
# thesee are the takes that are actually used
MMOR_TAKE_NAMES = ['001_PKA', '002_PKA', '003_TKA', '004_PKA', '005_TKA', '006_PKA', '007_TKA', '008_PKA', '009_TKA', '010_PKA', '011_TKA', '012_1_PKA', '012_2_PKA', '013_PKA', '014_PKA',
                   '015_PKA', '016_PKA', '017_PKA', '018_1_PKA', '018_2_PKA', '019_PKA', '020_PKA', '021_PKA', '022_PKA', '023_PKA', '024_PKA', '025_PKA', '026_PKA', '027_PKA', '028_PKA',
                   '029_PKA', '030_PKA', '031_PKA', '032_PKA', '033_PKA', '035_PKA', '036_PKA', '037_TKA', '038_TKA']
# seperate some takes even further. Rely on the take_jsons. Make sure combined takes are getting loaded correctly
MMOR_TAKE_NAME_TO_FOLDER = {'012_1_PKA': '012_PKA', '012_2_PKA': '012_PKA', '015_PKA': '015-018_PKA', '016_PKA': '015-018_PKA', '017_PKA': '015-018_PKA', '018_1_PKA': '015-018_PKA',
                            '018_2_PKA': '015-018_PKA',
                            '019_PKA': '019-022_PKA', '020_PKA': '019-022_PKA', '021_PKA': '019-022_PKA', '022_PKA': '019-022_PKA', '023_PKA': '023-032_PKA', '024_PKA': '023-032_PKA',
                            '025_PKA': '023-032_PKA', '026_PKA': '023-032_PKA',
                            '027_PKA': '023-032_PKA', '028_PKA': '023-032_PKA', '029_PKA': '023-032_PKA', '030_PKA': '023-032_PKA', '031_PKA': '023-032_PKA', '032_PKA': '023-032_PKA'}

MMOR_SPLIT_TO_TAKES = {
    "train": ['001_PKA', '003_TKA', '005_TKA', '006_PKA', '008_PKA', '010_PKA', '012_1_PKA', '012_2_PKA', '035_PKA', '037_TKA'],
    "small_train": ['001_PKA', '003_TKA', '035_PKA', '037_TKA', '005_TKA'],
    "mini_train": ['013_PKA'],  # just for debugging
    "val": ['002_PKA', '007_TKA', '009_TKA'],
    "test": ['004_PKA', '011_TKA', '036_PKA', '038_TKA'],
    'short_clips': ['013_PKA', '014_PKA', '015_PKA', '016_PKA', '017_PKA', '018_1_PKA', '018_2_PKA', '019_PKA', '020_PKA', '021_PKA', '022_PKA', '023_PKA', '024_PKA', '025_PKA', '026_PKA', '027_PKA',
                    '028_PKA', '029_PKA', '030_PKA', '031_PKA', '032_PKA', '033_PKA']  # these are not considered main takes, but can be used for some other things including anomaly detection
}
MMOR_DATA_ROOT_PATH = Path('../MM-OR_data')

TRACKER_OBJECT_MAP = {
    '8000050': 'base_array',
    '8000056': 'calibration_array',
    '8000057': 'upper_tracker',
    '8000058': 'lower_tracker',
    '8000054': 'green_tip',
    '8000053': 'blue_tip',
    '8000999': 'calibration_array'
}

DEPTH_SCALING = 2000

LIMBS = [
    [5, 4],  # (righthip-lefthip)
    [9, 7],  # (rightwrist - rightelbow)
    [7, 3],  # (rightelbow - rightshoulder)
    [2, 6],  # (leftshoulder - leftelbow)
    [6, 8],  # (leftelbow - leftwrist)
    [5, 3],  # (righthip - rightshoulder)
    [4, 2],  # (lefthip - leftshoulder)
    [3, 1],  # (rightshoulder - neck)
    [2, 1],  # (leftshoulder - neck)
    [1, 0],  # (neck - head)
    [10, 4],  # (leftknee,lefthip),
    [11, 5],  # (rightknee,righthip),
    [12, 10],  # (leftfoot,leftknee),
    [13, 11]  # (rightfoot,rightknee),

]

HUMAN_POSE_COLOR_MAP = {
    0: (255, 0, 0),
    1: (200, 0, 0),
    2: (68, 240, 65),
    3: (50, 166, 48),
    4: (65, 201, 224),
    5: (42, 130, 145),
    6: (66, 179, 245),
    7: (44, 119, 163),
    8: (245, 173, 66),
    9: (186, 131, 50)
}

IDX_TO_BODY_PART = ['head', 'neck', 'leftshoulder', 'rightshoulder', 'lefthip', 'righthip', 'leftelbow', 'rightelbow', 'leftwrist', 'rightwrist', 'leftknee',
                    'rightknee', 'leftfoot', 'rightfoot']

STATIONARY_OBJECTS = ['instrument_table',
                      'secondary_table']  # We might have to seperate these into different takes, if an object is only stationary in one take etc.
