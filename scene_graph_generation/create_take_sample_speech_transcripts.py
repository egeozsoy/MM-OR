import json
from functools import partial

import pysrt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from helpers.configurations import MMOR_TAKE_NAMES, MMOR_DATA_ROOT_PATH


def _helper_export_take_timepoint_transcript_sample(timestamp_info, subs, export_transcript_path):
    '''
    Timestamps are in 1 fps, so it directly corresponds to the seconds
    '''
    idx, _ = timestamp_info
    idx_str = str(idx).zfill(6)
    end_second = idx
    # Convert timestamp to pysrt time format (hours, minutes, seconds)
    time_point = pysrt.SubRipTime.from_ordinal(end_second * 1000)
    # Get all subtitles before the given timestamp
    before_subs = [sub for sub in subs if sub.end.to_time() <= time_point.to_time()]
    last_4_subs = before_subs[-4:]
    last_4_subs_str = ' '.join([sub.text for sub in last_4_subs])
    last_4_subs_str = last_4_subs_str[-200:]  # take only the last 200 characters
    # save it to json
    with open(export_transcript_path / f'{idx_str}.json', 'w') as f:
        json.dump({'text': last_4_subs_str}, f)


def main():
    for take in tqdm(MMOR_TAKE_NAMES):
        print(f'Processing take {take}')
        JSON_PATH = MMOR_DATA_ROOT_PATH / 'take_jsons' / f'{take}.json'
        INPUT_PATH = MMOR_DATA_ROOT_PATH / 'take_transcripts' / f'{take}.srt'
        EXPORT_PATH = MMOR_DATA_ROOT_PATH / 'take_transcripts_per_timepoint' / take
        EXPORT_PATH.mkdir(exist_ok=True, parents=True)

        with JSON_PATH.open() as f:
            take_json = json.load(f)
            timestamps = take_json['timestamps']
            timestamps = {int(k): v for k, v in timestamps.items()}

        # load the srt file and parse it.
        subs = pysrt.open(INPUT_PATH)

        print('Exporting Speech Transcripts per take')
        process_map(partial(_helper_export_take_timepoint_transcript_sample, subs=subs, export_transcript_path=EXPORT_PATH),
                    sorted(timestamps.items()), max_workers=8, chunksize=100)


if __name__ == '__main__':
    main()
