import json
import subprocess
from functools import partial

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from helpers.configurations import MMOR_TAKE_NAMES, MMOR_DATA_ROOT_PATH


def _generate_cropped_audio(input_audio, start_timestamp, end_timestamp, output_audio):
    command = [
        'ffmpeg', '-y', '-i', input_audio,
        '-ss', str(start_timestamp),
        '-t', str(end_timestamp - start_timestamp + 1),  # duration
        # '-b:a', '128k',
        '-c:a', 'copy',
        output_audio
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress output.


def _helper_export_take_timepoint_audio_sample(timestamp_info, input_audio_path, export_audio_path, n_seconds=1):
    '''
    An entire take audio is given as input_audio_path. Also a timepoint is given.
    This function will extract the last n_seconds before this timepoint, and export it as a seperate file. The file will be named as the timepoint.
    We will need to get the correct start and end AUDIO SECONDS for this to work.
    Timestamps are in 1 fps, so it directly corresponds to the seconds


    '''
    idx, info = timestamp_info
    idx_str = str(idx).zfill(6)
    end_second = idx
    begin_second = max(0, end_second - n_seconds)

    _generate_cropped_audio(input_audio_path, begin_second, end_second, export_audio_path / f'{idx_str}.mp3')


def main():
    for take in tqdm(MMOR_TAKE_NAMES):
        print(f'Processing take {take}')
        JSON_PATH = MMOR_DATA_ROOT_PATH / 'take_jsons' / f'{take}.json'
        INPUT_PATH = MMOR_DATA_ROOT_PATH / 'take_audios' / f'{take}.mp3'
        EXPORT_PATH = MMOR_DATA_ROOT_PATH / 'take_audio_per_timepoint' / take
        EXPORT_PATH.mkdir(exist_ok=True)

        with JSON_PATH.open() as f:
            take_json = json.load(f)
            timestamps = take_json['timestamps']
            timestamps = {int(k): v for k, v in timestamps.items()}

        print('Exporting Audios per take')
        process_map(partial(_helper_export_take_timepoint_audio_sample, input_audio_path=INPUT_PATH, export_audio_path=EXPORT_PATH),
                    sorted(timestamps.items()), max_workers=24, chunksize=10)


if __name__ == '__main__':
    main()
