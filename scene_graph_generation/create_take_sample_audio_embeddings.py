from pathlib import Path

import numpy as np
import torch
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from helpers.configurations import MMOR_TAKE_NAMES, MMOR_DATA_ROOT_PATH


class AudioDataset(Dataset):
    def __init__(self, audio_dir, export_dir):
        self.audio_dir = Path(audio_dir)
        self.audio_files = sorted(self.audio_dir.glob("*.mp3"))  # Assuming .mp3
        # if already the corresponding .pt exists, we skip it
        self.exported_audio_files = set(export_dir.glob("*.pt"))
        # keep audio files without corresponding exported files
        self.audio_files = [audio_file for audio_file in self.audio_files if (export_dir / f'{audio_file.stem}.pt') not in self.exported_audio_files]
        self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio = AudioSegment.from_file(str(audio_path))
        audio = np.array(audio.get_array_of_samples())
        audio = self.clap_processor(audios=audio, return_tensors="pt", sampling_rate=48000)
        return {'audio': audio, 'filename': audio_path.stem}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap_model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    for take in tqdm(MMOR_TAKE_NAMES):
        take_audio_samples_path = MMOR_DATA_ROOT_PATH / 'take_audio_per_timepoint' / take
        take_audio_embeddings_export_path = MMOR_DATA_ROOT_PATH / 'take_audio_embeddings_per_timepoint' / take
        take_audio_embeddings_export_path.mkdir(exist_ok=True, parents=True)

        take_audio_dataset = AudioDataset(take_audio_samples_path, take_audio_embeddings_export_path)
        take_dataloader = DataLoader(take_audio_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x: x, pin_memory=True)
        for batch in tqdm(take_dataloader, desc=f'Processing take {take}'):  # we don't currently support batchsize>1
            batch = batch[0]
            filename = batch['filename']
            export_file_path = take_audio_embeddings_export_path / f'{filename}.pt'
            if export_file_path.exists():
                continue

            with torch.no_grad():
                inputs = batch['audio'].to(device)
                audio_embed = clap_model.get_audio_features(**inputs)[0]

            torch.save(audio_embed, export_file_path)


if __name__ == '__main__':
    '''First required that you run create_take_sample_audios'''
    main()
