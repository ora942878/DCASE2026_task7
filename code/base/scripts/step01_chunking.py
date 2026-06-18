from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from configs.CFG_PATH import PATH
from configs.CFG_PATH import CFG

duration = CFG.duration
sr = CFG.sample_rate

"""
Split audio into 4-second chunks.

This version follows the official chunking style, with only one change:
it reads the English class name from the official csv and writes chunks as:

    <wav_id>-<chunk_id>-<class>.wav

Example:
    0000.wav, alarm -> 0000-00-alarm.wav
"""

# ======= tools =======
def _safe_label_name(label):
    return str(label).strip().lower()

def _load_dict_label_from_csv(csv_path:Path):
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError(f"Cannot find 'filename' column in {csv_path}. Columns: {list(df.columns)}")
    if "class" not in df.columns:
        raise ValueError(f"Cannot find 'class' column in {csv_path}. Columns: {list(df.columns)}")

    label_dict = {}
    for index, row in df.iterrows():
        wav_name = Path(row["filename"]).name
        label_name = _safe_label_name(row["class"])
        label_dict[wav_name] = label_name
    return label_dict

def _chunk_one_folder(wav_path:Path, csv_path:Path, output_path:Path):
    wav_path = Path(wav_path)
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    label_dict = _load_dict_label_from_csv(csv_path)
    audio_list = sorted(wav_path.glob("*.wav")) # a Path list

    tgt_len = int(sr * duration)
    hop_len = tgt_len
    min_len = tgt_len // 2

    for audio in tqdm(audio_list):
        if audio.name not in label_dict:
            print(f"{audio.name} not in label_dict")
            continue

        label_name = label_dict[audio.name]
        x, _ = librosa.load(str(audio), sr=sr) # resample function, returns data,sr

        if len(x) == 0:
            print(f"empty audio skipped: {audio}")
            continue

        num_of_new_segments = (len(x) - tgt_len + hop_len - 1) // hop_len + 1

        for k in range(num_of_new_segments):
            start = k * hop_len
            x_segment = x[start:start + tgt_len]
            assert len(x_segment) > 0

            if num_of_new_segments > 1 and len(x_segment) < min_len:
                break
            if len(x_segment) < tgt_len:
                x_segment = np.pad(x_segment, (0, tgt_len - len(x_segment)))

            assert len(x_segment) == tgt_len

            sf.write(output_path / f'{audio.stem}-{k:02d}-{label_name}.wav', x_segment, sr)


def main():
    jobs = [
        (PATH.TRAIN_D2, PATH.TRAIN_D2_CSV, PATH.TRAIN_D2_CHUNK_4),
        (PATH.TEST_D2,  PATH.TEST_D2_CSV,  PATH.TEST_D2_CHUNK_4),
        (PATH.TRAIN_D3, PATH.TRAIN_D3_CSV, PATH.TRAIN_D3_CHUNK_4),
        (PATH.TEST_D3,  PATH.TEST_D3_CSV,  PATH.TEST_D3_CHUNK_4),
    ]

    for audio_dir, csv_path, output_dir in jobs:
        if not Path(audio_dir).exists():
            print(f"audio directory does not exist, skipped: {audio_dir}")
            continue

        if not Path(csv_path).exists():
            print(f"csv file does not exist, skipped: {csv_path}")
            continue

        _chunk_one_folder(audio_dir, csv_path, output_dir)


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     from pathlib import Path
#     from configs.CFG_PATH import CFG
#     from configs.CFG_PATH import PATH
#
#     audio_list = sorted(Path(PATH.TEST_D2).glob("*.wav"))
#     print(len(list(Path(PATH.TEST_D2).glob("*.wav"))))
#     x, sr = librosa.load(str(audio_list[1]), sr=sr)
#     print(sr,len(x))