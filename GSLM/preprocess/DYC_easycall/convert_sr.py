import os
import nlp2
import pyaudioconvert as pac
from pathlib import Path
from tqdm import tqdm

source_path = '/storage/SpeechPrompt-v2/data/easycall/EasyCall'
target_path = '/storage/SpeechPrompt-v2/data/easycall/EasyCall_sr16k'

for dir in tqdm(list(nlp2.get_files_from_dir(source_path, match='wav'))):
    rel_dir = os.path.relpath(dir, source_path)
    target_dir = Path(target_path, rel_dir)
    os.makedirs(target_dir.parent, exist_ok=True)
    pac.convert_wav_to_16bit_mono(dir, target_dir)