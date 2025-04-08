import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
import time

# Thêm Xtts vào danh sách globals an toàn
torch.serialization.add_safe_globals([XttsConfig])
torch.serialization.add_safe_globals([XttsAudioConfig])
torch.serialization.add_safe_globals([BaseDatasetConfig])
torch.serialization.add_safe_globals([XttsArgs])
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

input_audio_path = "voice_fixed_jp.wav"
output_audio_path = "output_audio.wav"

def text_to_speech(text):
    start_time = time.time()
    tts.tts_to_file(text=text, speaker_wav=input_audio_path, language='ja', file_path=output_audio_path)
    end_time = time.time()
    print(f"Time taken to generate audio: {end_time - start_time} seconds")
    return output_audio_path

if __name__ == "__main__":
    text_to_speech("こんばんは、お元気ですか？")