# from utils.deep_speech import DeepSpeech

# DSModel = DeepSpeech('./asserts/output_graph.pb')
# ds_feature = DSModel.compute_audio_feature('output_audio.wav')

import torch
print("PyTorch version:", torch.__version__)
print("cuDNN version:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if len(gpus) > 1:
#     tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
# print(tf.__version__)
# print(tf.sysconfig.get_build_info()["cuda_version"])
# print(tf.sysconfig.get_build_info()["cudnn_version"])

# import tensorflow as tf

# def list_tensors(pb_path):
#     tf.compat.v1.disable_eager_execution()
#     with tf.io.gfile.GFile(pb_path, "rb") as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())

#     for node in graph_def.node:
#         print(node.name)

# model_path = "./asserts/output_graph.pb"  # Đường dẫn đến file model
# list_tensors(model_path)

# import wave
# from io import BytesIO

# import ffmpeg
# import numpy as np
# from deepspeech import Model


# def normalize_audio(audio):
#     out, err = (
#         ffmpeg.input("pipe:0")
#         .output(
#             "pipe:1",
#             f="WAV",
#             acodec="pcm_s16le",
#             ac=1,
#             ar="16k",
#             loglevel="error",
#             hide_banner=None,
#         )
#         .run(input=audio, capture_stdout=True, capture_stderr=True)
#     )
#     if err:
#         raise Exception(err)
#     return out


# class SpeechToTextEngine:
#     def __init__(self, model_path, scorer_path):
#         self.model = Model(model_path=model_path)
#         self.model.enableExternalScorer(scorer_path=scorer_path)

#     def read_audio(self, audio_path):
#         with wave.open(audio_path, 'rb') as wf:
#             assert wf.getnchannels() == 1, "Audio must be mono"
#             assert wf.getsampwidth() == 2, "Audio must be 16-bit"
#             audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
#         return audio
    
#     def run(self, audio_path):
#         audio = self.read_audio(audio_path)
#         result = self.model.stt(audio_buffer=audio)
#         return result

# if __name__ == '__main__':
#     model_path = "deepspeech-0.9.3-models.pbmm" 
#     scorer_path = 'deepspeech-0.9.3-models.scorer'
#     STTEngine = SpeechToTextEngine(model_path, scorer_path)
#     result = STTEngine.run('output_audio.wav')
#     print("result:", result)