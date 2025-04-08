ffmpeg -i voice.wav -acodec pcm_s16le -ar 16000 -ac 1 voice_fixed.wav

CUDA_VISIBLE_DEVICES=1 python inference_facial_dubbing.py \
        --mouth_region_size=256 \
        --source_video_path=./asserts/examples/test4.mp4 \
        --source_openface_landmark_path=./asserts/examples/test4.csv \
        --driving_audio_path=voice_fixed_jp.wav \
        --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth

TF_XLA_FLAGS="--tf_xla_cpu_global_jit" CUDA_VISIBLE_DEVICES=1 python main.py