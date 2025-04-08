# styletts
```bash
conda activate stylettsja
cd StyleTTS_JP
python server.py --reference_audio "path/to/audio.wav"
```

# deepspeech
```bash
conda activate dinet1
cd DINet
python main_deepspeech_file.py
```

# main
```bash
conda activate dinet
cd DINet
python main.py
```

# ngrok
```bash
ngrok http --url=cicada-eager-hound.ngrok-free.app 19000
```

Open https://cicada-eager-hound.ngrok-free.app in your browser to see the result.

# Note
- For deepspeech and main, to specific the gpu, edit the CUDA_VISIBLE_DEVICES in the DInet/.env file
- To use other input video, edit the source_video_path and source_openface_landmark_path in DINet/config/config.py
- It'll be better to convert the video to 25fps:
# convert input video to 25fps
```bash
ffmpeg -i input.mp4 -filter:v "fps=25" -c:v libx264 -preset slow -crf 18 -c:a copy output.mp4
```
- To get the openface_landmark csv use OpenFace with these setting:

| Record | Recording settings |  OpenFace setting | View | Face Detector | Landmark Detector |
|--|--|--|--|--|--|
| 2D landmark & tracked videos | Mask aligned image | Use dynamic AU models | Show video  | Openface (MTCNN)| CE-CLM |
