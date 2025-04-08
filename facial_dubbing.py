# from utils.deep_speech import DeepSpeech
from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DINetInferenceOptions
from models.DINet import DINet

import numpy as np
import glob
import os
import cv2
import torch
import subprocess
import random
from collections import OrderedDict
import time
from concurrent.futures import ThreadPoolExecutor
import gc
import requests

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def extract_frames_from_video(video_path,save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)
    return (int(frame_width),int(frame_height))

opt = DINetInferenceOptions().parse_args()
if not os.path.exists(opt.source_video_path):
    raise ('wrong video path : {}'.format(opt.source_video_path))
############################################## extract frames from source video ##############################################
print('extracting frames from video: {}'.format(opt.source_video_path))
video_frame_dir = opt.source_video_path.replace('.mp4', '')
if not os.path.exists(video_frame_dir):
    os.mkdir(video_frame_dir)
video_size = extract_frames_from_video(opt.source_video_path,video_frame_dir)

# if not os.path.exists(opt.deepspeech_model_path):
#         raise ('pls download pretrained model of deepspeech')
# DSModel = DeepSpeech(opt.deepspeech_model_path)

############################################## load facial landmark ##############################################
print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
if not os.path.exists(opt.source_openface_landmark_path):
    print('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))

# video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(np.int)
video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(int)

video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
if len(video_frame_path_list) != video_landmark_data.shape[0]:
    print('video frames are misaligned with detected landmarks')
video_frame_path_list.sort()
video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)

############################################## load pretrained model weight ##############################################
print('loading pretrained model from: {}'.format(opt.pretrained_clip_DINet_path))
model = DINet(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
if not os.path.exists(opt.pretrained_clip_DINet_path):
    raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DINet_path))
state_dict = torch.load(opt.pretrained_clip_DINet_path)['state_dict']['net_g']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove module.
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# def extract_deepspeech_features(audio_path):
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f'Invalid audio path: {audio_path}')
#     print(f'Extracting DeepSpeech features from: {audio_path}')
#     return DSModel.compute_audio_feature(audio_path)
url = 'http://127.0.0.1:8000/compute_audio_feature/'
def extract_deepspeech_features(audio_path):
    print('extracting deepspeech feature from: {}'.format(audio_path))
    data = {'audio_path': audio_path}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        ds_features = response.json()
        return np.array(ds_features['ds_features'])
    else:
        raise Exception(f"Failed to fetch DeepSpeech features: {response.status_code}")

def align_frames_with_audio(video_frames, video_landmarks, audio_length):
    print('aligning frames with driving audio')
    frame_count = len(video_frames)
    if frame_count >= audio_length:
        return video_frames[:audio_length], video_landmarks[:audio_length]
    else:
        multiplier = audio_length // frame_count
        remainder = audio_length % frame_count
        return video_frames * multiplier + video_frames[:remainder], \
               np.concatenate([video_landmarks] * multiplier + [video_landmarks[:remainder]], axis=0)

def crop_and_resize_face(img, landmarks, crop_radius, resize_w, resize_h):
    crop_radius_1_4 = crop_radius // 4
    cropped = img[
        landmarks[29, 1] - crop_radius:landmarks[29, 1] + 2 * crop_radius + crop_radius_1_4,
        landmarks[33, 0] - crop_radius - crop_radius_1_4:landmarks[33, 0] + crop_radius + crop_radius_1_4,
        :
    ]
    original_size = (cropped.shape[1], cropped.shape[0])
    return cv2.resize(cropped, (resize_w, resize_h)), original_size

def select_reference_images(video_frames, video_landmarks, video_size, resize_w, resize_h, num_refs=5):
    print(f'Selecting {num_refs} reference images')
    ref_images = []
    indices = random.sample(range(5, len(video_frames) - 2), num_refs)
    for idx in indices:
        crop_flag, crop_radius = compute_crop_radius(video_size, video_landmarks[idx - 5:idx])
        if not crop_flag:
            raise ValueError('Significant facial size changes detected, unsupported!')
        
        ref_img = cv2.imread(video_frames[idx - 3])[:, :, ::-1]
        ref_landmark = video_landmarks[idx - 3]
        ref_crop, _ = crop_and_resize_face(ref_img, ref_landmark, crop_radius, resize_w, resize_h)
        ref_images.append(ref_crop / 255.0)
    
    ref_tensor = torch.from_numpy(np.concatenate(ref_images, axis=2)).permute(2, 0, 1).unsqueeze(0).float().cuda()
    return ref_tensor

def process_frame(i, video_frames, video_landmarks, ds_feature_pad, ref_tensor, video_size, resize_w, resize_h):
    # print(f'Processing frame {i - 5}/{ds_feature_pad.shape[0] - 5}')
    crop_flag, crop_radius = compute_crop_radius(video_size, video_landmarks[i - 5:i], random_scale=1.05)
    if not crop_flag:
        raise ValueError('Significant facial size changes detected, unsupported!')
    
    frame = cv2.imread(video_frames[i - 3])[:, :, ::-1] # change BGR to RGB
    frame_landmark = video_landmarks[i - 3]
    cropped_face, original_size = crop_and_resize_face(frame, frame_landmark, crop_radius, resize_w, resize_h)
    cropped_face = cropped_face / 255.0
    cropped_face[opt.mouth_region_size // 2:opt.mouth_region_size // 2 + opt.mouth_region_size,
                 opt.mouth_region_size // 8:opt.mouth_region_size // 8 + opt.mouth_region_size, :] = 0
    
    frame_tensor = torch.from_numpy(cropped_face).float().cuda().permute(2, 0, 1).unsqueeze(0)
    ds_tensor = torch.from_numpy(ds_feature_pad[i - 5:i]).permute(1, 0).unsqueeze(0).float().cuda()
    
    with torch.no_grad():
        pred_frame = model(frame_tensor, ref_tensor, ds_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    resized_pred_frame = cv2.resize(pred_frame, original_size)
    frame[
        frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + 2 * crop_radius,
        frame_landmark[33, 0] - crop_radius - crop_radius // 4:frame_landmark[33, 0] + crop_radius + crop_radius // 4,
    ] = resized_pred_frame[:crop_radius * 3]
    return frame

def synthesize_video(video_frames, video_landmarks, ds_feature_pad, ref_tensor, output_path, resize_w, resize_h):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    print('Synthesizing frames...')
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    
    # frame_count = ds_feature_pad.shape[0]
    # for i in range(5, frame_count - 5):
    #     # print('synthesizing {}/{} frame'.format(i - 5, frame_count - 5))
    #     frame = process_frame(i, video_frames, video_landmarks, ds_feature_pad, ref_tensor, video_size, resize_w, resize_h)
    #     video_writer.write(frame[:, :, ::-1])

    num_workers = os.cpu_count() if os.cpu_count() < 32 else 32
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        frames = list(executor.map(lambda i: process_frame(i, video_frames, video_landmarks, ds_feature_pad, ref_tensor, video_size, resize_w, resize_h), range(5, ds_feature_pad.shape[0])))
    for frame in frames:
        video_writer.write(frame[:, :, ::-1])

    del frames
    video_writer.release()
    gc.collect()

def add_audio_to_video(video_path, audio_path, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    cmd = f'ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {output_path}'
    print('Adding audio to video...')
    subprocess.call(cmd, shell=True)
    print('Done!')

def facial_dubbing(audio_path=opt.driving_audio_path):
    if not os.path.exists(audio_path):
        print('wrong audio path : {}'.format(audio_path))
        return None
    ############################################## extract deep speech feature ##############################################
    start_time = time.time()
    ds_feature = extract_deepspeech_features(audio_path)
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
    end_time = time.time()
    print(f'Time taken to extract DeepSpeech features: {end_time - start_time} seconds')
    
    ############################################## align frame with driving audio ##############################################
    start_time = time.time()
    res_video_frame_path_list, res_video_landmark_data = align_frames_with_audio(video_frame_path_list_cycle, video_landmark_data_cycle, ds_feature.shape[0])
    
    res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                    + res_video_frame_path_list \
                                    + [video_frame_path_list_cycle[-1]] * 2
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
    end_time = time.time()
    print(f'Time taken to align frames with driving audio: {end_time - start_time} seconds')

    ############################################## randomly select 5 reference images ##############################################
    start_time = time.time()
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    ref_img_tensor = select_reference_images(res_video_frame_path_list_pad, res_video_landmark_data_pad, video_size, resize_w, resize_h)
    end_time = time.time()
    print(f'Time taken to select reference images: {end_time - start_time} seconds')

    ############################################## inference frame by frame ##############################################
    start_time = time.time()
    res_video_path = os.path.join(opt.res_video_dir,os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
    synthesize_video(res_video_frame_path_list_pad, res_video_landmark_data_pad, ds_feature_padding, ref_img_tensor, res_video_path, resize_w, resize_h)
    end_time = time.time()
    print(f'Time taken to synthesize video: {end_time - start_time} seconds')

    ############################################## add audio to video ##############################################
    start_time = time.time()
    video_add_audio_path = res_video_path.replace('.mp4', '_add_audio.mp4')
    add_audio_to_video(res_video_path, audio_path, video_add_audio_path)
    end_time = time.time()
    print(f'Time taken to add audio to video: {end_time - start_time} seconds')
    return video_add_audio_path

if __name__ == "__main__":
    output_video_path = facial_dubbing()
    print(output_video_path)