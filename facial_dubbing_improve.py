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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
    print('total frames: {}'.format(frames))
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
video_size = extract_frames_from_video(opt.source_video_path, video_frame_dir)

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
video_frame_data_list = [cv2.imread(video_frame_path)[:, :, ::-1] for video_frame_path in video_frame_path_list] # change BGR to RGB
video_frame_data_list_cycle = video_frame_data_list + video_frame_data_list[::-1] # lặp video frame từ đầu đến cuối và ngược lại
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

def align_frames_with_audio(data_video_frames, video_landmarks, audio_length):
    print('aligning frames with driving audio')
    frame_count = len(data_video_frames)
    if frame_count >= audio_length:
        return data_video_frames[:audio_length], video_landmarks[:audio_length]
    else:
        multiplier = audio_length // frame_count
        remainder = audio_length % frame_count
        return data_video_frames * multiplier + data_video_frames[:remainder], \
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

def batch_compute_crop_radius(video_size, landmark_data_clip, random_scale=1.05):
    '''
    Batch processing version of crop radius calculation
    '''
    video_w, video_h = video_size[0], video_size[1]
    
    # Calculate max landmarks for each frame in each window
    landmark_max_clip = np.max(landmark_data_clip, axis=2)  # Shape: (batch, window_size, 2)
    
    # Calculate radius components
    radius_h = (landmark_max_clip[:, :, 1] - landmark_data_clip[:, :, 29, 1]) * random_scale
    radius_w = (landmark_data_clip[:, :, 54, 0] - landmark_data_clip[:, :, 48, 0]) * random_scale
    
    # Stack and find max radius for each frame
    radius_clip = np.max(np.stack([radius_h, radius_w], axis=2), axis=2) // 2  # Shape: (batch, window_size)
    
    # Find max radius for each window
    radius_max = np.max(radius_clip, axis=1)  # Shape: (batch,)
    radius_max = (radius_max // 4 + 1) * 4  # Ensure divisible by 4
    
    radius_max_1_4 = radius_max // 4
    
    # Calculate crop boundaries
    clip_min_h = landmark_data_clip[:, :, 29, 1] - radius_max[:, np.newaxis]
    clip_max_h = landmark_data_clip[:, :, 29, 1] + 2 * radius_max[:, np.newaxis] + radius_max_1_4[:, np.newaxis]
    clip_min_w = landmark_data_clip[:, :, 33, 0] - radius_max[:, np.newaxis] - radius_max_1_4[:, np.newaxis]
    clip_max_w = landmark_data_clip[:, :, 33, 0] + radius_max[:, np.newaxis] + radius_max_1_4[:, np.newaxis]
    
    # Check validity for each window
    valid = (
        (np.all(clip_min_h >= 0, axis=1)) &
        (np.all(clip_max_h <= video_h, axis=1)) &
        (np.all(clip_min_w >= 0, axis=1)) &
        (np.all(clip_max_w <= video_w, axis=1)) &
        (np.max(radius_clip, axis=1) <= 1.5 * np.min(radius_clip, axis=1))
    )
    
    return valid, radius_max

def batch_crop_and_resize(frames, landmarks, radius_list, resize_w, resize_h):
    batch_output = []
    original_sizes = []
    
    for frame, landmark, radius in zip(frames, landmarks, radius_list):
        try:
            # Ensure radius is integer
            radius = int(round(radius))
            crop_radius_1_4 = radius // 4
            
            # Calculate coordinates and ensure they are integers
            y_start = int(round(landmark[29, 1] - radius))
            y_end = int(round(landmark[29, 1] + 2 * radius + crop_radius_1_4))
            x_start = int(round(landmark[33, 0] - radius - crop_radius_1_4))
            x_end = int(round(landmark[33, 0] + radius + crop_radius_1_4))
            
            # Validate coordinates
            h, w = frame.shape[:2]
            y_start = max(0, y_start)
            y_end = min(h, y_end)
            x_start = max(0, x_start)
            x_end = min(w, x_end)
            
            # Perform cropping
            cropped = frame[y_start:y_end, x_start:x_end, :]
            original_size = (cropped.shape[1], cropped.shape[0])
            
            if cropped.size > 0:
                resized = cv2.resize(cropped, (resize_w, resize_h))
                batch_output.append(resized)
                original_sizes.append(original_size)
            else:
                raise ValueError("Invalid crop region")
                
        except Exception as e:
            print(f"Crop error: {e}, using fallback")
            fallback = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
            batch_output.append(fallback)
            original_sizes.append((resize_w, resize_h))
    
    return np.array(batch_output), original_sizes

def select_reference_images(data_video_frames, video_landmarks, video_size, resize_w, resize_h, num_refs=5):
    print(f'Selecting {num_refs} reference images')
    ref_images = []
    indices = random.sample(range(5, len(data_video_frames) - 2), num_refs)
    for idx in indices:
        crop_flag, crop_radius = compute_crop_radius(video_size, video_landmarks[idx - 5:idx])
        if not crop_flag:
            raise ValueError('Significant facial size changes detected, unsupported!')
        
        ref_img = data_video_frames[idx - 3]
        ref_landmark = video_landmarks[idx - 3]
        ref_crop, _ = crop_and_resize_face(ref_img, ref_landmark, crop_radius, resize_w, resize_h)
        ref_images.append(ref_crop / 255.0)
    
    ref_tensor = torch.from_numpy(np.concatenate(ref_images, axis=2)).permute(2, 0, 1).unsqueeze(0).float().cuda()
    return ref_tensor

def process_frame(i, data_video_frames, video_landmarks, ds_feature_pad, ref_tensor, video_size, resize_w, resize_h):
    # print(f'Processing frame {i - 5}/{ds_feature_pad.shape[0] - 5}')
    crop_flag, crop_radius = compute_crop_radius(video_size, video_landmarks[i - 5:i], random_scale=1.05)
    if not crop_flag:
        raise ValueError('Significant facial size changes detected, unsupported!')
    
    frame = data_video_frames[i - 3]
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

def synthesize_video(data_video_frames, video_landmarks, ds_feature_pad, ref_tensor, output_path, resize_w, resize_h):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    print('Synthesizing frames...')
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    
    # frame_count = ds_feature_pad.shape[0]
    # for i in range(5, frame_count - 5):
    #     # print('synthesizing {}/{} frame'.format(i - 5, frame_count - 5))
    #     frame = process_frame(i, data_video_frames, video_landmarks, ds_feature_pad, ref_tensor, video_size, resize_w, resize_h)
    #     video_writer.write(frame[:, :, ::-1])

    num_workers = os.cpu_count() if os.cpu_count() < 32 else 32
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        frames = list(executor.map(lambda i: process_frame(i, data_video_frames, video_landmarks, ds_feature_pad, ref_tensor, video_size, resize_w, resize_h), range(5, ds_feature_pad.shape[0])))
    for frame in frames:
        video_writer.write(frame[:, :, ::-1])

    del frames
    video_writer.release()
    gc.collect()

def batch_synthesize_video(data_video_frames, video_landmarks, ds_feature_pad, ref_tensor, output_path, resize_w, resize_h):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    print('Synthesizing frames...')
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    
    frame_count = ds_feature_pad.shape[0]
    batch_size = 32  # Adjust based on GPU memory
    
    # Precompute all valid windows and crop radii
    valid_mask = []
    radius_list = []
    for i in range(5, frame_count):
        window = video_landmarks[i-5:i]
        # valid, radius = batch_compute_crop_radius(video_size, window[np.newaxis, ...])
        valid, radius = compute_crop_radius(video_size, window, random_scale=1.05)
        valid_mask.append(valid)
        radius_list.append(radius)
    
    if not all(valid_mask):
        raise ValueError('Invalid crop detected in some frames')
    
    # Batch processing main loop
    for batch_start in range(0, len(radius_list), batch_size):
        batch_end = min(batch_start + batch_size, len(radius_list))
        batch_indices = range(batch_start, batch_end)
        
        # Prepare batch data
        batch_frames = []
        batch_landmarks = []
        batch_radius = []
        batch_ds_features = []
        
        for idx in batch_indices:
            actual_frame_idx = idx + 2  # Offset to match original indices
            batch_frames.append(data_video_frames[actual_frame_idx])
            batch_landmarks.append(video_landmarks[actual_frame_idx])
            batch_radius.append(radius_list[idx])
            batch_ds_features.append(ds_feature_pad[actual_frame_idx-2:actual_frame_idx+3])
        
        # Process cropping
        cropped_faces, original_sizes = batch_crop_and_resize(
            batch_frames, batch_landmarks, batch_radius, resize_w, resize_h
        )
        
        # Prepare model input
        mouth_h_start = opt.mouth_region_size // 2
        mouth_h_end = mouth_h_start + opt.mouth_region_size
        mouth_w_start = opt.mouth_region_size // 8
        mouth_w_end = mouth_w_start + opt.mouth_region_size
        cropped_faces[:, mouth_h_start:mouth_h_end, mouth_w_start:mouth_w_end, :] = 0

        # Convert to tensors
        face_tensors = torch.stack([
            torch.from_numpy(face/255.).permute(2,0,1).float().cuda() 
            for face in cropped_faces
        ])
        
        ds_tensors = torch.stack([
            torch.from_numpy(feats).permute(1,0).float().cuda()
            for feats in batch_ds_features
        ])
        
        # Model inference
        with torch.no_grad():
            B = len(batch_indices)
            ref_batch = ref_tensor.expand(B, -1, -1, -1)  # fast broadcast, tránh repeat
            pred_faces = model(face_tensors, ref_batch, ds_tensors)  # [B, C, H, W]
        
        # Post-process predictions
        pred_faces = pred_faces.permute(0, 2, 3, 1).cpu().numpy() * 255  # [B, H, W, C]
        pred_faces = pred_faces.astype(np.uint8)

        for i, (pred, orig_size, radius, frame_idx) in enumerate(zip(
            pred_faces, original_sizes, batch_radius, batch_indices
        )):
            try:
                actual_frame_idx = frame_idx + 2
                landmark = video_landmarks[actual_frame_idx]
                radius = int(round(radius))
                
                # # Convert prediction to uint8
                # pred_np = pred.permute(1,2,0).cpu().numpy() * 255
                # pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
                
                # Resize prediction
                resized_pred = cv2.resize(pred, orig_size)
                
                # Calculate integer coordinates
                y_start = int(round(landmark[29,1] - radius))
                y_end = int(round(landmark[29,1] + 2 * radius))
                x_start = int(round(landmark[33,0] - radius - radius//4))
                x_end = int(round(landmark[33,0] + radius + radius//4))
                
                # # Validate coordinates
                # h, w = data_video_frames[actual_frame_idx].shape[:2]
                # y_start = max(0, y_start)
                # y_end = min(h, y_end)
                # x_start = max(0, x_start)
                # x_end = min(w, x_end)
                
                # Ensure prediction matches target region size
                # pred_region = resized_pred[:y_end-y_start, :x_end-x_start]
                pred_region = resized_pred[:radius*3]
                
                # Apply to frame
                data_video_frames[actual_frame_idx][y_start:y_end, x_start:x_end] = pred_region
                
            except Exception as e:
                print(f"Error applying prediction to frame {actual_frame_idx}: {e}")
                continue
    
    # Write all processed frames
    for i in range(5, frame_count):
        video_writer.write(data_video_frames[i-3][:, :, ::-1])
    
    video_writer.release()

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
    print("audio length: ", ds_feature.shape[0])
    print(f'Time taken to extract DeepSpeech features: {end_time - start_time} seconds')
    
    ############################################## align frame with driving audio ##############################################
    start_time = time.time()
    res_video_frame_data_list, res_video_landmark_data = align_frames_with_audio(video_frame_data_list_cycle, video_landmark_data_cycle, ds_feature.shape[0])
    
    res_video_frame_data_list_pad = [video_frame_data_list_cycle[0]] * 2 \
                                    + res_video_frame_data_list \
                                    + [video_frame_data_list_cycle[-1]] * 2
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    assert ds_feature_padding.shape[0] == len(res_video_frame_data_list_pad) == res_video_landmark_data_pad.shape[0]
    end_time = time.time()
    print(f'Time taken to align frames with driving audio: {end_time - start_time} seconds')

    ############################################## randomly select 5 reference images ##############################################
    start_time = time.time()
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    ref_img_tensor = select_reference_images(res_video_frame_data_list_pad, res_video_landmark_data_pad, video_size, resize_w, resize_h)
    end_time = time.time()
    print(f'Time taken to select reference images: {end_time - start_time} seconds')

    ############################################## inference frame by frame ##############################################
    start_time = time.time()
    res_video_path = os.path.join(opt.res_video_dir,os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
    # synthesize_video(res_video_frame_data_list_pad, res_video_landmark_data_pad, ds_feature_padding, ref_img_tensor, res_video_path, resize_w, resize_h)
    batch_synthesize_video(res_video_frame_data_list_pad, res_video_landmark_data_pad, ds_feature_padding, ref_img_tensor, res_video_path, resize_w, resize_h)
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