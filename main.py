import argparse
import asyncio
import json
import logging
import os
import platform
from collections import defaultdict
import time
import requests

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from fastapi.websockets import WebSocketState

from dotenv import load_dotenv

load_dotenv()
print("using devices: ", os.getenv("CUDA_VISIBLE_DEVICES"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from tts import text_to_speech
from facial_dubbing_improve import facial_dubbing

# -------------------------------------------------------------------
# Config logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WebRTC-Server")

# -------------------------------------------------------------------
# Shutdown: Đóng tất cả kết nối WebRTC khi server tắt
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Đóng tất cả kết nối WebRTC khi server tắt
    logger.info("Closing all WebRTC connections...")
    coros = [cleanup(client_id) for client_id in connections]
    await asyncio.gather(*coros)
    logger.info("All WebRTC connections closed")
    logger.info("Server shutdown complete")

# -------------------------------------------------------------------
# Khởi tạo ứng dụng FastAPI
# -------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# -------------------------------------------------------------------
# Phục vụ các file tĩnh (CSS, JS)
# -------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------------------------
# Quản lý nhiều kết nối WebSocket
# -------------------------------------------------------------------
connections = defaultdict(dict)  # {client_id: {"websocket": ws, "peer_connection": pc}}
relay = None
webcam = None

# -------------------------------------------------------------------
# Hàm đọc file HTML client
# -------------------------------------------------------------------
@app.get("/")
async def get():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# -------------------------------------------------------------------
# Hàm tạo stream từ file hoặc webcam
# -------------------------------------------------------------------
def create_local_tracks(play_from, decode):
    """Tạo stream từ file hoặc webcam."""
    global relay, webcam

    if play_from:
        logger.info(f"Playing media from file: {play_from}")
        player = MediaPlayer(play_from, decode=decode)
        return player.audio, player.video

    if relay is None:
        options = {"framerate": "30", "video_size": "640x480"}
        platform_name = platform.system()
        logger.info(f"Operating system: {platform_name}")

        if platform_name == "Darwin":
            webcam = MediaPlayer("default:none", format="avfoundation", options=options)
        elif platform_name == "Windows":
            webcam = MediaPlayer("video=Integrated Camera", format="dshow", options=options)
        elif platform_name == "Linux":
            webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)

        relay = MediaRelay()
        logger.info("Webcam is ready.")

    return None, relay.subscribe(webcam.video)

# -------------------------------------------------------------------
# Hàm ép codec cho stream media
# -------------------------------------------------------------------
def force_codec(pc, sender, forced_codec):
    """Ép codec cho stream media."""
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences([codec for codec in codecs if codec.mimeType == forced_codec])
    logger.info(f"Force codec {forced_codec} for {kind}.")

# -------------------------------------------------------------------
# Hàm xử lý khi client rời đi
# -------------------------------------------------------------------
async def cleanup(client_id):
    """Đóng kết nối WebRTC và WebSocket khi client rời đi."""
    if client_id in connections:
        pc = connections[client_id]["peer_connection"]
        await pc.close()
        del connections[client_id]
        logger.info(f"Closed connection for client {client_id}")

# -------------------------------------------------------------------
# Cấu hình STUN/TURN server
# -------------------------------------------------------------------
iceServers = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
    RTCIceServer(urls="stun:stun2.l.google.com:19302"),
    # Thêm TURN server nếu cần (ví dụ: "turn:your-turn-server.com")
]
RTC_CONFIGURATION = RTCConfiguration(iceServers)

# -------------------------------------------------------------------
# Hàm phân tích ICE candidate
# -------------------------------------------------------------------
# def parse_ice_candidate(candidate):
#     ip = candidate['candidate'].split(' ')[4]
#     port = candidate['candidate'].split(' ')[5]
#     protocol = candidate['candidate'].split(' ')[2]
#     priority = candidate['candidate'].split(' ')[3]
#     foundation = candidate['candidate'].split(' ')[0].split(':')[1]
#     component = "rtp" if candidate['candidate'].split(' ')[1] == "1" else "rtcp"
#     type = candidate['candidate'].split(' ')[7]
#     candidate = RTCIceCandidate(ip=ip,
#                                 port=port,
#                                 protocol=protocol,
#                                 priority=priority,
#                                 foundation=foundation,
#                                 component=component,
#                                 type=type,
#                                 sdpMid=candidate['sdpMid'], 
#                                 sdpMLineIndex=candidate['sdpMLineIndex'])
#     return candidate

# -------------------------------------------------------------------
# Hàm chuyển đổi văn bản thành giọng nói
# -------------------------------------------------------------------
url = "http://127.0.0.1:7000/tts"
def text_to_speech(text):
    start_time = time.time()
    data = {"text": text}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        output_audio_paths = response.json().get("audio_paths")
        if not output_audio_paths:
            logger.error("No audio path returned from TTS server")
            return None
        logger.info(f"Audio saved to {output_audio_paths[0]}")
    else:
        logger.error(f"Error: {response.status_code} - {response.text}")
    end_time = time.time()
    print(f"Time taken to generate audio: {end_time - start_time} seconds")
    return output_audio_paths[0]

# -------------------------------------------------------------------
# Hàm xử lý văn bản và tạo video/audio
# -------------------------------------------------------------------
async def process(text):
    """Xử lý văn bản và tạo video/audio."""
    output_audio_path = text_to_speech(text)
    output_video_path = facial_dubbing(output_audio_path)
    return output_audio_path, output_video_path

# -------------------------------------------------------------------
# WebSocket signaling server
# -------------------------------------------------------------------
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info("WebSocket connection established")

    pc = RTCPeerConnection()
    # pc = RTCPeerConnection(configuration=RTC_CONFIGURATION)
    connections[client_id] = {"websocket": websocket, "peer_connection": pc}

    # ice_gatherer = RTCIceGatherer()
    # ice_gatherer = RTCIceGatherer(iceServers=iceServers)
    # await ice_gatherer.gather()
    # my_ice_candidates = ice_gatherer.getLocalCandidates()
    # print("ice candidate gathered:")
    # for candidate in my_ice_candidates:
    #     print(candidate)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Client {client_id} WebRTC state: {pc.connectionState}")
        if pc.connectionState == 'connected':
            logger.info('Peers successfully connected')
        elif pc.connectionState == "failed":
            await cleanup(client_id)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state changed: {pc.iceConnectionState}")
        if pc.iceConnectionState == "connected":
            logger.info("ICE connection successful")
        elif pc.iceConnectionState == "failed":
            logger.info("ICE connection failed")
    
    @pc.on("signalingstatechange")
    async def on_signalingstatechange():
        logger.info(f"Signaling state changed: {pc.signalingState}")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state changed: {pc.iceGatheringState}")

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            logger.info("Video track added")
        elif track.kind == "audio":
            logger.info("Audio track added")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            # logger.info(f"Received WebSocket message: {message}")
            
            if message["type"] == "text":
                logger.info("Received text message from client")
                text = message["text"]
                logger.info(f"Text: {text}")
                task = asyncio.create_task(process(text))
                # output_audio_path, output_video_path =  process(text)
                # logger.info(f"Output audio path: {output_audio_path}")

            elif message["type"] == "offer":
                logger.info("Received offer from client")
                offer = RTCSessionDescription(
                    type=message["type"],
                    sdp=message["sdp"]
                )
                await pc.setRemoteDescription(offer)
                logger.info("Remote description set")
                # text = message["text"]
                # logger.info(f"Text: {text}")
                # output_audio_path = text_to_speech(text)
                # logger.info(f"Output audio path: {output_audio_path}")
                # output_video_path = facial_dubbing()
                # logger.info(f"Output video path: {output_video_path}")
                # output_video_path = "video.mp4"
                output_video_path = "asserts/inference_result/test4_facial_dubbing_add_audio.mp4"
                # Đợi process(text) hoàn thành nếu có task
                if "task" in locals():
                    logger.info("Waiting for task to complete...")
                    output_audio_path, output_video_path = await task
                    logger.info(f"Output audio path: {output_audio_path}")
                    logger.info(f"Output video path: {output_video_path}")

                # Mở nguồn phát video/audio
                audio, video = create_local_tracks(output_video_path, decode=not args.play_without_decoding)

                if audio:
                    audio_sender = pc.addTrack(audio)
                    if args.audio_codec:
                        force_codec(pc, audio_sender, args.audio_codec)

                if video:
                    video_sender = pc.addTrack(video)
                    if args.video_codec:
                        force_codec(pc, video_sender, args.video_codec)

                # Tạo WebRTC answer
                logger.info("Creating answer...")
                answer = await pc.createAnswer()
                logger.info("Setting local description...")
                await pc.setLocalDescription(answer)
                logger.info("Sent SDP answer to client")

                # Gửi answer qua WebSocket
                await websocket.send_text(json.dumps({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                }))
            elif message["type"] == "answer":
                logger.info("Received answer from client")
                answer = RTCSessionDescription(
                    type=message["type"],
                    sdp=message["sdp"]
                )
                await pc.setRemoteDescription(answer)
                logger.info("Remote description set")
            elif message["type"] == "candidate":
                logger.info("Received candidate from client")
                # candidate = parse_ice_candidate(message['candidate'])
                # await pc.addIceCandidate(candidate)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Gửi thông báo lỗi cho client
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        if pc.connectionState != "closed":
            await pc.close()
        if client_id in connections:
            del connections[client_id]
        logger.info("WebRTC connection closed")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC Server with FastAPI & WebSocket")
    parser.add_argument("--play-from", default="video.mp4", help="Read the media from a file and send it.")
    parser.add_argument("--play-without-decoding", action="store_true", help="Read the media without decoding.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=19000, help="Server port (default: 9000)")
    parser.add_argument("--audio-codec", help="Force a specific audio codec (e.g., audio/opus)")
    parser.add_argument("--video-codec", help="Force a specific video codec (e.g., video/H264)")

    args = parser.parse_args()

    import uvicorn
    logger.info(f"Starting WebRTC server on ws://{args.host}:{args.port}/ws")
    uvicorn.run(app, host=args.host, port=args.port)

'''昨日、私は公園へ行きました。天気がよくて、空は青く、風が気持ちよかったです。公園にはたくさんの人がいました。子どもたちは楽しそうに遊んでいて、大人たちはベンチに座って話していました。私は木の下で本を読みながら、リラックスしました。そのあと、カフェに行ってコーヒーを飲みました。とても楽しい一日でした。'''