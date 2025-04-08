from fastapi import FastAPI, File, UploadFile
import os
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")

from utils.deep_speech import DeepSpeech

app = FastAPI()

# Tạo instance của class DeepSpeech
DSModel = DeepSpeech('./asserts/output_graph.pb')
ds_features = DSModel.compute_audio_feature('init_audio.wav')

class DeepspeechRequest(BaseModel):
    audio_path: str
    
@app.post("/compute_audio_feature/")
async def compute_audio_feature(request: DeepspeechRequest):
    try:
        # Đọc và tính toán đặc trưng từ file âm thanh cố định
        ds_features = DSModel.compute_audio_feature(request.audio_path)
        
        # Trả về kết quả dưới dạng JSON (có thể bạn cần chuyển ds_features thành list hoặc array)
        return {"ds_features": ds_features.tolist()}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeech API")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()
    # Chạy FastAPI server
    import uvicorn
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
