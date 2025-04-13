import os
from faster_whisper import WhisperModel
from tqdm import tqdm

# 模型路徑
model_path = "faster_whisper"
# 選擇 GPU 或 CPU 運行
device = "cuda" 
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(model_path, device=device, compute_type=compute_type)

audio_folder = "test_audio"  # 音檔資料夾路徑
output_file = "transcriptions.txt"  # 轉錄文本

audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

with open(output_file, "w", encoding="utf-8") as out_f:
    for audio_file in tqdm(audio_files, desc="Processing files"):
        audio_path = os.path.join(audio_folder, audio_file)

        segments, _ = model.transcribe(audio_path, beam_size=5, language="zh")

        transcription = " ".join(segment.text for segment in segments)
    
        out_f.write(f"{audio_path} {transcription}\n")

print(f"所有轉錄結果已儲存於 {output_file}")
