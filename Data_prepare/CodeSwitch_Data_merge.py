import os
import random
import glob
from pydub import AudioSegment
from collections import defaultdict

# 資料夾路徑
languages = {
    "english": "/home/speech/chipsrin/espnet/egs2/formosa_taigi/asr2/merge_train/english",
    "chinese": "/home/speech/chipsrin/espnet/egs2/formosa_taigi/asr2/merge_train/chinese",
    "taiwanese": "/home/speech/chipsrin/espnet/egs2/formosa_taigi/asr2/merge_train/taiwanese"
}

output_audio_folder = "/home/speech/chipsrin/espnet/egs2/formosa_taigi/asr2/merge_train/merge_train_audio/audio"
output_transcript_file = "/home/speech/chipsrin/espnet/egs2/formosa_taigi/asr2/merge_train/merge_train_audio/transcripts.txt"

# 語言標籤
lang_tags = {"english": "[ENG]", "chinese": "[CHN]", "taiwanese": "[TWN]"}


os.makedirs(output_audio_folder, exist_ok=True)

# 加載語料
def load_transcripts(language_path, lang):
    transcript_path = os.path.join(language_path, "transcripts.txt")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcripts = f.readlines()
    
    data = []
    
    # 搜索音檔路徑
    audio_files = {os.path.basename(f).replace('.wav', ''): f for f in glob.glob(os.path.join(language_path, "audio", "**", "*.wav"), recursive=True)}

    for line in transcripts:
        parts = line.split()
        if len(parts) < 2:
            continue  # 確保至少有音檔名稱和轉錄文本

        audio_file_key = parts[0]
        if audio_file_key not in audio_files:
            continue  # 如果找不到對應音檔，跳過
        
        audio_file = audio_files[audio_file_key]
        transcript = f"{lang_tags[lang]} " + " ".join(parts[1:])  # 加上語言標籤
        
        try:
            audio_length = len(AudioSegment.from_file(audio_file))
            data.append((audio_file, transcript, audio_length, lang))
        except Exception as e:
            print(f"音檔加載失敗，跳過：{audio_file}, 錯誤信息：{e}")
            continue
    
    return data

all_data = {lang: load_transcripts(path, lang) for lang, path in languages.items()}

# 配置切換次數和測試集大小
switch_counts = {2: 16000, 3: 8000}
base_max_length = 25000  # 25 秒

# 組合測試集
test_audio_list = []
test_transcripts = []
language_statistics = defaultdict(lambda: defaultdict(int))
used_audio_files = {switch_count: [] for switch_count in switch_counts.keys()}  # 記錄使用的音檔

for switch_count, num_samples in switch_counts.items():
    max_combined_length = base_max_length - (switch_count - 2) * 5000  # 讓 4-switch 更短
    for _ in range(num_samples):
        combined_audio = AudioSegment.empty()
        combined_transcript = []
        current_length = 0
        language_durations = defaultdict(int)
        used_files = []  # 記錄當次合併的音檔

        # 確保語言有音檔
        available_languages = [lang for lang in all_data.keys() if all_data[lang]]
        num_languages = min(switch_count, len(available_languages))  # 避免超過可用語言數

        # 隨機決定語言順序
        selected_languages = random.sample(available_languages, num_languages)

        for lang in selected_languages:
            candidates = [d for d in all_data[lang] if d[2] + current_length <= max_combined_length]

            if not candidates:
                continue  # 如果沒有符合條件的音檔，就跳過這次選擇
            
            audio_file, transcript, audio_length, lang = random.choice(candidates)
            audio_segment = AudioSegment.from_file(audio_file)

            combined_audio += audio_segment
            combined_transcript.append(transcript)
            current_length += audio_length
            language_durations[lang] += audio_length
            used_files.append((audio_file, lang, audio_length))  # 記錄音檔
            
            if current_length >= max_combined_length:
                break
        
        if current_length <= max_combined_length and used_files:
            output_audio_file = os.path.join(output_audio_folder, f"combined_{len(test_audio_list) + 1}.wav")
            combined_audio.export(output_audio_file, format="wav")
            test_audio_list.append(output_audio_file)
            test_transcripts.append(" ".join(combined_transcript))

            # 記錄使用過的音檔
            used_audio_files[switch_count].extend(used_files)

            # 更新語言統計
            for lang, duration in language_durations.items():
                proportion = duration / current_length
                language_statistics[lang]["count"] += 1
                language_statistics[lang]["duration"] += duration
                language_statistics[lang]["proportion"] += proportion

# 儲存測試集的轉錄文本
with open(output_transcript_file, 'w', encoding='utf-8') as f:
    for audio_file, transcript in zip(test_audio_list, test_transcripts):
        f.write(f"{audio_file} {transcript}\n")

# 儲存使用過的音檔名稱
for switch_count, files in used_audio_files.items():
    used_files_path = os.path.join(output_audio_folder, f"used_files_switch_{switch_count}.txt")
    with open(used_files_path, 'w', encoding='utf-8') as f:
        for file, lang, duration in files:
            f.write(f"{file} ({lang_tags[lang]}, {duration}ms)\n")

# 輸出統計結果
print("\n合併比例統計：")
for lang, stats in language_statistics.items():
    avg_proportion = stats["avg_proportion"]
    total_duration = stats["duration"] / 1000  # 轉為秒
    print(f"語言: {lang}")
    print(f"  平均占比: {avg_proportion:.2%}")
    print(f"  總時長: {total_duration:.2f} 秒")
    print(f"  參與合併段數: {stats['count']}")

print(f"混合音檔已儲存至 {output_transcript_file}")
print(f"使用的音檔名稱已儲存至 {output_audio_folder}/used_files_switch_*.txt")
