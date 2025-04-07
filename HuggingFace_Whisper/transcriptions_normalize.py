import re

def normalize_text(text):
    # 去除標點符號
    text = re.sub(r'[，。！？、．,.!?]', '', text)
    text = re.sub(r'[-]', ' ', text)
    # 僅對中文部分添加前後空格 (匹配 Unicode 範圍 \u4e00-\u9fff)
    transcript_spaced = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text)
    transcript_spaced = re.sub(r'\s+', ' ', transcript_spaced).strip()
    return transcript_spaced

def process_file(input_file, output_file):
    results = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # 僅拆分第一個空格，確保只影響文本部分
            parts = line.split(' ', 1)
            if len(parts) == 2:
                filename, transcript = parts
                transcript_spaced = normalize_text(transcript)
                results.append(f"{filename} {transcript_spaced}")
            else:
                results.append(line.strip())
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for res in results:
            outfile.write(res + "\n")
    
    print(f"處理完成！結果已儲存至 {output_file}")

# 設定輸入和輸出文件
input_txt = "transcriptions.txt"
output_txt = "hyp.txt"

process_file(input_txt, output_txt)
