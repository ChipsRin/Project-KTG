import numpy as np

def normalize_spacing(text):
    return " ".join(text.split())

def calculate_wer_details(reference, hypothesis):
    reference = reference.lower().strip()
    hypothesis = hypothesis.lower().strip()

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)

    # 建立 DP 
    dp = np.zeros((len_ref + 1, len_hyp + 1), dtype=np.int32)
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j

    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,  # 刪除
                               dp[i][j - 1] + 1,  # 插入
                               dp[i - 1][j - 1] + 1)  # 替換

    wer_value = dp[len_ref][len_hyp] / max(1, len_ref)
    return wer_value

def calculate_wer_from_files(ref_file, hyp_file, output_file):
    # 讀取參考文本
    ref_dict = {}
    with open(ref_file, 'r', encoding='utf-8') as ref_f:
        for line in ref_f:
            line = line.strip()
            parts = line.split(' ', 1)
            if len(parts) == 2:
                ref_id, ref_text = parts
                ref_dict[ref_id] = normalize_spacing(ref_text)

    # 讀取假設文本
    hyp_dict = {}
    with open(hyp_file, 'r', encoding='utf-8') as hyp_f:
        for line in hyp_f:
            line = line.strip()
            parts = line.split(' ', 1)
            if len(parts) == 2:
                hyp_id, hyp_text = parts
                hyp_dict[hyp_id] = normalize_spacing(hyp_text)

    output = []
    error_log = []
    total_wer = 0
    matched_files = 0

    # 根據音檔名稱進行匹配計算 WER
    for ref_id, ref_text in ref_dict.items():
        if ref_id in hyp_dict:
            hyp_text = hyp_dict[ref_id]
            wer = calculate_wer_details(ref_text, hyp_text)

            total_wer += wer
            matched_files += 1

            output.append(f"File: {ref_id}")
            output.append(f"Reference: {ref_text}")
            output.append(f"Hypothesis: {hyp_text}")
            output.append(f"Pseudo_WER: {wer:.4f}")
            output.append("=" * 50)

            print(f"✅ 計算 {ref_id}: WER={wer:.4f}")
        else:
            error_log.append(f"❌ Missing hypothesis for {ref_id}")
            print(f"❌ Missing hypothesis for {ref_id}")

    # 計算平均 WER
    avg_wer = total_wer / matched_files if matched_files > 0 else 0
    output.append(f"平均 WER: {avg_wer:.4f}")
    print(f"平均 WER: {avg_wer:.4f}")

    # 保存結果到文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    # 如果有錯誤，記錄到錯誤日誌
    if error_log:
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(error_log))
        print("⚠️ 部分音檔缺失對應假設文本，已記錄到 error_log.txt")

    print(f"📄 WER 結果已保存至 {output_file}")

# 模型轉錄文本路徑
teacher_model_transcriptions = "teacher.txt"
student_model_transcriptions = "student.txt"
output_file = "Pseudo_MER.txt"

calculate_wer_from_files(teacher_model_transcriptions, student_model_transcriptions, output_file)
