import numpy as np
import re

# 定義聲母（initials）與韻母（finals）
consonants = ['p', 'ph', 'm', 'b', 't', 'th', 'n', 'l', 'k', 'kh', 'ng', 'g', 'ts', 'tsh', 's', 'j', 'h']
vowels = sorted([
    'm', 'mh', 'ng', 'ngh', 
    'a', 'ah', 'ann', 'annh', 'am', 'ap', 'an', 'at', 'ang', 'ak',
    'e', 'eh', 'enn', 'ennh', 'eng',
    'ee', 'eeh',
    'er', 'erh', 'erm',
    'i', 'ih', 'inn', 'innh', 'im', 'ip', 'in', 'it', 'ing', 'ik',
    'ir', 'irm', 'irp', 'irn', 'irt', 'irng', 'irk', 
    'o', 'oh', 'or', 'orh', 
    'oo', 'ooh', 'onn', 'onnh', 'om', 'op', 'ong', 'ok',
    'u', 'uh', 'un', 'ut', 
    'ai', 'aih', 'ainn', 'ainnh',
    'au', 'auh', 'aunn', 'aunnh',
    'ia', 'iah', 'iann', 'iannh', 'iam', 'iap', 'ian', 'iat', 'iang', 'iak', 
    'io', 'ioh', 'ior', 'iorh', 'ioo', 'ionn', 'iong', 'iok', 
    'iu', 'iuh', 'iunn', 'iunnh', 
    'iau', 'iauh', 'iaunn',
    'ua', 'uah', 'uann', 'uan', 'uat', 'uang', 
    'ue', 'ueh',
    'ere', 'ereh',
    'uee', 
    'ui', 'uih', 'uinn',
    'irinn',
    'uai', 'uaih', 'uainn', 'uainnh'
], key=len, reverse=True)  # 確保長韻母優先匹配

def preprocess_text(text):
    """ 去除聲調數字和 '-'，並執行 initial-final 拆分 """
    text = text.lower()  # 英文正規化為小寫
    text = re.sub(r'[-\d]', ' ', text)  # 去掉調號
    words = text.split()
    split_words = []

    for word in words:
        initial, final = split_pinyin(word)
        if initial and final:
            split_words.append(initial + " " + final)
        elif final:
            split_words.append("- " + final)
    
    return " ".join(split_words)

def split_pinyin(word):
    """ 拆分聲母（initial）與韻母（final） """
    for vowel in vowels:
        if word.endswith(vowel):
            initial = word[:-len(vowel)]
            if initial in consonants:
                return initial, vowel
            else:
                return ' ', word  # 沒有聲母時當作韻母
    return None, None

def normalize_spacing(text):
    """ 確保文本空格一致 """
    return " ".join(text.split())

def calculate_wer_details(reference, hypothesis):
    """
    計算 WER（Word Error Rate），並標記錯誤類型（插入 I、刪除 D、替換 S）
    """
    reference = reference.lower().strip()
    hypothesis = hypothesis.lower().strip()

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)

    # 建立 DP 表
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
    with open(ref_file, 'r', encoding='utf-8') as ref_f, open(hyp_file, 'r', encoding='utf-8') as hyp_f:
        ref_lines = ref_f.readlines()
        hyp_lines = hyp_f.readlines()

    output = []
    error_log = []

    for idx, (ref_line, hyp_line) in enumerate(zip(ref_lines, hyp_lines), start=1):
        try:
            ref_line = ref_line.strip()
            hyp_line = hyp_line.strip()
            if " " not in ref_line or " " not in hyp_line:
                raise ValueError(f"Line {idx} is malformed: '{ref_line}' | '{hyp_line}'")

            ref_id, ref_text = ref_line.split(' ', 1)
            hyp_id, hyp_text = hyp_line.split(' ', 1)

            ref_text = normalize_spacing(preprocess_text(ref_text))
            hyp_text = normalize_spacing(preprocess_text(hyp_text))

            wer = calculate_wer_details(ref_text, hyp_text)

            output.append(f"File: {ref_id}")
            output.append(f"Reference: {ref_text}")
            output.append(f"Hypothesis: {hyp_text}")
            output.append(f"PER: {wer:.4f}")
            output.append("=" * 50)

            print(f"✅ 計算 {ref_id}: WER={wer:.4f}")

        except ValueError as e:
            error_log.append(f"❌ Line {idx} Error: {e}")
            print(f"❌ Line {idx} Error: {e}")
            continue  # 跳過這一行

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    if error_log:
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(error_log))
        print("⚠️ 部分行有問題，已記錄到 error_log.txt")

    print(f"📄 WER 結果已保存至 {output_file}")

# 使用方法
teacher_model_transcriptions = "teacher.txt"
student_model_transcriptions = "student.txt"
output_file = "Pseudo_PER.txt"

calculate_wer_from_files(teacher_model_transcriptions, student_model_transcriptions, output_file)
