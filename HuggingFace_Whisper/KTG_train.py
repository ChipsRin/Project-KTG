import os
import json
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer

# 載入資料集
dataset = load_dataset("json", data_files={"train": "train_format.jsonl"}, split="train")

# 載入 Whisper Large 的處理器與模型
model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name, language="Chinese", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

def prepare_example(example):
    # 讀取音檔
    speech_array, sr = torchaudio.load(example["audio_filepath"])
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech_array = resampler(speech_array)
    
    speech_array = speech_array.squeeze().numpy()
    
    input_features = processor.feature_extractor(speech_array, sampling_rate=16000).input_features[0]
    example["input_features"] = input_features

    labels = processor.tokenizer(example["text"], max_length=448, truncation=True).input_ids
    example["labels"] = labels
    return example

dataset = dataset.map(prepare_example)

def data_collator(batch):
    # 動態 padding
    input_features = [torch.tensor(item["input_features"], dtype=torch.float) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    input_features = pad_sequence(input_features, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_features": input_features, "labels": labels}

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=6,         
    gradient_accumulation_steps=2,         
    num_train_epochs=3,                     
    learning_rate=1e-5,                      
    warmup_steps=1000,                       
    logging_steps=10,
    save_steps=500,
    fp16=True,                             
    evaluation_strategy="no",              
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

if __name__ == "__main__":
    # 若使用多 GPU 訓練，可使用 torchrun 或其他指令啟動，例如：
    # torchrun --nproc_per_node=8 KTG_train.py
    trainer.train()
