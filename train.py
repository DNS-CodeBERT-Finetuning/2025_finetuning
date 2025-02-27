import torch
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer, DataCollatorWithPadding
from datasets import load_from_disk
import pandas as pd

# ✅ 1. CodeBERT 모델 로드 (다중 레이블 분류 설정 추가)
print("🚀 모델 로드 중...")
# train_dataset = load_from_disk("train_dataset")  # ✅ Train 데이터 로드
train_dataset = load_from_disk("train_T_dataset")  # ✅ Train 데이터 로드
num_labels = len(train_dataset[0]["label"])  # ✅ 라벨 개수 가져오기

model = RobertaForSequenceClassification.from_pretrained(
    # "microsoft/codebert-base",
    "neulab/codebert-c",
    num_labels=num_labels,
    problem_type="multi_label_classification"  # 🔹 다중 레이블 분류 설정 추가
)

print(f"✅ 모델 로드 완료! 총 {num_labels}개의 CWE를 동시에 예측할 수 있도록 설정됨.")

# ✅ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 현재 학습 디바이스: {device}")
model.to(device)  # 모델을 GPU로 이동

# ✅ 2. 토크나이저 로드
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-c")

# ✅ 3. 데이터셋 로드 (Train / Validation)
print("🚀 데이터셋 로드 중...")
# valid_dataset = load_from_disk("valid_dataset")
valid_dataset = load_from_disk("valid_T_dataset")

print(f"✅ Train: {len(train_dataset)}개, Valid: {len(valid_dataset)}개")

# ✅ 4. 패딩 문제 해결 (Data Collator 추가)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ✅ 5. 학습 설정
training_args = TrainingArguments(
    # output_dir="./codebert_cwe_multi_label",
    output_dir="./codebert_cwe_multi_label_neulab_C",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,  # 🚀 GPU VRAM 부족하면 줄이기
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# ✅ 6. Trainer 설정 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

print("🚀 다중 레이블 학습 시작!")
trainer.train()
print("✅ 학습 완료!")

# ✅ 7. 모델 저장
# model.save_pretrained("./codebert_cwe_multi_label")
model.save_pretrained("./codebert_cwe_multi_label_neulab_C")
print("✅ 다중 레이블 모델 저장 완료!")
