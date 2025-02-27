import torch
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer, DataCollatorWithPadding
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
import numpy as np

# ✅ 1. 모델 & 토크나이저 로드
print("🚀 학습된 모델 로드 중...")
model_path = "./codebert_cwe_multi_label"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ✅ 패딩 문제 해결 (데이터셋 내 길이 차이를 맞춰줌)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ 모델 로드 완료!")

# ✅ 2. 데이터셋 로드
print("🚀 테스트 데이터셋 로드 중...")
test_dataset = load_from_disk("test_dataset")
print(f"✅ Test 데이터 개수: {len(test_dataset)}")

# ✅ 3. 평가 함수 정의 (세부적인 지표 추가)
def compute_metrics(pred):
    labels = pred.label_ids  # 실제 라벨 (One-hot Encoding)
    preds = torch.sigmoid(torch.tensor(pred.predictions)) > 0.3 # 임계값 0.3 이상을 1(예측)로 변환
    
    # 🔹 정확도 (Accuracy)
    accuracy = accuracy_score(labels, preds)

    # 🔹 Precision, Recall, F1-score (Macro, Micro, Weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    # 🔹 Hamming Loss (전체 예측 오류율)
    hamming = hamming_loss(labels, preds)

    return {
        "정확도(Accuracy)": accuracy,
        "정밀도(Precision, Macro)": precision_macro,
        "재현율(Recall, Macro)": recall_macro,
        "F1-점수(F1-score, Macro)": f1_macro,
        "정밀도(Precision, Weighted)": precision_weighted,
        "재현율(Recall, Weighted)": recall_weighted,
        "F1-점수(F1-score, Weighted)": f1_weighted,
        "Hamming Loss": hamming,
    }

# ✅ 4. 평가 설정
training_args = TrainingArguments(
    output_dir="./codebert_cwe_multi_label",
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # ✅ 패딩 자동 조정 추가
)

# ✅ 5. 테스트 데이터셋 평가
print("🚀 테스트 데이터셋 평가 중...")
test_results = trainer.evaluate()
# ✅ 6. 핵심 지표만 출력
print("\n📊 테스트 데이터셋 평가 결과:")
print(f"✅ 정확도(Accuracy): {test_results['eval_정확도(Accuracy)']:.4f}")
print(f"✅ 정밀도(Precision, Macro): {test_results['eval_정밀도(Precision, Macro)']:.4f}")
print(f"✅ 재현율(Recall, Macro): {test_results['eval_재현율(Recall, Macro)']:.4f}")
print(f"✅ F1-점수(F1-score, Macro): {test_results['eval_F1-점수(F1-score, Macro)']:.4f}")
print(f"✅ 정밀도(Precision, Weighted): {test_results['eval_정밀도(Precision, Weighted)']:.4f}")
print(f"✅ 재현율(Recall, Weighted): {test_results['eval_재현율(Recall, Weighted)']:.4f}")
print(f"✅ F1-점수(F1-score, Weighted): {test_results['eval_F1-점수(F1-score, Weighted)']:.4f}")
print(f"✅ Hamming Loss: {test_results['eval_Hamming Loss']:.6f}")
