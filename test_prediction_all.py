import json
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datasets import load_from_disk

# ✅ 학습된 모델 로드
model_path = "./codebert_cwe_multi_label"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ `test_dataset_new` 로드
test_dataset = load_from_disk("test_dataset")

# ✅ 라벨 역매핑 로드
with open("label_map.json", "r", encoding="utf-8") as f:
    reverse_label_map = json.load(f)  # {0: 78, 1: 134, 2: 190, ...}

# ✅ CWE 임계값 설정
THRESHOLD = 0.3

# ✅ 전체 데이터셋 예측
results = []

for i in range(len(test_dataset)):
    sample_text = test_dataset[i]["code_snippet"]

    # ✅ 모델 입력 변환
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # ✅ 모델 예측 수행
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)  # 다중 레이블이므로 Softmax 대신 Sigmoid 사용

    # ✅ CWE ID 변환 (임계값 0.3 이상인 CWE만 선택)
    predicted_labels = (probs > THRESHOLD).nonzero(as_tuple=True)[1].cpu().numpy()

    # ✅ 예측된 인덱스를 원래 CWE ID로 변환
    if len(predicted_labels) == 0:
        predicted_cwes = ["Safe Code"]  # CWE가 없으면 안전한 코드로 판단
    else:
        predicted_cwes = [reverse_label_map[str(idx)] for idx in predicted_labels]

    # # ✅ 모든 CWE 확률 출력 (상위 3개)
    # prob_values = probs.cpu().numpy()[0]  # 확률 값 가져오기
    # sorted_indices = prob_values.argsort()[::-1]  # 확률이 높은 순서대로 정렬
    # top_cwes = [(reverse_label_map[str(idx)], prob_values[idx]) for idx in sorted_indices[:3]]
    
    # ✅ 모든 CWE 확률 출력 (상위 3개) - float 변환 추가
    prob_values = probs.cpu().numpy()[0]  # 확률 값 가져오기
    sorted_indices = prob_values.argsort()[::-1]  # 확률이 높은 순서대로 정렬
    top_cwes = [(reverse_label_map[str(idx)], float(prob_values[idx])) for idx in sorted_indices[:3]]  # ✅ float 변환 추가


    # ✅ 실제 CWE 라벨 확인
    actual_cwe = test_dataset[i]["label"]

    # ✅ 결과 저장
    results.append({
        "sample_index": i,
        "actual_cwe": actual_cwe,
        "predicted_cwes": predicted_cwes,
        "top_cwes": top_cwes
    })

    # 🔹 진행 상황 출력 (100개마다 출력)
    if (i + 1) % 100 == 0:
        print(f"🔄 {i + 1}/{len(test_dataset)}개 예측 완료...")

# ✅ JSON 파일로 저장
output_path = "test_predictions.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\n✅ 전체 테스트 데이터 예측 완료!")
print(f"📊 결과 저장: {output_path}")
