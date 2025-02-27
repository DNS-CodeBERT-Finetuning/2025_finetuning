import json
from transformers import RobertaTokenizer

# ✅ CodeBERT 토크나이저 로드
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ✅ JSON 파일 불러오기
# json_path = "cwe_dataset_split.json"
json_path = "cwe_dataset_T_split.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ 토큰화 및 변환된 데이터 저장 리스트
tokenized_data = []
MAX_LENGTH = 512  # CodeBERT의 최대 토큰 길이
STRIDE = 256  # 슬라이딩 윈도우 (겹치는 토큰 수)

# ✅ JSON 데이터 토큰화 진행
for i, entry in enumerate(data):
    code_snippet = entry["code_snippet"]
    
    # 🔹 CodeBERT 토큰화 진행 (긴 코드 처리 포함)
    tokenized = tokenizer(code_snippet, padding=False, truncation=False)

    input_ids = tokenized["input_ids"]
    
    # 🔹 코드가 512개 이하이면 그대로 저장
    if len(input_ids) <= MAX_LENGTH:
        tokenized_data.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "label": entry["label"],
            "type": entry["type"],
            "code_snippet": code_snippet  # ✅ 원본 코드 유지
        })
    else:
        # 🔹 512개 초과 시 슬라이딩 윈도우 방식으로 나누기
        for start in range(0, len(input_ids), STRIDE):
            end = start + MAX_LENGTH
            chunk = input_ids[start:end]

            # 마지막 블록이 너무 작으면 버리기 (예: 100개 미만)
            if len(chunk) < 100:
                continue

            tokenized_data.append({
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
                "label": entry["label"],
                "type": entry["type"],
                "code_snippet": code_snippet  # ✅ 원본 코드 유지
            })

    # 🔄 진행 상황 출력 (1000개마다 한 번씩 출력)
    if (i + 1) % 1000 == 0:
        print(f"🔄 {i + 1}개 토큰화 완료...")

# ✅ 라벨 인덱싱을 위해 고유한 CWE 라벨 수집
unique_labels = sorted(set(str(entry["label"]) for entry in tokenized_data))  # 모든 값을 문자열로 변환 후 정렬
label_map = {label: i for i, label in enumerate(unique_labels)}  # 0부터 시작하는 인덱스 매핑
reverse_label_map = {v: k for k, v in label_map.items()}  # ✅ 역매핑 생성

# ✅ 기존 라벨을 인덱스로 변환
for entry in tokenized_data:
    entry["label"] = label_map[str(entry["label"])]  # ✅ 정수 → 문자열 변환 후 매핑

# ✅ 토큰화된 데이터 JSON 파일로 저장
# tokenized_json_path = "cwe_dataset_tokenized.json"
tokenized_json_path = "cwe_dataset_T_tokenized.json"

with open(tokenized_json_path, "w", encoding="utf-8") as f:
    json.dump(tokenized_data, f, indent=4)

# ✅ 라벨 매핑 JSON 저장
# with open("label_map.json", "w", encoding="utf-8") as f:
with open("label_T_map.json", "w", encoding="utf-8") as f:
    json.dump(reverse_label_map, f, indent=4)

print(f"✅ CodeBERT 토큰화 완료! 총 {len(tokenized_data)}개의 데이터가 저장됨.")
print(f"📁 JSON 파일 경로: {tokenized_json_path}")
print(f"✅ 라벨 매핑 저장 완료! 파일: label_T_map.json")
