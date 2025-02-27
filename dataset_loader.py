import json
import numpy as np
import pandas as pd
from datasets import Dataset

# ✅ JSON 파일 로드
# with open("cwe_dataset_tokenized.json", "r", encoding="utf-8") as f:
with open("cwe_dataset_T_tokenized.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ 1. 고유한 CWE 라벨 리스트 생성 (One-hot Encoding 적용)
unique_labels = sorted(set(entry["label"] for entry in data))  # 고유 CWE ID 정렬
label_map = {label: i for i, label in enumerate(unique_labels)}  # 라벨을 0부터 시작하는 인덱스로 매핑
num_labels = len(unique_labels)  # 총 CWE 개수 확인

# ✅ 2. 기존 정수 라벨을 One-hot Encoding으로 변환
for entry in data:
    one_hot = np.zeros(num_labels)  # 벡터 생성 (모든 값을 0으로 초기화)
    one_hot[label_map[entry["label"]]] = 1  # 해당 CWE ID 위치에 1 저장
    entry["label"] = one_hot.tolist()  # 리스트 형태로 저장

# ✅ 3. 데이터셋을 Pandas DataFrame으로 변환
df = pd.DataFrame(data)

# ✅ 4. 데이터셋을 Train(80%) / Valid(10%) / Test(10%)로 분할
train_test = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 데이터 셔플
train_size = int(0.8 * len(train_test))
valid_size = int(0.1 * len(train_test))

train_df = train_test[:train_size]
valid_df = train_test[train_size:train_size+valid_size]
test_df = train_test[train_size+valid_size:]  # ✅ Test 데이터에는 code_snippet 유지

# ✅ 5. `test_dataset`에만 `code_snippet` 포함
train_dataset = Dataset.from_pandas(train_df.drop(columns=["code_snippet"], errors="ignore"), preserve_index=False)  # ✅ 코드 제거
valid_dataset = Dataset.from_pandas(valid_df.drop(columns=["code_snippet"], errors="ignore"), preserve_index=False)  # ✅ 코드 제거
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)  # ✅ 코드 스니펫 포함

# ✅ 6. 데이터셋 저장
# train_dataset.save_to_disk("train_dataset")
# valid_dataset.save_to_disk("valid_dataset")
# test_dataset.save_to_disk("test_dataset")
train_dataset.save_to_disk("train_T_dataset")
valid_dataset.save_to_disk("valid_T_dataset")
test_dataset.save_to_disk("test_T_dataset")

print(f"✅ One-hot Encoding & 데이터셋 분할 완료!")
print(f"🔹 Train 데이터: {len(train_dataset)}개")
print(f"🔹 Validation 데이터: {len(valid_dataset)}개")
print(f"🔹 Test 데이터: {len(test_dataset)}개 (✅ 코드 스니펫 포함)")
