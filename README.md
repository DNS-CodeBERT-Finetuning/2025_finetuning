# **📌 README: CWE 탐지 모델 (CodeBERT 기반)**

## **📌 프로젝트 개요**
이 프로젝트는 **CodeBERT를 활용하여 C 언어 코드에서 CWE(공통 약점 열거, Common Weakness Enumeration)를 탐지하는 모델**을 개발하는 것을 목표로 합니다.  
다중 라벨 분류(Multi-label Classification) 방식으로 작동하며, **하나의 코드에서 여러 CWE를 동시에 예측할 수 있습니다.**  

---

## **📌 모델 학습 및 평가 과정**
이 프로젝트는 다음과 같은 흐름으로 진행됩니다:

1. **데이터 전처리**
   - `dataset_split.py`: C 코드에서 취약점이 있는 함수(Bad)와 안전한 함수(Good) 추출 및 JSON 저장  
   - `json_tokenizer.py`: CodeBERT 입력을 위해 코드 스니펫을 토큰화  

2. **데이터셋 로드 및 분할**
   - `dataset_loader.py`: 토큰화된 데이터를 `train(80%) / valid(10%) / test(10%)`로 분할 및 저장  

3. **모델 학습 (Fine-tuning)**
   - `train.py`: `microsoft/codebert-base` 또는 `neulab/codebert-c`를 사용하여 CWE 탐지 모델 학습  

4. **모델 평가**
   - `evaluate.py`: 학습된 모델을 사용하여 **테스트 데이터셋에서 성능 평가**  
   - Precision, Recall, F1-score, Hamming Loss 등 다양한 지표를 분석하여 모델 성능 최적화  

5. **모델 테스트 (실제 코드 입력)**
   - 학습된 모델을 활용하여 새로운 C 코드에서 CWE 탐지 수행  

---

## **📌 코드 설명**
### **1️⃣ 데이터 전처리**
#### 🔹 `dataset_split.py`
**기능:**  
- C 코드에서 CWE ID를 추출하고 **취약한 코드(Bad)와 안전한 코드(Good)를 분리하여 JSON 저장**  
- **NULL 포인터 역참조(CWE-476) 등 주요 CWE를 포함한 코드만 필터링 가능**  

📌 **출력:** `cwe_dataset_split.json`  
```json
[
  {
    "code_snippet": "int *ptr = NULL;\n*ptr = 10;",
    "label": 476,
    "type": "Bad"
  }
]
```

---

#### 🔹 `json_tokenizer.py`
**기능:**  
- **CodeBERT 입력을 위해 C 코드 스니펫을 토큰화**  
- 긴 코드는 **슬라이딩 윈도우(Stride 256) 방식으로 나누어 저장**  

📌 **출력:** `cwe_dataset_tokenized.json`  
```json
{
  "input_ids": [0, 1819, 287, 428, ..., 2],
  "attention_mask": [1, 1, 1, ..., 1],
  "label": 6
}
```

---

### **2️⃣ 데이터셋 로드 및 분할**
#### 🔹 `dataset_loader.py`
**기능:**  
- **토큰화된 데이터를 `train(80%) / valid(10%) / test(10%)`으로 분할**  
- **One-hot Encoding 적용하여 모델 학습 가능하게 변환**  
- `train_dataset`, `valid_dataset`, `test_dataset`을 저장  

📌 **출력 예시:**  
```json
{
  "input_ids": [...],
  "label": [0, 0, 0, 0, 0, 0, 1]  // One-hot Encoding 적용
}
```

---

### **3️⃣ 모델 학습**
#### 🔹 `train.py`
**기능:**  
- `microsoft/codebert-base` 또는 `neulab/codebert-c` 모델을 사용하여 **다중 레이블 분류 방식으로 학습**  
- `Trainer`를 사용하여 **Fine-tuning 진행**  

📌 **주요 설정:**  
```python
training_args = TrainingArguments(
    output_dir="./codebert_cwe_multi_label",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01
)
```
📌 **출력:** `./codebert_cwe_multi_label/` 폴더에 학습된 모델 저장  

---

### **4️⃣ 모델 평가**
#### 🔹 `evaluate.py`
**기능:**  
- 학습된 모델을 불러와 **테스트 데이터에서 CWE 탐지 성능 평가**  
- **Precision, Recall, F1-score, Hamming Loss 분석**  

📌 **평가 결과 예시:**  
```plaintext
정확도(Accuracy): 0.8857
정밀도(Precision, Macro): 0.9221
재현율(Recall, Macro): 0.9084
F1-점수(F1-score, Macro): 0.9046
Hamming Loss: 0.024833
```
➡ **Precision과 Recall이 균형을 이루며, 불필요한 CWE 탐지가 줄어듦!**

---

## **📌 성능 개선 전략**
현재 모델의 성능을 더 향상시키기 위해 **다음과 같은 최적화 작업을 수행 가능**  

**1️⃣ 임계값(Threshold) 조정 (`0.5 → 0.4` 실험 가능)**  
**2️⃣ 데이터 증강 (Augmentation) 적용 → 코드 변형, CWE별 샘플 균형 조정**  
**3️⃣ 학습률 튜닝 (`2e-5 → 3e-5`)**  
**4️⃣ `neulab/codebert-c` 모델로 교체 (C 코드 특화 학습)**  
**5️⃣ Class Weight 조정하여 특정 CWE 과소 학습 방지**  
**6️⃣ Attention Score 분석을 통한 취약점 강조**  

---

## **📌 모델 테스트 (새로운 코드 입력)**
**훈련된 모델을 활용하여 새로운 C 코드에서 CWE 탐지 가능**  

📌 **예제 코드 (CWE-476 포함)**  
```c
#include <stdio.h>
#include <stdlib.h>

void unsafe_function() {
    int *ptr = NULL;
    *ptr = 10;  // CWE-476 (NULL Pointer Dereference)
}

int main() {
    unsafe_function();
    return 0;
}
```
📌 **모델 예측 코드 (`predict.py`)**
```python
inputs = tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)  # Softmax 대신 Sigmoid 사용

THRESHOLD = 0.4  # 임계값 설정
predicted_labels = (probs > THRESHOLD).nonzero(as_tuple=True)[1].cpu().numpy()
predicted_cwes = [reverse_label_map[str(idx)] for idx in predicted_labels]
```
📌 **예측 결과 예시**  
```plaintext
🚀 입력된 코드 예측 결과:
📌 예측된 CWE ID: [476]
📊 상위 CWE 확률: [(476, 0.91)]
```
🔥 **즉, 모델이 새 코드를 입력하면 CWE-476을 정확하게 탐지할 수 있음!** 🚀  

---

## **📌 결론**
✔ **CodeBERT를 활용하여 C 코드에서 CWE를 자동 탐지하는 모델을 개발**  
✔ **Fine-tuning을 통해 성능 최적화 진행 (Precision 92.21%, Recall 90.84%)**  
✔ **새로운 C 코드에서도 CWE 예측 가능 → 실제 보안 점검에 활용 가능**  

🚀 **추후 발전 방향:**  
- **다양한 CWE 추가 학습 (현재는 CWE-476 중심)**
- **C 코드의 구조적 패턴 학습을 강화하여 탐지 정확도 향상**
- **모델의 Attention Score 분석을 활용한 중요한 코드 강조**  

🔥 **즉, 이 프로젝트는 CWE 탐지 자동화를 위한 강력한 기반을 제공하며, 실전 보안 분석에 활용될 수 있는 잠재력이 큼!** 🚀  
