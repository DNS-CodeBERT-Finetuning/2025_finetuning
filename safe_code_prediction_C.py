import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import json

# ✅ 학습된 모델 로드
model_path = "./codebert_cwe_multi_label_neulab_C"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-c")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 라벨 역매핑 로드
with open("label_map.json", "r", encoding="utf-8") as f:
    reverse_label_map = json.load(f)  # {0: 78, 1: 134, 2: 190, ...}

# ✅ 안전한 C 코드 입력
safe_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void safe_string_copy(char *dest, const char *src, size_t size) {
    if (dest != NULL && src != NULL) {
        strncpy(dest, src, size - 1);
        dest[size - 1] = '\\0';
    }
}

void safe_memory_usage() {
    int *ptr = (int *)malloc(sizeof(int) * 10);
    if (ptr == NULL) {
        printf("메모리 할당 실패\\n");
        return;
    }
    free(ptr);
}

int main() {
    char safe_buffer[50];
    safe_string_copy(safe_buffer, "This is a safe C program!", sizeof(safe_buffer));
    safe_memory_usage();
    printf("✅ 안전한 코드 실행 완료\\n");
    return 0;
}
"""

# ✅ 모델 입력 변환
inputs = tokenizer(safe_code, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

# ✅ 모델 예측 수행
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)  # 다중 레이블이므로 Softmax 대신 Sigmoid 사용

# ✅ CWE ID 변환 (임계값0.3 이상인 CWE만 선택)
THRESHOLD = 0.6
predicted_labels = (probs > THRESHOLD).nonzero(as_tuple=True)[1].cpu().numpy()

# ✅ 예측된 인덱스를 원래 CWE ID로 변환
if len(predicted_labels) == 0:
    predicted_cwes = ["Safe Code"]  # CWE가 없으면 안전한 코드로 판단
else:
    predicted_cwes = [reverse_label_map[str(idx)] for idx in predicted_labels]

# ✅ 모든 CWE 확률 출력 (상위 3개)
prob_values = probs.cpu().numpy()[0]  # 확률 값 가져오기
sorted_indices = prob_values.argsort()[::-1]  # 확률이 높은 순서대로 정렬
top_cwes = [(reverse_label_map[str(idx)], float(prob_values[idx])) for idx in sorted_indices[:3]]

print(f"\n🚀 입력된 안전한 코드 예측 결과:")
print(f"📌 예측된 CWE ID: {predicted_cwes}")
print(f"📊 상위 CWE 확률: {top_cwes}")
