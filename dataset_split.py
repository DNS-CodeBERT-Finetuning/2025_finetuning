import os
import json
import re


# 현재 실행 중인 Python 파일의 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# C 파일들이 있는 폴더 경로 (Windows 환경)
DATASET_PATH = r"SourceFile_C"

# ✅ 추출할 CWE ID 목록 (78, 134, 190, 400, 416, 476만 포함)
TARGET_CWE_IDS = {"78", "134", "190", "400", "416", "476"}

# 데이터 저장 리스트
dataset = []
previous_cwe = None  # 이전 CWE ID 저장 변수

# 폴더 내 모든 C 파일 처리
for file in os.listdir(DATASET_PATH):
    if file.endswith(".c"):  # C 파일만 처리
        file_path = os.path.join(DATASET_PATH, file)

        # CWE ID 추출 (예: CWE191_Integer_Underflow__char_min_multiply_12.c → 191)
        cwe_match = re.search(r'CWE(\d+)', file)
        if not cwe_match:
            continue  # CWE ID가 없는 경우 스킵
        cwe_id = cwe_match.group(1)
        
        # ✅ 특정 CWE만 추출 (78, 134, 190, 400, 416, 476)
        if cwe_id not in TARGET_CWE_IDS:
            continue  # 해당 CWE가 아니면 스킵

        # CWE가 변경될 때만 출력
        if cwe_id != previous_cwe:
            print(f"🔄 CWE {cwe_id} 코드 스니펫 진행 중...")
            previous_cwe = cwe_id  # 현재 CWE ID를 저장

        # 코드 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        # # ✅ Bad 코드 추출
        # bad_match = re.search(r'void (CWE\d+_.+?_bad)\(\)', code)
        # if bad_match:
        #     bad_func_name = bad_match.group(1)
        #     bad_code = re.search(r'void ' + bad_func_name + r'\(\)\s*\{(.*?)\}', code, re.DOTALL)
        #     if bad_code:
        #         dataset.append({
        #             "code_snippet": bad_code.group(1).strip(),
        #             "label": int(cwe_id),
        #             "type": "Bad"
        #         })

        # # ✅ Good 코드 추출 (goodG2B, goodB2G, good1, good2 포함, but good() 제외)
        # good_matches = re.findall(r'void (good(?:G2B|B2G|[1-9]\d*))\(\)', code)
        # for good_func_name in good_matches:
        #     good_code = re.search(r'void ' + good_func_name + r'\(\)\s*\{(.*?)\}', code, re.DOTALL)
        #     if good_code:
        #         dataset.append({
        #             "code_snippet": good_code.group(1).strip(),
        #             "label": int(cwe_id),
        #             "type": "Good"
        #         })
        
        
        # ✅ Bad 코드 추출 (전체 함수 블록 포함)
        bad_matches = re.finditer(r'void (CWE\d+_.+?_bad)\s*\(\)\s*\{((?:.|\n)*?)\}', code)
        for match in bad_matches:
            dataset.append({
                "code_snippet": match.group(2).strip(),
                "label": int(cwe_id),
                "type": "Bad"
            })

        # ✅ Good 코드 추출 (goodG2B, goodB2G, good1, good2 포함, but good() 제외)
        good_matches = re.finditer(r'void (good(?:G2B|B2G|[1-9]\d*))\s*\(\)\s*\{((?:.|\n)*?)\}', code)
        for match in good_matches:
            dataset.append({
                "code_snippet": match.group(2).strip(),
                # "label": int(cwe_id),
                "label": "Safe Code",  # ✅ 기존 CWE ID 대신 "Safe Code" 사용
                "type": "Good"
            })

# ✅ 중복 제거
unique_dataset = []
seen_snippets = set()

for entry in dataset:
    snippet = entry["code_snippet"]
    if snippet not in seen_snippets:
        seen_snippets.add(snippet)
        unique_dataset.append(entry)


        

# JSON 파일 저장 경로 (현재 실행 중인 Python 파일과 동일한 폴더)
# json_path = os.path.join(SCRIPT_DIR, "cwe_dataset_split.json")
json_path = os.path.join(SCRIPT_DIR, "cwe_dataset_T_split.json")

# JSON 파일로 저장
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print(f"✅ 데이터셋 저장 완료! 총 {len(dataset)}개의 코드 스니펫을 저장했습니다.")
print(f"📁 JSON 파일 경로: {json_path}")
