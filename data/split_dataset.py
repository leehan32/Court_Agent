import json
import random

# 원본 데이터 파일 이름
original_file = 'validation.jsonl'
# 새로 만들어질 파일 이름
train_file = 'train.jsonl'
test_file = 'test.jsonl'
# 테스트 데이터로 추출할 샘플 개수
num_test_samples = 100

print(f"'{original_file}' 파일을 읽고 있습니다...")

try:
    with open(original_file, 'r', encoding='utf-8') as f:
        all_cases = [json.loads(line) for line in f]
    
    print(f"총 {len(all_cases)}개의 사건 데이터를 읽었습니다.")
    
    if len(all_cases) < num_test_samples:
        print(f"오류: 데이터 개수({len(all_cases)}개)가 테스트 샘플 개수({num_test_samples}개)보다 적습니다.")
    else:
        # 데이터셋을 무작위로 섞음
        random.shuffle(all_cases)
        
        # 테스트셋과 훈련셋으로 분리
        test_cases = all_cases[:num_test_samples]
        train_cases = all_cases[num_test_samples:]
        
        # 테스트셋 저장
        with open(test_file, 'w', encoding='utf-8') as f:
            for case in test_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        print(f"✅ '{test_file}' 파일에 무작위 사건 {len(test_cases)}개를 저장했습니다.")
        
        # 훈련셋 저장
        with open(train_file, 'w', encoding='utf-8') as f:
            for case in train_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        print(f"✅ '{train_file}' 파일에 나머지 사건 {len(train_cases)}개를 저장했습니다.")
        
except FileNotFoundError:
    print(f"오류: '{original_file}' 파일을 찾을 수 없습니다. 파일이 이 스크립트와 같은 위치에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")