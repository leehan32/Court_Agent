from src.graph import app

if __name__ == "__main__":
    print("🚀 모의 법정 시뮬레이션을 시작합니다.")

    # 초기 재판 정보 설정
    initial_state = {
        "case_file": "아파트 층간소음으로 인한 손해배상 청구",
        "plaintiff_lawyer": "원고측 변호사",
        "defendant_lawyer": "피고측 변호사",
    }

    # 그래프 실행 및 결과 스트리밍
    for event in app.stream(initial_state):
        for key, value in event.items():
            print(f"\n--- Node '{key}' 완료 ---")
            # 각 단계의 상세 결과를 보려면 아래 주석을 해제하세요.
            # print(value) 
            print("-" * 25)