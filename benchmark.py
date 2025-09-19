import json
import argparse
import time
import csv
import os
from datetime import datetime
from src.graph import app
from src.agents import redis_client
from src.vector_db import vector_store
import src.console as console
from rich.rule import Rule
from rich.table import Table
from collections import defaultdict

def run_benchmark(test_filepath: str, is_trained: bool):
    """
    주어진 테스트 데이터셋으로 벤치마크를 수행하고, 실제 점수를 집계하여 CSV 파일로 저장합니다.
    """
    mode = "학습 후 (Trained)" if is_trained else "학습 전 (Untrained)"
    console.print_header(f"벤치마크 테스트 시작: {mode}")

    if not is_trained:
        console.console.print("[bold yellow]경고: 모든 DB(Redis, PostgreSQL)의 데이터를 초기화합니다.[/bold yellow]")
        redis_client.flushall()
        console.console.print("🔴 Redis DB가 초기화되었습니다.")
        try:
            vector_store.delete_collection()
            console.console.print("🔴 PostgreSQL 벡터 DB가 초기화되었습니다.")
        except Exception as e:
            console.console.print(f"🟡 PostgreSQL 벡터 DB 초기화 중 참고: {e}")
    else:
        console.console.print("사전 학습된 DB를 사용하여 성능을 측정합니다.")

    try:
        with open(test_filepath, 'r', encoding='utf-8') as f:
            test_cases = [json.loads(line) for line in f]
    except FileNotFoundError:
        console.console.print(f"[bold red]오류: 테스트 파일을 찾을 수 없습니다 - {test_filepath}[/bold red]")
        return

    # 결과를 저장할 CSV 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"benchmark_results_{mode.replace(' ', '_')}_{timestamp}.csv"
    
    with open(results_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = ['case_id', 'logical_consistency_score', 'legal_validity_score', 'social_consideration_score']
        writer.writerow(header)

        total_scores = defaultdict(float)
        total_runs = 0

        for i, case in enumerate(test_cases):
            case_id = case.get("caseId", "N/A")
            console.console.print(Rule(f"[bold]테스트 케이스 {i+1}/{len(test_cases)} 실행 (ID: {case_id})[/bold]"))
            
            initial_state = {
                "case_file": f"원고 주장: {case['plaintiff_statement']}\n피고 주장: {case['defendant_statement']}",
                "plaintiff_lawyer": "원고측 변호사",
                "defendant_lawyer": "피고측 변호사",
            }

            final_event = None
            for event in app.stream(initial_state):
                if "__end__" in event:
                    final_event = event["__end__"]

            final_state = final_event or {}
            critique_scores = final_state.get('critique_scores', [])

            row_data = {'case_id': case_id}
            for item in critique_scores:
                criteria = item.get('criteria')
                score = item.get('score', 0)
                
                if "논리적" in criteria:
                    total_scores['논리적 일관성'] += score
                    row_data['logical_consistency_score'] = score
                elif "법률적" in criteria:
                    total_scores['법률적 타당성'] += score
                    row_data['legal_validity_score'] = score
                elif "사회적" in criteria:
                    total_scores['사회적 가치 고려'] += score
                    row_data['social_consideration_score'] = score
            
            writer.writerow([row_data.get(h, 'N/A') for h in header])
            total_runs += 1
            
            time.sleep(1)

    console.print_header(f"벤치마크 테스트 완료: {mode}")
    console.console.print(f"결과가 [bold cyan]{results_filename}[/bold cyan] 파일에 저장되었습니다.")

    table = Table(title="평균 점수 (Pass 비율)")
    table.add_column("평가 항목", justify="right", style="cyan", no_wrap=True)
    table.add_column("평균 점수 (%)", justify="center", style="magenta")

    if total_runs > 0:
        for criteria, total_score in total_scores.items():
            avg_score = (total_score / total_runs) * 100
            table.add_row(criteria, f"{avg_score:.2f}%")
    
    console.console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모의 법정 시스템 벤치마크 테스트")
    parser.add_argument('--mode', type=str, required=True, choices=['trained', 'untrained'],
                        help="'trained' 또는 'untrained' 모드를 선택하세요.")
    args = parser.parse_args()

    # 현재 스크립트 파일의 위치를 기준으로 데이터 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dataset_path = os.path.join(current_dir, "data", "test.jsonl")
    
    if args.mode == 'untrained':
        run_benchmark(test_dataset_path, is_trained=False)
    elif args.mode == 'trained':
        run_benchmark(test_dataset_path, is_trained=True)