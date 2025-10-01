import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

from rich.rule import Rule
from rich.table import Table

import src.console as console
from src.agents import CRITIQUE_CRITERIA, redis_client
from src.graph import app
from src.vector_db import ensure_collection, vector_store

CRITERIA_HEADERS: Dict[str, Tuple[str, str]] = {
    "논리적 일관성": ("logical_consistency_score", "logical_consistency_reason"),
    "법률적 타당성": ("legal_validity_score", "legal_validity_reason"),
    "사회적 가치 고려": ("social_consideration_score", "social_consideration_reason"),
}


def run_benchmark(test_filepath: str, is_trained: bool):
    """주어진 테스트 데이터셋으로 벤치마크를 수행하고, 결과를 CSV로 저장합니다."""
    mode = "학습 후 (Trained)" if is_trained else "학습 전 (Untrained)"
    console.print_header(f"벤치마크 테스트 시작: {mode}")

    if not is_trained:
        console.console.print("[bold yellow]경고: 모든 DB(Redis, PostgreSQL)의 데이터를 초기화합니다.[/bold yellow]")
        redis_client.flushall()
        console.console.print("🔴 Redis DB가 초기화되었습니다.")
        try:
            vector_store.delete_collection()
            ensure_collection()
            console.console.print("🔴 PostgreSQL 벡터 DB가 초기화되었습니다.")
        except Exception as e:
            console.console.print(f"🟡 PostgreSQL 벡터 DB 초기화 중 참고: {e}")
    else:
        console.console.print("사전 학습된 DB를 사용하여 성능을 측정합니다.")

    try:
        with open(test_filepath, "r", encoding="utf-8") as f:
            test_cases = [json.loads(line) for line in f]
    except FileNotFoundError:
        console.console.print(f"[bold red]오류: 테스트 파일을 찾을 수 없습니다 - {test_filepath}[/bold red]")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"benchmark_results_{mode.replace(' ', '_')}_{timestamp}.csv"

    with open(results_filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = [
            "case_id",
            "expected_outcome",
            "model_outcome",
            "is_correct",
            "logical_consistency_score",
            "logical_consistency_reason",
            "legal_validity_score",
            "legal_validity_reason",
            "social_consideration_score",
            "social_consideration_reason",
        ]
        writer.writerow(header)

        total_scores = defaultdict(float)
        total_runs = 0
        paired_outcomes: List[Tuple[str, str]] = []

        for i, case in enumerate(test_cases):
            case_id = case.get("caseId", "N/A")
            console.console.print(Rule(f"[bold]테스트 케이스 {i + 1}/{len(test_cases)} 실행 (ID: {case_id})[/bold]"))

            initial_state = {
                "case_file": f"원고 주장: {case['plaintiff_statement']}\n피고 주장: {case['defendant_statement']}",
                "plaintiff_lawyer": "원고측 변호사",
                "defendant_lawyer": "피고측 변호사",
            }

            final_event = None
            for event in app.stream(initial_state):
                if "__end__" in event:
                    final_event = event["__end__"]

            # LangGraph의 최종 이벤트는 {"state": {...}} 혹은 {"return": {...}} 형태로
            # 래핑되어 전달될 수 있으므로, 실제 상태 딕셔너리를 안전하게 추출합니다.
            final_state_data = final_event or {}
            if isinstance(final_state_data, dict):
                if "state" in final_state_data and isinstance(final_state_data["state"], dict):
                    final_state = final_state_data["state"]
                elif "return" in final_state_data and isinstance(final_state_data["return"], dict):
                    final_state = final_state_data["return"]
                else:
                    final_state = final_state_data
            else:
                final_state = {}
            critique_scores = final_state.get("critique_scores", []) or []
            model_outcome = final_state.get("plaintiff_outcome")
            expected_outcome = case.get("expected_outcome")

            row_data: Dict[str, object] = {
                "case_id": case_id,
                "expected_outcome": expected_outcome or "N/A",
                "model_outcome": (model_outcome or ("미예측" if expected_outcome else "N/A")),
                "is_correct": "N/A",
            }

            for criteria, (score_key, reason_key) in CRITERIA_HEADERS.items():
                row_data[score_key] = 0
                row_data[reason_key] = "평가 결과가 기록되지 않았습니다."

            for item in critique_scores:
                criteria = item.get("criteria")
                if criteria not in CRITERIA_HEADERS:
                    continue
                score_key, reason_key = CRITERIA_HEADERS[criteria]
                score = int(item.get("score", 0))
                reason = item.get("reason") or "평가 이유가 제공되지 않았습니다."
                row_data[score_key] = score
                row_data[reason_key] = reason
                total_scores[criteria] += score

            if expected_outcome:
                predicted_label = model_outcome or "미예측"
                paired_outcomes.append((expected_outcome, predicted_label))
                if model_outcome:
                    row_data["is_correct"] = "Y" if expected_outcome == model_outcome else "N"
                else:
                    row_data["is_correct"] = "N"
            elif model_outcome:
                row_data["is_correct"] = "정보 부족"

            writer.writerow([row_data.get(h, "N/A") for h in header])
            total_runs += 1
            time.sleep(1)

    console.print_header(f"벤치마크 테스트 완료: {mode}")
    console.console.print(f"결과가 [bold cyan]{results_filename}[/bold cyan] 파일에 저장되었습니다.")

    table = Table(title="평균 점수 (Pass 비율)")
    table.add_column("평가 항목", justify="right", style="cyan", no_wrap=True)
    table.add_column("평균 점수 (%)", justify="center", style="magenta")

    if total_runs > 0:
        for criteria in CRITIQUE_CRITERIA:
            avg_score = (total_scores[criteria] / total_runs) * 100 if total_runs else 0.0
            table.add_row(criteria, f"{avg_score:.2f}%")

    console.console.print(table)

    accuracy = 0.0
    macro_f1 = 0.0
    expected_win_rate = 0.0
    model_win_rate = 0.0
    per_label_metrics: List[Tuple[str, float, float, float]] = []

    if paired_outcomes:
        total_cases = len(paired_outcomes)
        correct_predictions = sum(1 for expected, predicted in paired_outcomes if expected == predicted)
        accuracy = correct_predictions / total_cases
        expected_win_rate = sum(1 for expected, _ in paired_outcomes if expected == "승리") / total_cases
        model_win_rate = sum(1 for _, predicted in paired_outcomes if predicted == "승리") / total_cases
        labels = sorted({expected for expected, _ in paired_outcomes})
        if labels:
            f1_sum = 0.0
            for label in labels:
                tp = sum(1 for expected, predicted in paired_outcomes if expected == label and predicted == label)
                fp = sum(1 for expected, predicted in paired_outcomes if expected != label and predicted == label)
                fn = sum(1 for expected, predicted in paired_outcomes if expected == label and predicted != label)
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                f1_sum += f1
                per_label_metrics.append((label, precision, recall, f1))
            macro_f1 = f1_sum / len(labels) if labels else 0.0

    metrics_table = Table(title="판결 결과 정량 지표")
    metrics_table.add_column("지표", justify="left", style="green")
    metrics_table.add_column("값", justify="center", style="white")
    metrics_table.add_row("정확도 (Accuracy)", f"{accuracy * 100:.2f}%")
    metrics_table.add_row("Macro F1", f"{macro_f1:.4f}")
    metrics_table.add_row("예상 원고 승소율", f"{expected_win_rate * 100:.2f}%")
    metrics_table.add_row("모델 원고 승소율", f"{model_win_rate * 100:.2f}%")
    console.console.print(metrics_table)

    if per_label_metrics:
        label_table = Table(title="클래스별 Precision/Recall/F1")
        label_table.add_column("라벨", style="cyan")
        label_table.add_column("Precision", justify="center")
        label_table.add_column("Recall", justify="center")
        label_table.add_column("F1", justify="center")
        for label, precision, recall, f1 in per_label_metrics:
            label_table.add_row(label, f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}")
        console.console.print(label_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모의 법정 시스템 벤치마크 테스트")
    parser.add_argument("--mode", type=str, required=True, choices=["trained", "untrained"],
                        help="'trained' 또는 'untrained' 모드를 선택하세요.")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dataset_path = os.path.join(current_dir, "data", "test.jsonl")

    if args.mode == "untrained":
        run_benchmark(test_dataset_path, is_trained=False)
    else:
        run_benchmark(test_dataset_path, is_trained=True)
