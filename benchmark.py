"""Lightweight benchmarking harness for the new assistant workflow."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from rich import print
from rich.table import Table

from src.graph import app


def _load_cases(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def run_benchmark(dataset_path: Path, firm_id: int, user_id: int) -> None:
    cases = _load_cases(dataset_path)
    if not cases:
        print("[yellow]평가할 데이터가 없습니다.[/yellow]")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = dataset_path.with_name(f"benchmark_results_{timestamp}.csv")

    with output_file.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "case_id",
            "summary",
            "issues",
            "draft_text",
            "simulation_report",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            case_id = case.get("case_id") or case.get("id")
            document_text = case.get("document_text") or case.get("text")
            if not document_text:
                print(f"[red]문서 내용이 비어 있는 케이스를 건너뜁니다: {case_id}[/red]")
                continue

            state = {
                "firm_id": firm_id,
                "user_id": user_id,
                "file_name": f"benchmark_{case_id or 'unknown'}.txt",
                "mime_type": "text/plain",
                "file_bytes": document_text.encode("utf-8"),
                "enable_ocr": False,
                "user_feedback": None,
            }

            final_state: Dict[str, object] = {}
            for event in app.stream(state):
                final_state.update(event)

            writer.writerow(
                {
                    "case_id": case_id or "N/A",
                    "summary": final_state.get("summarize", {}).get("summary"),
                    "issues": " | ".join(final_state.get("summarize", {}).get("issues", [])),
                    "draft_text": final_state.get("draft", {}).get("draft_text"),
                    "simulation_report": final_state.get("simulate", {}).get("simulation_report"),
                }
            )

    print(f"[green]벤치마크 결과가 {output_file}에 저장되었습니다.[/green]")

    table = Table(title="샘플 결과 미리보기", show_lines=True)
    table.add_column("케이스 ID", style="cyan")
    table.add_column("요약", style="white", max_width=50, overflow="fold")
    table.add_column("주요 쟁점", style="magenta", max_width=30, overflow="fold")
    table.add_column("반론 시뮬레이션", style="green", max_width=40, overflow="fold")

    with output_file.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in list(reader)[:5]:
            table.add_row(
                row["case_id"],
                row["summary"] or "-",
                row["issues"] or "-",
                row["simulation_report"] or "-",
            )

    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="새로운 법률 어시스턴트 파이프라인 벤치마크")
    parser.add_argument("dataset", type=str, help="평가할 JSONL 데이터셋 경로")
    parser.add_argument("--firm-id", type=int, default=1)
    parser.add_argument("--user-id", type=int, default=1)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[red]데이터셋을 찾을 수 없습니다: {dataset_path}[/red]")
        return

    run_benchmark(dataset_path, firm_id=args.firm_id, user_id=args.user_id)


if __name__ == "__main__":
    main()
