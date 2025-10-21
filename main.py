"""Command line entry point to demonstrate the assistant workflow."""
from __future__ import annotations

from rich import print

from src.graph import app

if __name__ == "__main__":
    sample_document = (
        "원고는 아파트 층간소음으로 정신적 손해를 입었다고 주장하며 피고에게 위자료 지급을 청구한다. "
        "피고는 적법한 방음 공사를 수행했고 공동주택관리규약을 준수했다고 항변한다."
    )

    initial_state = {
        "firm_id": 1,
        "user_id": 1,
        "file_name": "sample_case.txt",
        "mime_type": "text/plain",
        "file_bytes": sample_document.encode("utf-8"),
        "enable_ocr": False,
    }

    print("[bold cyan]🚀 B2B 법률 AI 어시스턴트 파이프라인을 실행합니다.[/bold cyan]")

    for event in app.stream(initial_state):
        for node, value in event.items():
            print(f"\n[green]노드 '{node}' 완료[/green]")
            if node == "summarize":
                print(f"요약: {value.get('summary')}")
                print(f"쟁점: {value.get('issues')}")
            elif node == "rag":
                print("참조 근거:")
                for item in value.get("rag_results", []):
                    print(f"  - {item['source']}")
            elif node == "draft":
                print("초안 미리보기:")
                print(value.get("draft_text", ""))
            elif node == "simulate":
                print("시뮬레이션 결과:")
                print(value.get("simulation_report", ""))
            elif node == "feedback":
                saved = value.get("feedback_saved")
                print(f"피드백 저장 여부: {saved}")
