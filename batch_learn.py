import json
import time
import os
from src.agents import (
    redis_client,
    batch_judge_chain,
    evaluation_chain,
    reflection_chain
)
from src.vector_db import add_case_to_db
import src.console as console
from rich.rule import Rule

def run_batch_learning(filepath: str):
    """
    .jsonl 파일로부터 여러 사건 데이터를 읽어와 일괄 학습을 수행합니다.
    """
    console.print_header("데이터셋 일괄 학습 시작")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cases = [json.loads(line) for line in f]
    except FileNotFoundError:
        console.console.print(f"[bold red]오류: 파일을 찾을 수 없습니다 - {filepath}[/bold red]")
        return
    except json.JSONDecodeError:
        console.console.print(f"[bold red]오류: 파일이 올바른 JSONL 형식이 아닙니다 - {filepath}[/bold red]")
        return
    
    console.console.print(f"총 {len(cases)}개의 사건을 학습합니다.\n")

    for i, case in enumerate(cases):
        case_id = case.get("caseId", "N/A")
        plaintiff_statement = case.get("plaintiff_statement", "")
        defendant_statement = case.get("defendant_statement", "")

        console.console.print(Rule(f"[bold]사건 {i+1}/{len(cases)} 처리 중 (ID: {case_id})[/bold]"))

        # 1. 모의 판결 생성
        console.console.print("1. 재판장 에이전트가 모의 판결 생성 중...")
        verdict_response = batch_judge_chain.invoke({
            "plaintiff_statement": plaintiff_statement,
            "defendant_statement": defendant_statement
        })
        final_verdict = verdict_response.content.strip()
        console.print_final_verdict(final_verdict)

        # 2. 승패 분석
        console.console.print("2. 평가 에이전트가 승패 분석 중...")
        evaluation_response = evaluation_chain.invoke({"final_verdict": final_verdict})
        plaintiff_outcome = evaluation_response.content.strip()
        defendant_outcome = "승리" if plaintiff_outcome == "패배" else ("패배" if plaintiff_outcome == "승리" else "무승부")
        
        outcomes = {
            "원고측 변호사": {"outcome": plaintiff_outcome, "db_key_prefix": "plaintiff_lawyer", "speech": plaintiff_statement},
            "피고측 변호사": {"outcome": defendant_outcome, "db_key_prefix": "defendant_lawyer", "speech": defendant_statement}
        }
        
        lessons = {}
        
        # 3. 양측 교훈 도출
        console.console.print("3. 회고 에이전트가 교훈 도출 중...")
        for lawyer_name, info in outcomes.items():
            reflection_response = reflection_chain.invoke({
                "outcome": info['outcome'],
                "my_speeches": info['speech']
            })
            lesson = reflection_response.content.strip()
            lessons[info['db_key_prefix']] = lesson
            console.print_lesson(lawyer_name, info['outcome'], lesson)
            
            # 4. 개인 DB (Redis) 업데이트
            if info['outcome'] == "승리":
                redis_client.rpush(f"{info['db_key_prefix']}:successful_strategies", lesson)
            elif info['outcome'] == "패배":
                redis_client.rpush(f"{info['db_key_prefix']}:failed_strategies", lesson)

        # 5. 사건 아카이브 (PostgreSQL) 업데이트
        case_summary = f"원고 주장: {plaintiff_statement[:100]}...\n피고 주장: {defendant_statement[:100]}..."
        add_case_to_db(
            case_summary=case_summary,
            verdict=final_verdict,
            plaintiff_lesson=lessons.get("plaintiff_lawyer", ""),
            defendant_lesson=lessons.get("defendant_lawyer", "")
        )
        
        # API 속도 제한 방지를 위해 잠시 대기
        time.sleep(1) 

    console.print_header("데이터셋 일괄 학습 완료")

if __name__ == "__main__":
    # 현재 스크립트 파일의 위치를 기준으로 데이터 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "data", "train.jsonl")
    run_batch_learning(dataset_path)