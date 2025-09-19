import random
import json
from src.state import TrialState
import src.console as console
from src.agents import (
    redis_client,
    JUDGE_PERSONALITY_POOL,
    lawyer_chain,
    judge_chain,
    presiding_judge_chain,
    evaluation_chain,
    reflection_chain,
    critic_chain
)
from src.vector_db import add_case_to_db, search_similar_cases

def start_trial(state: TrialState):
    """재판 시작: 초기 설정 및 서브 판사 3명 무작위 선택"""
    console.print_header("모의 법정 시뮬레이션을 시작합니다")
    state['max_turns'] = 4
    state['turn_count'] = 0
    state['debate_transcript'] = []
    
    selected_judges = random.sample(JUDGE_PERSONALITY_POOL, 3)
    state['selected_judges'] = selected_judges
    
    console.print_judge_panel(selected_judges)
    return state

def lawyer_debate_node(state: TrialState):
    """변호사 토론: 유사 사건 검색 및 개인 DB를 바탕으로 변론"""
    turn = state['turn_count'] + 1
    console.print_turn_header(turn)
    
    transcript_str = "\n".join(
        [f"{msg['agent_name']}: {msg['speech']}" for msg in state['debate_transcript']]
    )

    if state['turn_count'] % 2 == 0:
        speaker_name = state['plaintiff_lawyer']
        client_type = "원고"
        db_key_prefix = "plaintiff_lawyer"
    else:
        speaker_name = state['defendant_lawyer']
        client_type = "피고"
        db_key_prefix = "defendant_lawyer"
    
    similar_cases_str = search_similar_cases(state['case_file'])
    
    successful_lessons = "\n".join(redis_client.lrange(f"{db_key_prefix}:successful_strategies", 0, -1))
    failed_lessons = "\n".join(redis_client.lrange(f"{db_key_prefix}:failed_strategies", 0, -1))
    past_lessons_str = f"성공 전략:\n{successful_lessons}\n\n실패 전략:\n{failed_lessons}"
    
    if not successful_lessons and not failed_lessons:
        past_lessons_str = "아직 재판 경험이 없습니다."
        
    response_ai = lawyer_chain.invoke({
        "client_type": client_type,
        "case_file": state['case_file'],
        "transcript": transcript_str,
        "past_lessons": past_lessons_str,
        "similar_cases": similar_cases_str
    })
    response = response_ai.content
        
    console.print_speech(speaker_name, response)
    state['debate_transcript'].append({"agent_name": speaker_name, "speech": response})
    state['turn_count'] += 1
    return state

def associate_judge_deliberation_node(state: TrialState):
    """서브 판사 심의: 실제 LLM을 호출하여 페르소나 기반 판결"""
    console.print_verdict_header("서브 판사 심의")
    
    transcript_str = "\n".join(
        [f"{msg['agent_name']}: {msg['speech']}" for msg in state['debate_transcript']]
    )
    
    verdicts = []
    for i, judge_info in enumerate(state['selected_judges']):
        judge_name = judge_info['name']
        judge_description = judge_info['description']
        
        response_ai = judge_chain.invoke({
            "judge_name": judge_name,
            "judge_description": judge_description,
            "transcript": transcript_str
        })
        verdict = response_ai.content
        
        console.print_speech(judge_name, verdict)
        verdicts.append({"agent_name": judge_name, "speech": verdict})
    
    state['associate_judge_verdicts'] = verdicts
    return state

def final_judgment_node(state: TrialState):
    """최종 판결: 재판장 LLM이 모든 내용을 종합하여 판결문 생성"""
    console.print_verdict_header("최종 판결 선고")
    
    transcript_str = "\n".join(
        [f"{msg['agent_name']}: {msg['speech']}" for msg in state['debate_transcript']]
    )
    judge_verdicts_str = "\n\n".join(
        [f"[{msg['agent_name']}의 의견]\n{msg['speech']}" for msg in state['associate_judge_verdicts']]
    )
    
    response_ai = presiding_judge_chain.invoke({
        "transcript": transcript_str,
        "judge_verdicts": judge_verdicts_str
    })
    final_verdict = response_ai.content
    
    console.print_final_verdict(final_verdict)
    state['final_verdict'] = final_verdict
    return state

def update_knowledge_base_node(state: TrialState):
    """변호사 DB 업데이트 및 이번 사건을 벡터 DB에 저장"""
    console.print_update_header()
    
    evaluation_response = evaluation_chain.invoke({"final_verdict": state['final_verdict']})
    plaintiff_outcome = evaluation_response.content.strip()
    console.console.print(f"분석 결과: 원고측 '{plaintiff_outcome}'\n")

    outcomes = {
        state['plaintiff_lawyer']: {"outcome": plaintiff_outcome, "db_key_prefix": "plaintiff_lawyer"},
        state['defendant_lawyer']: {
            "outcome": "승리" if plaintiff_outcome == "패배" else ("패배" if plaintiff_outcome == "승리" else "무승부"),
            "db_key_prefix": "defendant_lawyer"
        }
    }

    lessons = {}
    for lawyer_name, info in outcomes.items():
        outcome = info['outcome']
        db_key_prefix = info['db_key_prefix']
        
        my_speeches = "\n".join(
            [s['speech'] for s in state['debate_transcript'] if s['agent_name'] == lawyer_name]
        )
        
        reflection_response = reflection_chain.invoke({"outcome": outcome, "my_speeches": my_speeches})
        lesson = reflection_response.content.strip()
        lessons[db_key_prefix] = lesson
        
        console.print_lesson(lawyer_name, outcome, lesson)

        if outcome == "승리":
            redis_client.rpush(f"{db_key_prefix}:successful_strategies", lesson)
        elif outcome == "패배":
            redis_client.rpush(f"{db_key_prefix}:failed_strategies", lesson)

    add_case_to_db(
        case_summary=state['case_file'],
        verdict=state['final_verdict'],
        plaintiff_lesson=lessons.get("plaintiff_lawyer", "N/A"),
        defendant_lesson=lessons.get("defendant_lawyer", "N/A")
    )
    
    return state

def critique_node(state: TrialState):
    """비평가 에이전트가 최종 판결을 평가하고 점수를 State에 기록합니다."""
    console.print_verdict_header("판결 품질 평가 (벤치마크 점수)")

    transcript_str = "\n".join(
        [f"{msg['agent_name']}: {msg['speech']}" for msg in state['debate_transcript']]
    )

    critique_response = critic_chain.invoke({
        "transcript": transcript_str,
        "final_verdict": state['final_verdict']
    })
    
    scores = []
    try:
        scores = json.loads(critique_response.content.strip())
        
        console.console.print("\n[bold]판결 품질 벤치마크:[/bold]")
        for item in scores:
            result = "[bold green]PASS[/bold green]" if item.get('score') == 1 else "[bold red]FAIL[/bold red]"
            console.console.print(f"- [bold]{item.get('criteria', 'N/A')}[/bold]: {result}")
            console.console.print(f"  (평가 이유: {item.get('reason', 'N/A')})")
            
    except (json.JSONDecodeError, TypeError):
        console.console.print("\n[bold yellow]품질 평가 결과(Raw):[/bold yellow]")
        console.console.print(critique_response.content)

    state['critique_scores'] = scores
    return state