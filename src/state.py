from typing import List, TypedDict, Optional

class AgentSpeech(TypedDict):
    """에이전트의 발언을 저장하는 형식"""
    agent_name: str
    speech: str

class TrialState(TypedDict):
    """재판 전체의 상태를 관리하는 형식"""
    case_file: str
    plaintiff_lawyer: str
    defendant_lawyer: str
    selected_judges: List[dict]
    debate_transcript: List[AgentSpeech]
    turn_count: int
    max_turns: int
    associate_judge_verdicts: List[AgentSpeech]
    final_verdict: Optional[str]
    plaintiff_outcome: Optional[str]
    critique_scores: Optional[list]  # 👈 벤치마크 점수를 저장할 필드
