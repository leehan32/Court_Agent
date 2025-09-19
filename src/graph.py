from langgraph.graph import StateGraph, END
from src.state import TrialState
from src.nodes import (
    start_trial,
    lawyer_debate_node,
    associate_judge_deliberation_node,
    final_judgment_node,
    update_knowledge_base_node,
    critique_node,
)

# 조건부 엣지를 위한 함수
def should_continue_debate(state: TrialState):
    return "continue_debate" if state['turn_count'] < state['max_turns'] else "end_debate"

# 그래프 워크플로우 생성
workflow = StateGraph(TrialState)

# 노드 추가
workflow.add_node("start_trial", start_trial)
workflow.add_node("lawyer_debate", lawyer_debate_node)
workflow.add_node("associate_judge_deliberation", associate_judge_deliberation_node)
workflow.add_node("final_judgment", final_judgment_node)
workflow.add_node("update_knowledge_base", update_knowledge_base_node)
workflow.add_node("critique", critique_node)

# 엣지(흐름) 연결
workflow.set_entry_point("start_trial")
workflow.add_edge("start_trial", "lawyer_debate")
workflow.add_conditional_edges(
    "lawyer_debate",
    should_continue_debate,
    {"continue_debate": "lawyer_debate", "end_debate": "associate_judge_deliberation"}
)
workflow.add_edge("associate_judge_deliberation", "final_judgment")
workflow.add_edge("final_judgment", "update_knowledge_base")
workflow.add_edge("update_knowledge_base", "critique")
workflow.add_edge("critique", END)

# 그래프 컴파일
app = workflow.compile()
