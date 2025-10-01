import os
from typing import List, Literal, Optional

import redis
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
__all__ = (
    "llm",
    "redis_client",
    "lawyer_chain",
    "judge_chain",
    "presiding_judge_chain",
    "batch_judge_chain",
    "evaluation_chain",
    "reflection_chain",
    "critic_chain",
    "CRITIQUE_CRITERIA",
    "JUDGE_PERSONALITY_POOL",
)

# .env 파일에서 환경 변수 로드
load_dotenv()


def _init_llm() -> BaseChatModel:
    """환경 변수에 따라 사용할 LLM 클라이언트를 초기화합니다."""

    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    if llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "환경 변수에 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )

        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(model=openai_model, temperature=temperature)

    if llm_provider == "nvidia":
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as exc:  # pragma: no cover - import error 확인용
            raise ImportError(
                "langchain-nvidia-ai-endpoints 패키지가 설치되어 있어야 합니다."
            ) from exc

        if not os.getenv("NVIDIA_API_KEY"):
            raise ValueError(
                "환경 변수에 NVIDIA_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )

        nvidia_model = os.getenv("NVIDIA_NIM_MODEL", "meta/llama3-70b-instruct")
        base_url: Optional[str] = os.getenv("NVIDIA_NIM_BASE_URL")

        llm_kwargs = {"model": nvidia_model, "temperature": temperature}
        if base_url:
            llm_kwargs["base_url"] = base_url

        return ChatNVIDIA(**llm_kwargs)

    raise ValueError(
        "지원하지 않는 LLM_PROVIDER 값입니다. openai 또는 nvidia 중 하나를 사용해주세요."
    )


# 사용할 LLM 모델 설정
llm = _init_llm()


class CritiqueItem(BaseModel):
    """판결 품질 평가 항목을 구조화한 스키마."""

    criteria: Literal["논리적 일관성", "법률적 타당성", "사회적 가치 고려"] = Field(
        ...,
        description="평가 기준 이름",
    )
    score: Literal[0, 1] = Field(
        ...,
        description="기준 충족 여부. 충족 시 1, 미충족 시 0",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="해당 점수를 부여한 이유",
    )


class CritiqueEvaluation(BaseModel):
    """세 가지 기준에 대한 평가 결과 목록."""

    evaluations: List[CritiqueItem] = Field(
        ...,
        min_items=3,
        description="각 기준별 평가 결과 목록",
    )


CRITIQUE_CRITERIA: List[str] = [
    "논리적 일관성",
    "법률적 타당성",
    "사회적 가치 고려",
]
class CritiqueItem(BaseModel):
    """판결 품질 평가 항목을 구조화한 스키마."""

    criteria: Literal["논리적 일관성", "법률적 타당성", "사회적 가치 고려"] = Field(
        ...,
        description="평가 기준 이름",
    )
    score: Literal[0, 1] = Field(
        ...,
        description="기준 충족 여부. 충족 시 1, 미충족 시 0",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="해당 점수를 부여한 이유",
    )


class CritiqueEvaluation(BaseModel):
    """세 가지 기준에 대한 평가 결과 목록."""

    evaluations: List[CritiqueItem] = Field(
        ...,
        min_items=3,
        description="각 기준별 평가 결과 목록",
    )


CRITIQUE_CRITERIA: List[str] = [
    "논리적 일관성",
    "법률적 타당성",
    "사회적 가치 고려",
]


# ------------------- 데이터베이스 클라이언트 -------------------
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ------------------- 변호사 에이전트 -------------------
lawyer_prompt_template = """
# 역할(Role)
당신은 {client_type}을(를) 대리하는 유능한 변호사입니다.

# 목표(Goal)
당신의 목표는 주어진 사건 파일과 재판의 맥락을 완벽하게 파악하여, 의뢰인에게 '가장 유리하고 현실적인 결과'를 이끌어 내는 것입니다.
- 만약 의뢰인의 책임이 명백한 상황이라면, 무조건적인 무죄나 승소를 주장하는 대신, '책임과 손해를 최소화'하는 방향으로 변론을 이끌어야 합니다.
- 불필요한 억지 주장을 펼쳐 재판부에 나쁜 인상을 주지 않도록 주의해야 합니다.

# 행동 강령(Code of Conduct)
1.  **근거 기반 주장**: 모든 주장은 사건 파일에 명시된 사실이나 증거에 기반해야 합니다.
2.  **논리적 반박**: 상대 변호사의 주장을 먼저 요약하고, 그 주장의 논리적 허점을 명확하게 반박하세요.
3.  **핵심 집중**: 사건의 핵심 쟁점에서 벗어나는 발언은 최소화하세요.
4.  **과거 경험 학습**: 아래 '과거 재판의 교훈'을 참고하여 성공 전략은 강화하고, 실패 전략은 피하세요.
5.  **유사 사건 참고**: 아래 '유사 과거 사건' 정보를 현재 사건과 비교하여 변론 전략을 세우세요.

---
[사건 파일]
{case_file}

[유사 과거 사건]
{similar_cases}

[과거 재판의 교훈 (개인 DB)]
{past_lessons}

[이전 토론 기록]
{transcript}
---
위 정보를 바탕으로, 이제 당신의 차례입니다. 의뢰인을 위해 최고의 변론을 펼치세요.
"""
lawyer_prompt = ChatPromptTemplate.from_template(lawyer_prompt_template)
lawyer_chain = lawyer_prompt | llm

# ------------------- 서브 판사 에이전트 -------------------
judge_prompt_template = """
# 역할(Role)
당신은 합의부의 서브 판사입니다. 당신의 판단 스타일은 다음과 같습니다.
- 이름: {judge_name}
- 판단 기준: {judge_description}
# 임무(Mission)
아래의 변호사 토론 기록 전체를 읽고, 당신의 '판단 기준'에 입각하여 이 사건에 대한 당신의 의견을 한 문단으로 명확하게 제시하세요.
---
[변호사 토론 기록]
{transcript}
---
[당신의 의견]
"""
judge_prompt = ChatPromptTemplate.from_template(judge_prompt_template)
judge_chain = judge_prompt | llm

# ------------------- 재판장 에이전트 -------------------
presiding_judge_prompt_template = """
# 역할(Role)
당신은 재판을 총괄하는 재판장입니다.
# 임무(Mission)
당신의 임무는 아래에 제시된 '변호사 토론 기록'과 '서브 판사들의 의견'을 모두 종합하여, 최종 판결문을 작성하는 것입니다.
판결문은 반드시 "주문:"으로 시작하고, 간결하고 명확한 판결 요지를 담아야 합니다.
---
[변호사 토론 기록]
{transcript}
[서브 판사들의 의견]
{judge_verdicts}
---
[최종 판결문]
"""
presiding_judge_prompt = ChatPromptTemplate.from_template(presiding_judge_prompt_template)
presiding_judge_chain = presiding_judge_prompt | llm

# ------------------- 일괄 판결 생성 에이전트 -------------------
batch_judge_prompt_template = """
# 역할(Role)
당신은 재판을 주재하는 판사입니다.
# 임무(Mission)
아래에 주어진 원고와 피고의 진술을 바탕으로 사건의 쟁점을 정리하고, 최종 판결문을 작성하세요.
판결문은 반드시 "주문:"으로 시작해야 하며, 핵심 판시 이유를 2~3문장으로 덧붙이세요.
과도하게 긴 설명은 피하고 명확하고 간결하게 작성합니다.
---
[원고측 진술]
{plaintiff_statement}
[피고측 진술]
{defendant_statement}
---
[최종 판결문]
"""
batch_judge_prompt = ChatPromptTemplate.from_template(batch_judge_prompt_template)
batch_judge_chain = batch_judge_prompt | llm

# ------------------- 학습/진화 에이전트 -------------------
evaluation_prompt_template = """
# 역할(Role)
당신은 재판 분석가입니다.
# 임무(Mission)
아래의 최종 판결문을 분석하여 '원고' 입장에서의 승패 여부를 판단해주세요.
- 원고의 청구가 전부 또는 대부분 받아들여졌다면 "승리"
- 원고의 청구가 전부 또는 대부분 기각되었다면 "패배"
- 그 외의 경우(일부 인용, 조정 등)는 "무승부"
오직 "승리", "패배", "무승부" 중 하나의 단어로만 답변하세요.
---
[최종 판결문]
{final_verdict}
---
[원고측 승패 여부]
"""
evaluation_prompt = ChatPromptTemplate.from_template(evaluation_prompt_template)
evaluation_chain = evaluation_prompt | llm

reflection_prompt_template = """
# 역할(Role)
당신은 변론 전략 코치입니다.
# 임무(Mission)
아래의 '재판 결과'와 '변론 내용'을 바탕으로, 이 변론에서 가장 유효했거나 혹은 가장 아쉬웠던 핵심 전략을 한 문장으로 요약하여 '교훈'을 도출해주세요.
---
[재판 결과]
{outcome}
[자신의 변론 내용]
{my_speeches}
---
[핵심 전략 및 교훈]
"""
reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)
reflection_chain = reflection_prompt | llm

# ------------------- 비평가 에이전트 (논문 방식 적용) -------------------
critic_prompt_template = """
# 역할(Role)
당신은 냉철하고 객관적인 법률 분석가입니다.
# 임무(Mission)
아래의 '변호사 토론 기록'과 '최종 판결문'을 읽고, 다음 세 가지 기준 각각에 대해 판결의 품질이 기준을 '충족'하는지 '미충족'하는지 판단해주세요.
- 기준을 충족하면 score에 1, 미충족이면 0을 부여하세요.
- 그 이유(reason)를 한 문장으로 간결하게 설명해주세요.
결과는 반드시 하나의 JSON 객체로만 출력해야 하며, 아래 스키마를 따라야 합니다.
{{
    "evaluations": [
        {{"criteria": "논리적 일관성", "score": 0 또는 1, "reason": "..."}},
        {{"criteria": "법률적 타당성", "score": 0 또는 1, "reason": "..."}},
        {{"criteria": "사회적 가치 고려", "score": 0 또는 1, "reason": "..."}}
    ]
}}
# 평가 기준
1.  **논리적 일관성 (Logical Consistency)**: 변호사들의 주장과 최종 판결의 논리가 일관되는가?
2.  **법률적 타당성 (Legal Validity)**: 판결이 법률 원칙과 상식적인 법 감정에 부합하는가?
3.  **사회적 가치 고려 (Moral & Social Consideration)**: 판결이 법을 넘어선 윤리적, 사회적 가치를 충분히 고려하였는가?
---
[변호사 토론 기록]
{transcript}
[최종 판결문]
{final_verdict}
---
"""
critic_prompt = ChatPromptTemplate.from_template(critic_prompt_template)
critic_chain = critic_prompt | llm.with_structured_output(CritiqueEvaluation)

# ------------------- 데이터 및 설정 -------------------
JUDGE_PERSONALITY_POOL = [
    {"name": "법리/판례 기반 판사", "description": "과거의 법률과 유사 판례에만 근거하여 판단합니다. 사회적 여론이나 개인적인 감정은 철저히 배제합니다."},
    {"name": "윤리/도덕 기반 판사", "description": "법의 잣대를 넘어, 이 사건이 윤리적으로 어떤 의미를 갖는지, 그리고 더 정의로운 결정은 무엇인지 고심합니다."},
    {"name": "사회적 이슈 기반 판사", "description": "이 판결이 사회에 미칠 영향과 현재의 사회적 통념 및 여론을 가장 중요한 판단 기준으로 삼습니다."},
    {"name": "결과주의 판사", "description": "판결의 과정보다는, 어떤 판결이 궁극적으로 사회에 가장 큰 이익을 가져올 것인지 그 결과에만 집중합니다."},
    {"name": "원칙주의 판사", "description": "어떤 상황에서든 예외를 허용하지 않으며, 정해진 법률 원칙을 고수하는 것을 최우선으로 생각합니다."}
]