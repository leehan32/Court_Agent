# 파일명: src/llm_client.py (신규)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 데모를 위해 OpenAI 모델을 사용합니다. (OPENAI_API_KEY 필요)
# 로컬 모델(Llama 3, SOLAR)을 사용하려면 이 부분을 수정하세요.
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

RAG_PROMPT_TEMPLATE = """
당신은 변호사를 보조하는 전문 법률 AI 어시스턴트입니다.
제시된 [참고 자료]를 바탕으로 변호사의 [질문]에 대해 전문적이고 간결하게 답변하세요.
답변은 반드시 [참고 자료]에 근거해야 하며, 자료에 없는 내용은 추측하지 마세요.

[참고 자료]
{context}

[질문]
{query}

[답변]
"""

def get_rag_answer(query, context_chunks):
    """검색된 RAG 청크와 쿼리를 바탕으로 LLM 답변을 생성합니다."""
    
    context_str = "\n\n---\n\n".join(
        f"출처: {chunk['source']}\n내용: {chunk['text']}" for chunk in context_chunks
    )
    
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "context": context_str,
        "query": query
    })
    return response, context_str