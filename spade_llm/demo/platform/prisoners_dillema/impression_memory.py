from spade_llm.demo import models
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing import Dict, Optional

embeddings = models.EMBEDDINGS

AGENT_VECTOR_MEMORIES: Dict[int, VectorStore] = {}


def get_agent_memory(agent_id: int) -> VectorStore:
    """
    Персональное векторное хранилище агента agent_id
    """
    if agent_id not in AGENT_VECTOR_MEMORIES:
        AGENT_VECTOR_MEMORIES[agent_id] = Chroma(
            collection_name=f"agent_{agent_id}_reputation",
            embedding_function=embeddings,
        )
    return AGENT_VECTOR_MEMORIES[agent_id]


def add_agent_impression(
    agent_id: int,
    target_agent_id: int,
    text: str,
    score: Optional[int] = None
):
    """
    Агент agent_id сохраняет факт о target_agent_id
    """
    memory = get_agent_memory(agent_id)

    doc = Document(
        page_content=text,
        metadata={
            "target_agent_id": target_agent_id,
            "score": score
        }
    )

    memory.add_documents([doc])


def get_agent_impression(
    agent_id: int,
    target_agent_id: int,
    k: int = 5
) -> str:
    """
    Агент agent_id извлекает СВОЁ мнение о target_agent_id
    """
    if agent_id not in AGENT_VECTOR_MEMORIES:
        return "Нет предыдущих взаимодействий с этим агентом."

    memory = AGENT_VECTOR_MEMORIES[agent_id]

    docs = memory.similarity_search(
        query=f"поведение агента {target_agent_id}",
        k=k,
        filter={"target_agent_id": target_agent_id}
    )

    if not docs:
        return "Нет предыдущих взаимодействий с этим агентом."

    return "\n".join(d.page_content for d in docs)






