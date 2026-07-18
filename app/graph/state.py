from typing import List
from typing_extensions import TypedDict

class AgentState(TypedDict):
    session_id: str
    user_id: str
    question: str
    documents: List[str]
    web_results: List[str]
    generation: str
    chat_history: List[dict]
    rewritten_query: str