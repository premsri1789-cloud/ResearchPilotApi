from typing import List, Literal
from typing_extensions import TypedDict

class AgentState(TypedDict):
    session_id: str
    user_id: str
    question: str
    documents: List[str]
    web_results: List[str]
    generation: str
    chat_history: List[dict]    # Incoming history from SQLite
    rewritten_query: str        # The contextually resolved question