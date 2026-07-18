import os
import json
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding
from tavily import TavilyClient # or standard from tavily import TavilyClient

from app.core.vector_db import qdrant_client
from app.graph.state import AgentState

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def query_rewriter_node(state: AgentState) -> dict:
    history = state["chat_history"]
    question = state["question"]
    if not history:
        return {"rewritten_query": question}
    
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = f"Rewrite this question to be a standalone query containing context from history.\nHistory:\n{history_text}\nQuestion: {question}"
    return {"rewritten_query": llm.invoke([HumanMessage(content=prompt)]).content.strip()}

def retrieve_node(state: AgentState) -> dict:
    search_query = state.get("rewritten_query", state["question"])
    session_id = state["session_id"]
    user_id = state["user_id"]

    embeddings_generator = embedding_model.embed([search_query])
    query_vector = list(embeddings_generator)[0].tolist()

    # Local Qdrant handles unindexed field matching seamlessly without throwing a 400 error!
    response = qdrant_client.query_points(
        collection_name="Research_documents",
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ]
        ),
        limit=3
    )
    return {"documents": [res.payload["text"] for res in response.points]}

def grade_documents_node(state: AgentState) -> dict:
    search_query = state.get("rewritten_query", state["question"])
    docs = state["documents"]
    if not docs:
        return {"documents": []}
    
    prompt = f"Evaluate relevancy. Question: {search_query}\nDocs: {docs}\nRespond strictly in JSON: {{'score': 'yes' | 'no'}}"
    response = llm.invoke([SystemMessage(content=prompt)], response_format={"type": "json_object"})
    try:
        score = json.loads(response.content).get("score", "no")
    except:
        score = "no"
    return {"documents": docs if score == "yes" else []}

def web_search_node(state: AgentState) -> dict:
    search_query = state.get("rewritten_query", state["question"])
    res = tavily.search(query=search_query, max_results=2)
    return {"web_results": [r["content"] for r in res.get("results", [])]}

def generate_node(state: AgentState) -> dict:
    context = "\n\n".join(state["documents"] + state["web_results"])
    prompt = f"Answer the user question using this context.\nContext:\n{context}\nQuestion: {state['question']}"
    return {"generation": llm.invoke([HumanMessage(content=prompt)]).content}

def conditional_route(state: AgentState) -> Literal['web_search', 'generate']:
    return "generate" if state.get("documents") else "web_search"

graph = StateGraph(AgentState)
graph.add_node("query_rewriter", query_rewriter_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade_documents", grade_documents_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "query_rewriter")
graph.add_edge("query_rewriter", "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", conditional_route)
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)
app_graph = graph.compile()

def run_agent(session_id: str, user_id: str, question: str, history: list) -> str:
    res = app_graph.invoke({
        "session_id": session_id, "user_id": user_id, "question": question,
        "chat_history": history, "documents": [], "web_results": [], "generation": ""
    })
    return res["generation"]