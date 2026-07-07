import os
import sys
import json
from pathlib import Path
from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_openai import OpenAIEmbeddings
from tavily import TavilyClient

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.core.vector_db import qdrant_client
from graph.state import AgentState

load_dotenv()
tavily_key = os.getenv("TAVILY_API_KEY")

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
#groq_initializing
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


tavily = TavilyClient(api_key=tavily_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def query_rewriter_node(state: AgentState) -> dict:
    """Uses chat history to rewrite the user query so it can be searched accurately."""
    print("--- [NODE] REWRITING QUERY (MEMORY RESOLUTION) ---")

    history = state["chat_history"]
    question = state["question"]

    if not history:
        print(f"No history found. Using original question: {question}")
        return {"rewritten_query": question}

    # Format history into text for the LLM
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    prompt = f"""Given the following conversation history and the user's new question,
    rewrite the question to be a standalone query that contains all context needed for a vector database search.
    If the question doesn't refer to past context, just return it identically.

    Chat History:
    {history_text}

    User's New Question: {question}

    Respond ONLY with the rewritten standalone question, nothing else."""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )
    rewritten = response.content.strip()
    print(f"Rewritten Query: {rewritten}")
    
    return {"rewritten_query": rewritten}    

def retrieve_node(state: AgentState) -> dict:
    """Retrieves document chunks from Qdrant strictly scoped to the session_id."""
    print("--- [NODE] RETRIEVING FROM VECTOR DB ---")

    search_query = state.get("rewritten_query", state["question"])
    session_id = state["session_id"]
    user_id = state["user_id"] # Get user_id from state

    # Embed question
    query_vector = embedding_model.embed_query(search_query)

    # Query Qdrant with metadata filtering
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

    results = response.points

    if results:
        docs = [res.payload["text"] for res in results]
    else:
        docs = []

    return {"documents": docs}

def grade_documents_node(state: AgentState) -> dict:
    """Uses Groq to evaluate if the retrieved documents actually answer the question."""
    print("--- [NODE] GRADING RETRIEVED DOCUMENTS ---")

    search_query = state.get("rewritten_query", state["question"])
    docs = state["documents"]

    if not docs:
        return {"documents": []}
    
    # Instruct Groq to output JSON for easy parsing
    prompt = f"""You are a strict grader assessing relevancy of the retrieved document
     to the user question.
     Question: {search_query}
     Documents: {docs}

     Check does the document contain keyword(s) or semantic meaning that answers the user question. 
     Respond strictly in json format with single key 'score' and value only 'yes' or 'no'."""
    
    response = llm.invoke(
        [SystemMessage(content=prompt)],
        response_format={"type": "json_object"}
    )

    try:
        score = json.loads(response.content).get("score", "no")
    except:
        score = "no"

    print(f"Grader Score: {score}")

    if score == "yes":
        return {"documents": docs}
    else:
        return {"documents": []}
    
def web_search_node(state: AgentState) -> dict:
    """Falls back to Tavily Web Search if local documents are insufficient."""
    search_query = state.get("rewritten_query", state["question"])

    tavily_search = tavily.search(query=search_query, search_depth="advanced", max_results=2)
    search_result = [res["content"] for res in tavily_search.get("results", [])]

    return {"web_results": search_result}

def generate_node(state: AgentState) -> dict:
    """Generates final LLM response based on retrieved documents or tavily search for the user question"""
    print("--- [NODE] GENERATING FINAL ANSWER ---")

    question = state["question"]
    docs = state["documents"]
    web_results = state["web_results"]

    context = "\n\n".join(docs + web_results)

    prompt= f""" You are an expert AI research assistant. Answer the user's question based strictly on the provided context.
    If the context contains web results, mention that you searched the web.
    
    Context:
    {context}
    
    User Question: {question} """

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {"generation": response.content}

# ==========================================
# CONDITIONAL EDGES
# ==========================================
def conditional_route(state: AgentState) -> Literal['web_search', 'generate']:
    """Determines whether to generate immediately or fall back to web search."""
    if not state.get("documents"):
        print("--- [ROUTING] Documents irrelevant/missing -> Routing to Web Search ---")
        return "web_search"
    print("--- [ROUTING] Documents relevant -> Routing to Generate ---")
    return "generate"

# ==========================================
# BUILD & COMPILE GRAPH
# ==========================================
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

app = graph.compile()
        

def run_agent(session_id: str, user_id: str, question: str, history: list) -> str:

    initial_state = {
        "session_id": session_id,
        "user_id": user_id,     # Pass user_id into state
        "question": question,
        "chat_history": history,
        "documents": [],
        "web_results": [],
        "generation": ""
    }

    # 2. Invoke the Graph Engine
    result = app.invoke(initial_state)
    final_answer = result["generation"]

    return final_answer

