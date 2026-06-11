from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from rag_tool import rag_answer

#defining graph state
class AgentState(TypedDict):
    question: str
    route: str
    result: str
    answer: str

#router node 
def router_node(state: AgentState) -> AgentState:
    return {"route": "document"}

#route decision function
def route_decision(state: AgentState) -> Literal["document"]:
    return state["route"]

#document node 
def document_node(state: AgentState) -> AgentState:
    result = rag_answer(state["question"])
    return {"result": result}

#final answer node
def final_node(state: AgentState) -> AgentState:
    if state["route"] == "document":
        return {"answer": state["result"]}

#building the graph
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("document", document_node)
graph.add_node("final_node", final_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "document": "document",
    }
)

graph.add_edge("document", "final_node")
graph.add_edge("final_node", END)

app = graph.compile()