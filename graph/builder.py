from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from core.state import FinancialState
from graph.nodes import process_document, generate_report, aggregate_web_expenses


def fan_out_documents(state: FinancialState):
    """Fan-out: create a Send() for each document to process them in parallel.
    
    Each Send targets the 'process_document' node with a DocumentInput
    containing a single document. LangGraph runs all sends concurrently.
    Results are merged back via operator.add reducers on the FinancialState.
    """
    documents = state["documents"]
    company_cif = state.get("company_cif")
    company_name = state.get("company_name")

    return [
        Send("process_document", {
            "doc_b64": doc,
            "doc_index": i,
            "total_docs": len(documents),
            "company_cif": company_cif,
            "company_name": company_name,
        })
        for i, doc in enumerate(documents)
    ]


def route_by_source(state: FinancialState) -> str:
    """Route to mobile or web aggregation based on source flag."""
    if state.get("source") == "web":
        return "aggregate_web_expenses"
    return "generate_report"


def create_graph():
    """Create the financial report graph with parallel document processing.
    
    After processing, routes to either:
    - generate_report (mobile flow)
    - aggregate_web_expenses (web flow)
    """
    graph_builder = StateGraph(FinancialState)

    graph_builder.add_node("process_document", process_document)
    graph_builder.add_node("generate_report", generate_report)
    graph_builder.add_node("aggregate_web_expenses", aggregate_web_expenses)

    graph_builder.add_conditional_edges(START, fan_out_documents, ["process_document"])

    graph_builder.add_conditional_edges(
        "process_document",
        route_by_source,
        ["generate_report", "aggregate_web_expenses"]
    )

    graph_builder.add_edge("generate_report", END)
    graph_builder.add_edge("aggregate_web_expenses", END)

    return graph_builder.compile()