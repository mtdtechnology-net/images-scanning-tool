from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from core.state import FinancialState
from graph.nodes import process_document, generate_report


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


def create_graph():
    """Create the financial report graph with parallel document processing.
    """
    graph_builder = StateGraph(FinancialState)

    graph_builder.add_node("process_document", process_document)
    graph_builder.add_node("generate_report", generate_report)

    graph_builder.add_conditional_edges(START, fan_out_documents, ["process_document"])

    graph_builder.add_edge("process_document", "generate_report")

    graph_builder.add_edge("generate_report", END)

    return graph_builder.compile()