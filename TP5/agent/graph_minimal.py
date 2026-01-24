# TP5/agent/graph_minimal.py
from langgraph.graph import StateGraph, END

from TP5.agent.state import AgentState
from TP5.agent.routing import route

from TP5.agent.nodes.classify_email import classify_email
from TP5.agent.nodes.maybe_retrieve import maybe_retrieve
from TP5.agent.nodes.draft_reply import draft_reply
from TP5.agent.nodes.check_evidence import check_evidence
from TP5.agent.nodes.rewrite_query import rewrite_query
from TP5.agent.nodes.finalize import finalize

from TP5.agent.nodes.stubs import (
    stub_ask_clarification,
    stub_escalate,
    stub_ignore,
)


def build_graph():
    g = StateGraph(AgentState)

    # ======================
    # Nodes
    # ======================
    g.add_node("classify_email", classify_email)

    # Reply path (avec RAG + boucle)
    g.add_node("maybe_retrieve", maybe_retrieve)
    g.add_node("reply", draft_reply)
    g.add_node("check_evidence", check_evidence)
    g.add_node("rewrite_query", rewrite_query)

    # Autres paths (stubs)
    g.add_node("ask_clarification", stub_ask_clarification)
    g.add_node("escalate", stub_escalate)
    g.add_node("ignore", stub_ignore)
    g.add_node("finalize", finalize)

    # ======================
    # Entry point
    # ======================
    g.set_entry_point("classify_email")

    # ======================
    # Routing après classification
    # ======================
    g.add_conditional_edges(
        "classify_email",
        route,
        {
            "reply": "maybe_retrieve",
            "ask_clarification": "ask_clarification",
            "escalate": "escalate",
            "ignore": "ignore",
        },
    )

    # ======================
    # Branche reply avec boucle contrôlée
    # ======================
    g.add_edge("maybe_retrieve", "reply")
    g.add_edge("reply", "check_evidence")
    # g.add_edge("reply", END)
    g.add_edge("ask_clarification", "finalize")
    g.add_edge("escalate", "finalize")
    g.add_edge("ignore", "finalize")

    g.add_edge("finalize", END)

    def after_check(state: AgentState) -> str:
        if state.evidence_ok:
            return "end"
        if state.budget.retrieval_attempts == 0:
            return "end"
        if state.budget.retrieval_attempts < state.budget.max_retrieval_attempts:
            return "rewrite"
        return "end"




    g.add_conditional_edges("check_evidence", after_check, {
    "end": "finalize",
    "rewrite": "rewrite_query",
    })
    g.add_edge("rewrite_query", "maybe_retrieve")

    # ======================
    # Autres branches → END
    # ======================
    #g.add_edge("ask_clarification", END)
    #g.add_edge("escalate", END)
    #g.add_edge("ignore", END)

    return g.compile()
