# TP5/agent/nodes/finalize.py
import re
from typing import List

from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

RE_CIT = re.compile(r"\[(doc_\d+)\]")

def _extract_citations(text: str) -> List[str]:
    return sorted(set(RE_CIT.findall(text or "")))

def finalize(state: AgentState) -> AgentState:


    log_event(state.run_id, "node_start", {"node": "finalize"})

    if not state.budget.can_step():
        log_event(state.run_id, "node_end", {"node": "finalize", "status": "budget_exceeded"})
        return state
    state.budget.steps_used += 1


    #log_event(state.run_id, "node_start", {"node": "finalize"})

    intent = state.decision.intent

    if intent == "reply":
        cits = _extract_citations(state.draft_v1)
        state.final_kind = "reply"
        if cits:
            state.final_text = state.draft_v1.strip() + "\n\nSources: " + " ".join(f"[{c}]" for c in cits)
        else:
            state.final_text = state.draft_v1.strip() or "Merci pour votre email . Réponse en cours de finalisation."  # TODO: fallback reply

    elif intent == "ask_clarification":
        state.final_kind = "clarification"
        state.final_text = state.draft_v1.strip() or "Pouvez-vous préciser votre demande afin que je puisse vous répondre ?"  # TODO: fallback questions

    elif intent == "escalate":
        state.final_kind = "handoff"
        # TODO: action mockée (packet)
        state.actions.append({
            "type": "handoff_packet",
            "run_id": state.run_id,
            "email_id": state.email_id,
            "summary": state.decision.rationale,
            "evidence_ids": [d.doc_id for d in state.evidence],
        })
        state.final_text = "Votre demande nécessite une validation humaine. Je transmets avec un résumé et les sources."

    else:
        state.final_kind = "ignore"
        state.final_text = "Aucune action requise pour ce message.Votre demande nécessite une validation humaine. Je transmets avec un résumé et les sources."  # TODO: texte minimal ignore

    log_event(state.run_id, "node_end", {"node": "finalize", "status": "ok", "final_kind": state.final_kind})
    return state