# scoring.py (async-safe: TLM calls run in a thread, robust fallbacks)
import os
import asyncio
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import CrossEncoder
from cleanlab_tlm.tlm import TLM
from lit_search import search_papers_combined

NLI_MODEL = None

async def load_nli_model():
    global NLI_MODEL
    if NLI_MODEL is None:
        # downloads once; CPU is fine
        NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    return NLI_MODEL

async def literature_agreement(answer: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """NLI-based agreement with literature abstracts."""
    if not papers:
        return {"s_lit": None, "evidence": []}

    model = await load_nli_model()
    pairs = [(p.get("abstract", "")[:2500], answer) for p in papers]
    logits = model.predict(pairs)  # (n, 3) order: [contradiction, entailment, neutral]
    # softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = exp / exp.sum(axis=1, keepdims=True)
    p_contra, p_entail = prob[:, 0], prob[:, 1]

    s_lit = float(np.clip(p_entail.mean() * (1.0 - p_contra.mean()), 0.0, 1.0))

    evidence = []
    for i, p in enumerate(papers):
        tag = "Support" if (p_entail[i] > 0.5 and p_contra[i] < 0.4) else ("Contradict" if p_contra[i] > 0.5 else "Uncertain")
        evidence.append({
            "title": p.get("title"),
            "doi": p.get("doi"),
            "url": p.get("url"),
            "venue": p.get("venue"),
            "year": p.get("year"),
            "source": p.get("source"),
            "citations": p.get("citations"),
            "p_entail": round(float(p_entail[i]), 3),
            "p_contra": round(float(p_contra[i]), 3),
            "verdict": tag,
        })
    return {"s_lit": s_lit, "evidence": evidence}

async def cleanlab_trust(question: str, answer: str) -> Dict[str, Any]:
    """Call Cleanlab TLM in a background thread to avoid 'event loop is already running'."""
    if not os.getenv("CLEANLAB_TLM_API_KEY"):
        return {"s_tlm": None, "explain": "TLM scoring is disabled (no CLEANLAB_TLM_API_KEY set)."}

    def _call_sync() -> Dict[str, Any]:
        tlm = TLM(quality_preset="medium", options={"log": ["explanation"]})
        return tlm.get_trustworthiness_score(prompt=question, response=answer)

    try:
        s_obj = await asyncio.to_thread(_call_sync)
        score = float(s_obj.get("trustworthiness_score") or 0.0)
        explain = (s_obj.get("log") or {}).get("explanation")
        return {"s_tlm": score, "explain": explain}
    except Exception as e:
        # Never break the page due to TLM errors
        print(f"[TLM get_trustworthiness_score] soft-fail: {e!r}")
        return {"s_tlm": None, "explain": "TLM temporarily unavailable."}

async def summarize_experts(question: str, papers: List[Dict[str, Any]]) -> str:
    """Summarize expert conclusions; uses TLM when available, otherwise a simple fallback list."""
    api_key = os.getenv("CLEANLAB_TLM_API_KEY")
    top = papers[:6]

    if api_key and top:
        context = "\n\n".join(
            [f"[{i+1}] {p['title']} ({p.get('venue','')}, {p.get('year','')})\nAbstract: {p.get('abstract','')[:1200]}"
             for i, p in enumerate(top)]
        )
        prompt = (
            "You are a systematic-review assistant. Given a user question and several paper titles+abstracts, "
            "output 3–6 bullet points of expert conclusions (in English). Be faithful to the evidence, "
            "mark each point with the source number like [1], and note any contradictions or uncertainty.\n\n"
            f"Question: {question}\n\nPapers:\n{context}\n\n"
            "Output format:\n- Conclusion A ... [#]\n- Conclusion B ... [#]\n"
        )

        def _prompt_sync() -> Dict[str, Any]:
            tlm = TLM(quality_preset="medium")
            return tlm.prompt(prompt)

        try:
            resp = await asyncio.to_thread(_prompt_sync)
            return resp.get("response", "") or "(No summary returned.)"
        except Exception as e:
            print(f"[TLM prompt] soft-fail: {e!r}")
            # fall back to simple list
            if not top:
                return "(No related literature found.)"
            return "(TLM disabled) Top papers:\n" + "\n".join([f"- {p['title']} ({p.get('venue','')}, {p.get('year','')})" for p in top])

    if not top:
        return "(No related literature found.)"
    return "(TLM disabled) Top papers:\n" + "\n".join([f"- {p['title']} ({p.get('venue','')}, {p.get('year','')})" for p in top])

async def compute_scores_and_summary(question: str, answer: str) -> Dict[str, Any]:
    # 1) Robust search (never raise)
    try:
        papers = await search_papers_combined((question or "").strip(), limit_total=10)
    except Exception as e:
        print(f"[search] exception swallowed: {e!r}")
        papers = []

    # 2) Run tasks concurrently; each sub-task handles its own errors
    lit_task = asyncio.create_task(literature_agreement(answer, papers))
    tlm_task = asyncio.create_task(cleanlab_trust(question, answer))
    sum_task = asyncio.create_task(summarize_experts(question, papers))
    lit_res, tlm_res, expert_summary = await asyncio.gather(lit_task, tlm_task, sum_task)

    s_lit, s_tlm = lit_res.get("s_lit"), tlm_res.get("s_tlm")

    # 3) Weighted overall score (use what's available)
    weights, comps = [], []
    if s_tlm is not None:
        weights.append(0.6); comps.append(s_tlm)
    if s_lit is not None:
        weights.append(0.4); comps.append(s_lit)
    final_score = None
    if weights:
        w = np.array(weights); c = np.array(comps)
        final_score = float((w * c).sum() / w.sum())

    return {
        "papers": lit_res.get("evidence", []),
        "s_lit": None if s_lit is None else round(s_lit, 3),
        "s_tlm": None if s_tlm is None else round(s_tlm, 3),
        "tlm_explain": tlm_res.get("explain"),
        "final_score": None if final_score is None else round(final_score, 3),
        "expert_summary": expert_summary,
    }