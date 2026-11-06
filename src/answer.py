"""End-to-end answer pipeline: input guards -> retrieval -> reranking ->
generation -> output guards -> answer, or fallback with escalation.

Components are injected so the API can run against the real stack while tests
run against stubs, without any code path differences in the orchestration."""
import time
import uuid

from .config import load_prompt
from .guards import INJECTION_REFUSAL, injection_screen, parse_citations

GROUND_SYSTEM = (
    "You verify whether an answer is supported by source passages.\n"
    "Reply with exactly one word: SUPPORTED if every factual claim in the answer "
    "is backed by the passages, or UNSUPPORTED if any claim is not.")


def format_passages(context):
    return "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context))


class SupportPipeline:
    def __init__(self, cfg: dict, retriever, reranker, generator, masker):
        self.cfg = cfg
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.masker = masker
        self.prompt_version = "answer_v1"
        self.system_prompt, self.user_template = load_prompt(self.prompt_version)
        self.abstain = cfg["guards"]["abstain_sentence"]
        self.grounding_enabled = cfg["guards"]["grounding_check"]
        self.injection_enabled = cfg["guards"]["injection_screen"]

    # -- output-guard helpers -------------------------------------------------
    def _grounding_check(self, context, answer):
        user = (f"Passages:\n{format_passages(context)}\n\n"
                f"Answer to verify:\n{answer}\n\nOne word:")
        out = self.generator.generate(GROUND_SYSTEM, user, max_tokens=5)
        verdict = out["text"].strip().upper()
        return verdict.startswith("SUPPORTED"), verdict

    def _fallback_links(self, context):
        links = []
        for c in context[:3]:
            links.append({"title": c["title"] or f"Support topic: {c['intent']}",
                          "url": c.get("url"), "doc_id": c["doc_id"]})
        return links

    # -- the pipeline ---------------------------------------------------------
    def answer(self, question: str) -> dict:
        request_id = uuid.uuid4().hex[:16]
        trace = {"request_id": request_id, "prompt_version": self.prompt_version,
                 "model": self.generator.model, "stages": {}, "guards": {}}
        t_all = time.time()

        # input guards (before retrieval; only masked text may appear in logs)
        t0 = time.time()
        masked_q, pii_ents = self.masker.mask(question)
        trace["masked_question"] = masked_q
        trace["guards"]["input_pii_entities"] = pii_ents
        inj, pattern = injection_screen(question) if self.injection_enabled else (False, None)
        trace["guards"]["injection"] = {"hit": inj, "pattern": pattern}
        trace["stages"]["input_guards_s"] = round(time.time() - t0, 3)
        if inj:
            trace["stages"]["total_s"] = round(time.time() - t_all, 2)
            return {"request_id": request_id, "answer": INJECTION_REFUSAL,
                    "citations": [], "escalate": False,
                    "guard_results": {"refused": "injection"}, "trace": trace}

        # retrieve + rerank
        t0 = time.time()
        candidates = self.retriever.retrieve(question)
        full, context = self.reranker.rerank(question, candidates)
        trace["stages"]["retrieve_rerank_s"] = round(time.time() - t0, 2)
        trace["context_chunk_ids"] = [c["chunk_id"] for c in context]
        trace["context_rerank_scores"] = [round(c["rerank_score"], 3) for c in context]

        # generate
        gen = self.generator.generate(
            self.system_prompt,
            self.user_template.format(passages=format_passages(context),
                                      question=question))
        draft = gen["text"].strip()
        trace["stages"]["generate_s"] = gen["gen_seconds"]
        trace["usage"] = gen["usage"]

        # abstention
        if self.abstain.lower()[:40] in draft.lower():
            trace["guards"]["abstained"] = True
            trace["stages"]["total_s"] = round(time.time() - t_all, 2)
            return {"request_id": request_id, "answer": self.abstain,
                    "citations": [], "escalate": True,
                    "fallback_links": self._fallback_links(context),
                    "guard_results": {"abstained": True}, "trace": trace}

        # output guards
        t0 = time.time()
        valid_cites, invalid_cites = parse_citations(draft, len(context))
        if self.grounding_enabled:
            grounded, verdict = self._grounding_check(context, draft)
        else:
            grounded, verdict = True, "SKIPPED"
        out_masked, out_pii = self.masker.mask(draft)
        trace["masked_answer"] = out_masked        # tracing may only see masked text
        trace["guards"]["citations"] = {"valid": valid_cites, "invalid": invalid_cites}
        trace["guards"]["grounding"] = verdict
        trace["guards"]["output_pii_entities"] = out_pii
        trace["stages"]["output_guards_s"] = round(time.time() - t0, 2)
        trace["stages"]["total_s"] = round(time.time() - t_all, 2)

        if (not grounded) or (not valid_cites) or invalid_cites:
            return {"request_id": request_id, "answer": self.abstain,
                    "citations": [], "escalate": True,
                    "fallback_links": self._fallback_links(context),
                    "guard_results": {"grounded": grounded,
                                      "citations_ok": bool(valid_cites),
                                      "invalid_citations": invalid_cites,
                                      "failed": True},
                    "trace": trace}

        citations = [{"n": n, "chunk_id": context[n-1]["chunk_id"],
                      "doc_id": context[n-1]["doc_id"],
                      "title": context[n-1]["title"] or context[n-1]["intent"],
                      "url": context[n-1].get("url")}
                     for n in valid_cites]
        return {"request_id": request_id, "answer": draft, "citations": citations,
                "escalate": False,
                "guard_results": {"grounded": True, "citations_ok": True},
                "trace": trace}
