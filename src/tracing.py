"""Langfuse tracing, optional: enabled only when the three LANGFUSE_* env vars are set.
Only masked text reaches the tracer; the Presidio input guard runs first."""


class NoopTracer:
    def trace_request(self, result: dict):
        pass

    def attach_feedback(self, request_id: str, helpful: bool, comment: str | None):
        pass


class LangfuseTracer:
    def __init__(self, cfg: dict):
        from langfuse import Langfuse
        env = cfg["env"]
        self.client = Langfuse(public_key=env["langfuse_public_key"],
                               secret_key=env["langfuse_secret_key"],
                               host=env["langfuse_host"])

    def trace_request(self, result: dict):
        tr = result["trace"]
        # Only masked text may reach Langfuse. On refusal/abstention/guard-failure
        # paths the answer is a fixed sentence with no user content, so it is
        # safe as a fallback when no masked_answer was produced.
        masked = tr.get("masked_answer", result["answer"])
        t = self.client.trace(id=tr["request_id"], name="ask",
                              input={"masked_question": tr["masked_question"]},
                              output={"answer": masked[:500],
                                      "escalate": result["escalate"]},
                              metadata={"prompt_version": tr["prompt_version"],
                                        "model": tr["model"],
                                        "guards": tr["guards"],
                                        "context_chunk_ids": tr.get("context_chunk_ids"),
                                        "usage": tr.get("usage")})
        for stage, seconds in tr["stages"].items():
            t.span(name=stage, metadata={"seconds": seconds})

    def attach_feedback(self, request_id: str, helpful: bool, comment: str | None):
        self.client.score(trace_id=request_id, name="user_feedback",
                          value=1 if helpful else 0, comment=comment)


def make_tracer(cfg: dict):
    env = cfg["env"]
    if env["langfuse_public_key"] and env["langfuse_secret_key"] and env["langfuse_host"]:
        return LangfuseTracer(cfg)
    return NoopTracer()
