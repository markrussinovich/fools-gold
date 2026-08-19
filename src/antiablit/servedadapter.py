"""Generation backend against a served OpenAI-compatible endpoint (vLLM).

Drop-in for the ModelAdapter.generate contract on lines whose model cannot be
loaded in-process (397B: FP8 TP=8 vLLM serve is the validated pattern). Same
signature and semantics: chat template applied server-side with chat_kwargs
(enable_thinking) passed through, temperature=None -> greedy, per-request
seeds derived from a fixed base so shards/reruns reproduce. Sampling RNG
differs from the HF in-process path — never mix backends within one
comparison (registered caveat, 2026-07-26); a served line is internally
consistent.

No tokenizer/weights are exposed: capture_layer_states and weight edits are
meaningless here and raise. Attack state is a pre-materialized checkpoint
served under its own model name (e.g. M0a-FP8), selected via model_cfg.
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class ServedAdapter:
    def __init__(self, model_cfg: dict, device: str = "unused"):
        self.cfg = model_cfg
        self.url = model_cfg["served_url"].rstrip("/")
        self.model = model_cfg["served_model"]
        self.chat_kwargs = dict(model_cfg.get("chat_kwargs", {}))
        # dsv4 encoding seam (adversarial review 2026-08-05 finding 1): the
        # vLLM deepseek_v4 tokenizer shim and reasoning parser read ONLY the
        # thinking/enable_thinking booleans and silently DROP a thinking_mode
        # key — without this translation the registered channel mode held
        # only by server-default coincidence. Renders byte-identically today
        # (chat == default); makes the mode transmitted, asserted state.
        if ("thinking_mode" in self.chat_kwargs
                and not any(k in self.chat_kwargs
                            for k in ("thinking", "enable_thinking"))):
            self.chat_kwargs["thinking"] = \
                self.chat_kwargs["thinking_mode"] == "thinking"
        # mistral served-path seam (ms4-b0 funny_rabbit 400, 2026-08-09):
        # vLLM 0.26's mistral tokenizer mode REJECTS any request that carries
        # chat_template_kwargs — even {} (tokenizers/mistral.py
        # validate_request_params: `chat_template_kwargs is not None`) — and
        # instead accepts reasoning_effort as a TOP-LEVEL request param
        # (protocol field, threaded into the render; enum none|high). Map the
        # registered chat_kwargs identity onto the transport: reasoning_effort
        # rides top-level; chat_template_kwargs is attached only when
        # non-empty ({} == absent is a no-op on the HF template path, which
        # merges default kwargs). Same pattern as the thinking_mode
        # translation above: the mode becomes transmitted, asserted state.
        self.extra_body = {}
        if "reasoning_effort" in self.chat_kwargs:
            self.extra_body["reasoning_effort"] = \
                self.chat_kwargs.pop("reasoning_effort")
        self.seed_base = int(model_cfg.get("seed_base", 1234))
        self.parallel = int(model_cfg.get("served_parallel", 16))
        self.timeout = int(model_cfg.get("served_timeout", 1800))
        # closed-CoT seam (registered 2026-08-12, oss120 champ-followon
        # pre-submit F6; spec section 8 item 5): a non-empty gen_prefix
        # switches generation to the /v1/completions endpoint with
        # CLIENT-SIDE chat render + forced-prefix prompt surgery — the exact
        # ModelAdapter gen_prefix identity (render with
        # add_generation_prompt + chat_kwargs, append the prefix verbatim,
        # final-channel extraction when the prefix ends in the opener).
        # ABSENT/EMPTY KEY = byte-identical chat-completions behavior for
        # every other line (unit-tested both flavors).
        self.gen_prefix = str(model_cfg.get("gen_prefix") or "")
        self._tok = None

    def wait_ready(self, budget_s: int = 3600):
        t0 = time.time()
        while time.time() - t0 < budget_s:
            try:
                urllib.request.urlopen(f"{self.url}/health", timeout=10)
                return
            except Exception:
                time.sleep(15)
        raise TimeoutError(f"server at {self.url} not healthy in {budget_s}s")

    def _render(self, prompt: str) -> str:
        # mirrors ModelAdapter's gen_prefix path (modeladapter.py: chat
        # render with add_generation_prompt=True + chat_kwargs, then the
        # prefix appended verbatim)
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(
                self.cfg.get("tokenizer_id") or self.cfg["hf_id"])
        return self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, **self.chat_kwargs) + self.gen_prefix

    def _one_completions(self, prompt: str, max_new_tokens: int, temperature, seed):
        # gen_prefix path: raw /v1/completions on the surgically-rendered
        # prompt (server-side chat rendering would silently DROP the prefix
        # — the original 2026-08-03 review finding this seam resolves)
        body = {"model": self.model,
                "prompt": self._render(prompt),
                "max_tokens": max_new_tokens,
                "seed": seed}
        if temperature:
            body.update(temperature=temperature, top_p=0.95)
        else:
            body["temperature"] = 0.0
        req = urllib.request.Request(
            f"{self.url}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(20 * (attempt + 1))
        ch = resp["choices"][0]
        text = ch.get("text") or ""
        # extraction identity with ModelAdapter's gen_prefix path: a prefix
        # ending in the final-channel opener means the completion IS
        # final-channel text — cut at the next channel marker
        from .modeladapter import FINAL_CHANNEL, harmony_final
        if self.gen_prefix.endswith(FINAL_CHANNEL):
            # harmony_final returns (content, no_final); no_final is
            # impossible here — the forced prefix supplies the final opener
            text, _ = harmony_final(FINAL_CHANNEL + text)
        usage = resp.get("usage") or {}
        return {"text": text,
                "reasoning": None,
                "finish_reason": ch.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens")}

    def _one_full(self, prompt: str, max_new_tokens: int, temperature, seed):
        if self.gen_prefix:
            return self._one_completions(prompt, max_new_tokens, temperature, seed)
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "seed": seed,
                **self.extra_body}
        if self.chat_kwargs:
            body["chat_template_kwargs"] = self.chat_kwargs
        if temperature:
            body.update(temperature=temperature, top_p=0.95)
        else:
            body["temperature"] = 0.0
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(20 * (attempt + 1))
        ch = resp["choices"][0]
        m = ch["message"]
        # reasoning channel: vLLM 0.26 renamed the response field
        # reasoning_content -> reasoning (the old name survives only as a
        # REQUEST-message alias) — read both so either server version works.
        # Field-name provenance: dsv4 think_on AssertionError root cause,
        # 2026-08-05 (repro job sad_fish).
        reasoning = m.get("reasoning")
        if reasoning is None:
            reasoning = m.get("reasoning_content")
        usage = resp.get("usage") or {}
        return {"text": m.get("content") or "",
                "reasoning": reasoning,
                "finish_reason": ch.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens")}

    def _one(self, prompt: str, max_new_tokens: int, temperature, seed):
        return self._one_full(prompt, max_new_tokens, temperature, seed)["text"]

    def generate(self, prompts: list[str], max_new_tokens: int = 96,
                 batch_size: int = 32, temperature: float | None = None) -> list[str]:
        # batch_size is an HF memory knob; here it just caps request concurrency
        workers = min(self.parallel, max(batch_size, 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(
                lambda ip: self._one(ip[1], max_new_tokens, temperature,
                                     self.seed_base + ip[0]),
                enumerate(prompts)))

    def generate_full(self, prompts: list[str], max_new_tokens: int = 96,
                      batch_size: int = 32,
                      temperature: float | None = None) -> list[dict]:
        """generate() with per-request metadata rows: {text, reasoning,
        finish_reason, completion_tokens}. Same sampling identity (per-request
        seeds seed_base + global index). Callers own content hygiene for the
        returned text/reasoning fields — never log them. Progress markers
        (counts only) every 256 completions so long waves are stall-visible
        (infra-vigilance: a silent 25-min generation looked identical to a
        wedge)."""
        import threading
        workers = min(self.parallel, max(batch_size, 1))
        n = len(prompts)
        done = [0]
        lock = threading.Lock()

        def one(ip):
            r = self._one_full(ip[1], max_new_tokens, temperature,
                               self.seed_base + ip[0])
            with lock:
                done[0] += 1
                if done[0] % 256 == 0 or done[0] == n:
                    print(f"[served] {done[0]}/{n} generations", flush=True)
            return r

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, enumerate(prompts)))

    def __getattr__(self, name):
        raise AttributeError(
            f"ServedAdapter has no '{name}': weight/activation access is not "
            f"available on a served backend (pre-materialize instead)")


def make_adapter(line_cfg: dict, model_cfg: dict, device: str = "cuda:0"):
    """Backend factory: line config selects in-process HF or served vLLM."""
    if line_cfg.get("backend") == "served":
        merged = dict(line_cfg.get("served_defaults", {}), **model_cfg)
        # closed-CoT gen_prefix is IMPLEMENTED on the served backend since
        # 2026-08-12 (oss120 champ-followon F6): completions-endpoint prompt
        # surgery inside ServedAdapter — see ServedAdapter.__init__/_render.
        # An empty/absent gen_prefix keeps the chat-completions path
        # byte-identical (the pre-seam behavior).
        return ServedAdapter(merged, device)
    from .modeladapter import ModelAdapter
    return ModelAdapter(model_cfg, device)
