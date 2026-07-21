"""Shared kl_agent_builder ReAct agent helper.

Public:
    build_agent(cfg, *, system_prompt, max_turns) -> Callable[[str], str]
    run_agent(agent_fn, task) -> str

Requires kl_agent_builder (pip install -e <local prototype-internal-KL checkout>).
Gracefully raises ImportError when not installed — callers must catch.

Env-override caveat: KL_MODEL/KL_BASE_URL/KL_API_KEY override any dict config.
We clear them before building the client so local-ollama config is deterministic.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Cost/runaway bounds for a single agent run (overridable via cfg["agent"]). The expensive case
# is a wandering run that web-fetches large pages each turn and never finalizes; these caps stop
# it. The happy path (~2 turns, ~4k tokens, <60s) is well within them.
#
# NOTE (measured 2026-06-16): a per-model "strong" budget that RAISED max_turns to 10 for non-qwen
# was tried and REJECTED — it regressed the remote `sota` tier from 4/5 → 3/5 on the real top-5 cards. With
# the finalize contract below, the model finalizes promptly at ~turn 5; extra turns only invite it
# to wander PAST coverage-complete (MAX_WANDERING=8) and accumulate context past max_input_tokens →
# completion_guard_force_synthesized / budget_exceeded → ungrounded salvage. Tight bounds + a clear
# finalize contract beat loose bounds. Both tiers therefore share these caps.
_AGENT_DEFAULTS = {
    "max_turns": 6,
    "max_tool_calls": 6,          # каждый web_search/web_fetch — один вызов; топит контекст
    "max_wall_time_seconds": 120,  # жёсткий потолок по времени на один запуск агента
    "max_input_tokens": 60_000,    # жёсткий потолок по входным токенам (накопление observations)
}


# The kl ReAct runtime parses every reply as a structured action. A bare result object/array (which
# our investigate/discover prompts ask for) is mapped to a final_answer whose `content` is "" —
# because kl reads content from the `content`/`answer` key, absent in a schema object (see
# prototype-internal-KL shared/parsing.py::action_from_mapping). An empty final_answer is rejected
# as DEGENERATE (safety.py::is_substantive) and re-prompted; with max_turns ≤ MAX_DEGENERATE(8) the
# loop never accepts it → exhausts turns → ungrounded salvage prose → our parse fails. This contract
# teaches EVERY agent (investigate + discover) the kl protocol up front so the result lands INSIDE
# `content`. Grounded in kl's own _HARMONY_REPROMPT (runtime/loop/react.py). Generalizable: object OR
# array; not specific to any one task.
_FINALIZE_CONTRACT = (
    "\n\nOUTPUT PROTOCOL (your reply is parsed by a ReAct runtime — follow exactly):\n"
    "- To use a tool, emit exactly ONE JSON object and nothing else:\n"
    '    {"action_type":"tool_call","tool_name":"<tool>","arguments":{...}}\n'
    "- To finish, emit exactly ONE JSON object and nothing else:\n"
    '    {"action_type":"final_answer","content":"<your answer>"}\n'
    "- Whatever JSON the task tells you to \"return\" (a JSON object OR array) MUST be placed, "
    "serialized as a STRING, inside the `content` field of that final_answer action. "
    "Do NOT reply with the bare result object/array by itself: a final_answer with empty `content` "
    "is rejected as degenerate and the run fails.\n"
    "- Example finish (note the JSON is a string inside content):\n"
    '    {"action_type":"final_answer","content":"{\\"verdict\\":\\"ok\\",\\"company_known\\":true}"}'
)


def _snippet_search_tool(timeout_s: float, backend: str, searxng_url: str | None):
    """The agent's WebSearchTool, but with kl's empty-snippet DuckDuckGo HTML scraper swapped for
    schabasch's ddgs adapter (browsing/search.py) which returns rich snippet BODIES.

    Root-cause fix (measured 2026-06-21, card 1496): kl's ``ddg`` backend returns titles+URLs with
    EMPTY snippets, so the agent can't read the company description from search results — it re-searches
    the same query 3-4× and exhausts max_turns without ever finalizing. schabasch's ddgs search already
    returns bodies ("equensWorldline SE is a payment company…"), so one search suffices. We reuse kl's
    tool descriptor + ``_format_results`` and only swap the result-fetch; on an empty ddgs result we
    fall back to kl's original scraper (never worse). Generalizable to every investigate/discover run."""
    from kl_agent_builder.tools.web.search_tool import WebSearchTool  # type: ignore

    from .browsing import search as _search

    class _SnippetWebSearchTool(WebSearchTool):  # type: ignore[misc]
        def _search_ddg(self, query, max_results):  # noqa: ANN001
            hits = _search.search(query, max_results=max_results, searxng_url=searxng_url)
            if hits:
                return self._format_results(query, hits, backend="ddg")
            return super()._search_ddg(query, max_results)  # ddgs empty → kl's scraper

    return _SnippetWebSearchTool(timeout_seconds=timeout_s, backend=backend)


def build_agent(cfg: dict, *, system_prompt: str, max_turns: int | None = None) -> Callable[[str], str]:
    """Build and return a ReAct agent callable (task: str) -> str.

    Bounded by cfg["agent"] (max_turns / max_tool_calls / max_wall_time_seconds / max_input_tokens)
    so a non-finalizing run can't blow up time/tokens. Raises ImportError when kl is not installed.
    """
    from kl_agent_builder import react_agent  # type: ignore
    from kl_agent_builder.llm import LLMClient  # type: ignore
    from kl_agent_builder.shared.sandbox import CodeExecutor  # type: ignore
    from kl_agent_builder.tools.web.fetch_tool import WebFetchTool  # type: ignore

    # Clear env overrides so the dict config below is authoritative
    for var in ("KL_MODEL", "KL_BASE_URL", "KL_API_KEY"):
        os.environ.pop(var, None)

    a_cfg = {**_AGENT_DEFAULTS, **(cfg.get("agent") or {})}
    if max_turns is not None:
        a_cfg["max_turns"] = max_turns

    # KEYLESS, config-driven search backend for the agent's WebSearchTool. kl supports searx/ddg/tavily;
    # we NEVER default to tavily (the user supplies no API keys). A self-hosted SearXNG (browsing.
    # searxng_url) wins when set (multi-engine, robust); otherwise keyless DuckDuckGo. Deterministic
    # (explicit backend) so a stray TAVILY_API_KEY in the env can't silently select a keyed backend.
    b_cfg = cfg.get("browsing") or {}
    searxng_url = b_cfg.get("searxng_url")
    if searxng_url:
        os.environ["SEARXNG_URL"] = str(searxng_url)
    search_backend = b_cfg.get("search_backend") or ("searx" if searxng_url else "ddg")

    # Role-routed model (llm.roles.agent) — defaults to local ollama qwen3:8b, but can be pointed at
    # the local 35B MLX server or a remote OpenAI-compatible `sota` tier to fix qwen3:8b's max-turn exhaustion on ReAct.
    from .llm_clients import agent_client_params
    params = agent_client_params(cfg)
    agent_model = params["model"]
    client = LLMClient.from_config({**params, "temperature": 0})

    workdir = Path(cfg.get("paths", {}).get("agent_workdir", "/tmp/schabasch_agent"))
    workdir.mkdir(parents=True, exist_ok=True)
    executor = CodeExecutor(workdir)   # kl's CodeExecutor expects a Path (calls .resolve())

    # Teach EVERY agent the kl final_answer protocol up front (root-cause fix for strong models
    # returning a bare result object → empty-content degenerate → max-turn exhaustion). Appended
    # after the task-specific prompt so it has the last word on output format.
    system_prompt = system_prompt + _FINALIZE_CONTRACT

    # qwen3 is a THINKING model: by default it wraps every turn in <think>…</think>, which kl's
    # ReAct action-parser can't read → the loop never registers an action and exhausts max_turns
    # ('[max turns exhausted]'). '/no_think' disables thinking so the agent emits parseable actions
    # and actually reaches a final_answer. This is qwen-SPECIFIC — a stronger non-qwen model (the 35B
    # MLX / remote sota) must NOT get the directive (it would leak into the prompt as garbage).
    # '/no_think' MUST stay the first token, so it is prepended AFTER the contract is appended.
    if "qwen" in agent_model.lower():
        system_prompt = "/no_think\n" + system_prompt

    agent = react_agent(
        client=client,
        executor=executor,
        system_prompt=system_prompt,
        tools=[
            # ddg backend uses schabasch's snippet-returning ddgs adapter (kl's scraper returns empty
            # snippets → the agent loops re-searching; see _snippet_search_tool). searx unchanged.
            _snippet_search_tool(15.0, search_backend, b_cfg.get("searxng_url")),
            WebFetchTool(timeout_seconds=20.0),
        ],
        max_turns=int(a_cfg["max_turns"]),
        max_tool_calls=int(a_cfg["max_tool_calls"]),
        max_wall_time_seconds=float(a_cfg["max_wall_time_seconds"]),
        max_input_tokens=int(a_cfg["max_input_tokens"]),
    )

    def _run(task: str, context: dict | None = None) -> str:
        from kl_agent_builder import AgentInput  # type: ignore
        # `context` rides in AgentInput.context (rendered into the prompt) — NOT in `task`. kl's
        # facet-coverage gate scans ONLY `task` (runtime/loop/react.py _check_facet_coverage), so any
        # free-text whose incidental words ('current'/'today'/a year) would otherwise be mis-extracted
        # as a required research facet — and silently BLOCK an already-valid final_answer until
        # max_turns — must be passed as context, not inlined into the task. See investigate.py.
        output = agent(AgentInput(task=task, context=context or {}))
        # telemetry: surface token cost + why the run stopped (cost-telemetry pattern)
        md = getattr(output, "metadata", None) or {}
        usage = (md.get("token_usage_by_model") or {}) if isinstance(md, dict) else {}
        total = sum(int(u.get("total_tokens", 0) or 0) for u in usage.values() if isinstance(u, dict))
        _run.last_usage = {  # type: ignore[attr-defined]
            "total_tokens": total,
            "turns": md.get("turns") if isinstance(md, dict) else None,
            "stopped_reason": md.get("stopped_reason") if isinstance(md, dict) else None,
            "wall_seconds": md.get("wall_seconds") if isinstance(md, dict) else None,
        }
        logger.info("agent run: %s", _run.last_usage)  # type: ignore[attr-defined]
        return output.content

    _run.last_usage = {}  # type: ignore[attr-defined]
    return _run


def run_agent(agent_fn: Callable[..., str], task: str, context: dict | None = None) -> str:
    """Invoke a built agent and return its final content string. `context` (optional) is passed to
    AgentInput.context — use it for free-text the model should see but that must stay OUT of the kl
    facet-coverage scan (which reads only `task`); see build_agent._run."""
    return agent_fn(task, context)


def parse_json_output(text: str) -> Any:
    """Parse agent output robustly: strip qwen3 <think> reasoning, markdown fences, and surrounding
    prose, then JSON-parse — falling back to ast.literal_eval for the single-quoted (Python-dict)
    style qwen3 frequently emits. Raises a clear ValueError on any kl non-finalize sentinel
    ('[max turns exhausted]', '[budget exceeded]', the salvage preamble, …) so a run that never
    finalized is classified honestly instead of crashing in ast.literal_eval."""
    import ast
    text = (text or "").strip()
    # The kl ReAct loop emits one of several sentinels / a salvage preamble when it never produced a
    # clean final_answer. Recognize them ALL and raise a clear "did not finalize" error — otherwise a
    # bracketed sentinel like '[budget exceeded]' reaches ast.literal_eval and surfaces as a confusing
    # 'invalid syntax / perhaps you forgot a comma' SyntaxError that hides the real cause.
    low = text.lower()
    _NONFINAL = ("[max turns exhausted]", "max turns exhausted", "[budget exceeded]",
                 "[no answer]", "[unparseable response]")
    if low in _NONFINAL or low.startswith("partial answer (synthesized from tool records"):
        raise ValueError(f"agent did not finalize ({text[:60]})")
    # 1) drop qwen3 <think>...</think> reasoning blocks (incl. an unterminated trailing one)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 2) strip ```json ... ``` / ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text).strip()

    def _parse(s: str) -> Any:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return ast.literal_eval(s)   # tolerate single-quoted Python-dict output (safe: literals only)

    try:
        return _parse(text)
    except (ValueError, SyntaxError):
        # 3) extract the first balanced JSON array/object embedded in prose
        m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        if m:
            return _parse(m.group(1))
        raise
