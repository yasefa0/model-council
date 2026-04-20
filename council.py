"""
Model Council — Upgraded build
Architecture: Karpathy-style three-stage council with hybrid routing

Stage 1 — Collect:   All panel models answer in parallel (fan-out)
Stage 2 — Rank:      Panel models cross-rank anonymously (peer jury)
Stage 3 — Synthesize: Chairman (Claude Opus, direct Anthropic) writes final answer

Key upgrades from v1:
 - Hybrid routing: Anthropic SDK direct for chairman, OpenRouter for panel
 - Panel now includes Grok (eliminates same-family bias under an Opus chairman)
 - asyncio.gather(return_exceptions=True) + quorum check (no silent total failures)
 - stamina retry on rate-limit / timeout errors only
 - FastMCP 3.x @mcp.tool(timeout=...) with Context progress notifications (optional dep)
 - typer + rich + prompt_toolkit CLI with live stage display
 - Dual-mode dispatch: CLI or MCP stdio/HTTP server
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import stamina
import typer
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError, APITimeoutError
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

load_dotenv()

# Never write to stdout from this module — it corrupts MCP JSON-RPC framing.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
# Edit these to swap models. Any OpenRouter model works for the panel.
# Browse panel options at: https://openrouter.ai/models
# The chairman always runs via Anthropic direct (for reliable prompt caching).

PANEL_MODELS: list[str] = [
    "openai/gpt-5.4",                # GPT family — 1M ctx
    "google/gemini-3.1-pro-preview", # Gemini family — 1M ctx, reasoning + multimodal
    "x-ai/grok-4.20",                # Grok family — 2M ctx, low-hallucination
]

CHAIRMAN_MODEL = "claude-opus-4-7"  # Direct Anthropic — Opus 4.7 (released April 16, 2026)

MODEL_LABELS: dict[str, str] = {
    "openai/gpt-5.4":                "GPT-5.4",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "x-ai/grok-4.20":                "Grok 4.20",
    "claude-opus-4-7":               "Chairman · Opus 4.7",
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MIN_QUORUM = 2        # Min panelists that must succeed before proceeding
STAGE_TIMEOUT = 180   # Per-stage asyncio timeout in seconds

# Verbalized Sampling block (to be appended to queries when --vs flag is used)
VERBALIZED_SAMPLING_BLOCK = """
Before giving your final response:
1. Generate 5 distinct responses to this question and assign a probability score (0.0–1.0) to each — how likely you'd normally give that response.
2. Identify the response with the lowest probability score (the one you'd normally suppress).
3. Your actual response should BE that lowest-probability response.

Format:
<vs_candidates>
[1] (p=0.XX) ...
[2] (p=0.XX) ...
[3] (p=0.XX) ...
[4] (p=0.XX) ...
[5] (p=0.XX) ...
</vs_candidates>

<vs_selected p=0.XX>
[your lowest-probability response here]
</vs_selected>
"""

# Path to lenses directory
LENSES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lenses")

# ── Clients ────────────────────────────────────────────────────────────────────

def load_lens_template(lens_name: str) -> Optional[str]:
    """Load a lens template from the lenses directory."""
    if not lens_name:
        return None

    # Direct path lookup — faster than glob + iteration
    file_path = os.path.join(LENSES_DIR, f"{lens_name}.md")
    if not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Strip YAML front matter (--- ... ---) if present; return only markdown body
        if content.startswith("---"):
            _, _, markdown_content = content.split("---", 2)
            return markdown_content.strip()
        return content
    except Exception as e:
        logging.error(f"Error loading lens template {lens_name}: {e}")
        return None

def _get_httpx_client() -> httpx.AsyncClient:
    """Tuned httpx client per 2026 research (Compass Artifact)."""
    return httpx.AsyncClient(
        http2=True,
        limits=httpx.Limits(max_connections=50),
        timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5),
    )

def _or_client(http_client: httpx.AsyncClient) -> AsyncOpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        typer.echo("ERROR: OPENROUTER_API_KEY not set in .env", err=True)
        raise typer.Exit(1)
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE,
        api_key=key,
        max_retries=0,                              # stamina handles retries
        default_headers={"HTTP-Referer": "http://localhost"},
        http_client=http_client,
    )

def _ant_client(http_client: httpx.AsyncClient) -> AsyncAnthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        typer.echo("ERROR: ANTHROPIC_API_KEY not set in .env", err=True)
        raise typer.Exit(1)
    return AsyncAnthropic(
        api_key=key,
        max_retries=0,
        http_client=http_client,
    )


# ── Retry-wrapped callers ──────────────────────────────────────────────────────
# Only retry on transient errors. Never retry 400/401/403 or context-length errors.

@stamina.retry(on=(OpenAIRateLimitError, APITimeoutError), attempts=3, wait_initial=1.5, wait_max=20.0)
async def _call_openrouter(client: AsyncOpenAI, model: str, messages: list[dict]) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=3000,                            # Further increased for Gemini depth
    )
    choice = resp.choices[0]
    reason = choice.finish_reason
    if reason == "length":
        logging.warning(f"Model {model} truncated output (max_tokens reached).")
    elif reason and reason != "stop":
        logging.warning(f"Model {model} stopped with reason: {reason}")
    
    return choice.message.content.strip()


@stamina.retry(on=(AnthropicRateLimitError,), attempts=3, wait_initial=1.5, wait_max=20.0)
async def _call_anthropic(
    client: AsyncAnthropic,
    model: str,
    messages: list[dict],
    system: str | None = None,
) -> str:
    # Increased max_tokens for MCP usage to prevent truncation in Claude Desktop
    kwargs: dict[str, Any] = dict(model=model, max_tokens=4000, messages=messages)
    if system:
        kwargs["system"] = system
    resp = await client.messages.create(**kwargs)
    return resp.content[0].text.strip()


async def call_model(
    or_client: AsyncOpenAI,
    ant_client: AsyncAnthropic,
    model: str,
    messages: list[dict],
    system: str | None = None,
) -> str:
    """Route to the correct client, catch all exceptions, return error string."""
    try:
        if model.startswith("claude"):
            return await _call_anthropic(ant_client, model, messages, system)
        else:
            if system:
                messages = [{"role": "system", "content": system}, *messages]
            return await _call_openrouter(or_client, model, messages)
    except Exception as exc:
        return f"[ERROR: {exc}]"


# ── Stage 1: Fan-out ───────────────────────────────────────────────────────────

async def stage1_collect(
    or_client: AsyncOpenAI,
    ant_client: AsyncAnthropic,
    query: str,
    use_vs: bool = False,
    lens_template: Optional[str] = None,
) -> dict[str, str]:
    # Prepare base query
    base_query = query
    
    # Add verbalized sampling block if requested
    if use_vs:
        base_query = f"{query}\n\n{VERBALIZED_SAMPLING_BLOCK}"
    
    tasks = [
        call_model(or_client, ant_client, m, [{"role": "user", "content": base_query}], lens_template)
        for m in PANEL_MODELS
    ]
    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=STAGE_TIMEOUT,
    )
    out: dict[str, str] = {}
    for model, result in zip(PANEL_MODELS, results):
        out[model] = f"[ERROR: {result}]" if isinstance(result, Exception) else result

    successes = sum(1 for v in out.values() if not v.startswith("[ERROR"))
    if successes < MIN_QUORUM:
        raise RuntimeError(
            f"Quorum not met: only {successes}/{len(PANEL_MODELS)} panelists succeeded."
        )
    return out


# ── Stage 2: Peer ranking ──────────────────────────────────────────────────────

def extract_vs_parts(response: str) -> Tuple[str, str]:
    """Extract verbalized sampling candidates and selected response from a VS-formatted response."""
    candidates = ""
    selected = ""
    
    # Extract candidates
    candidates_match = re.search(r'<vs_candidates>(.*?)</vs_candidates>', response, re.DOTALL)
    if candidates_match:
        candidates = candidates_match.group(1).strip()
    
    # Extract selected response
    selected_match = re.search(r'<vs_selected p=(\d+\.\d+|\d+)>(.*?)</vs_selected>', response, re.DOTALL)
    if selected_match:
        selected = selected_match.group(2).strip()
    
    return candidates, selected

async def stage2_rank(
    or_client: AsyncOpenAI,
    ant_client: AsyncAnthropic,
    query: str,
    stage1: dict[str, str],
    use_vs: bool = False,
    lens_template: Optional[str] = None,
) -> tuple[dict[str, int], dict[str, str], dict[str, str], list[str], dict[str, tuple[str, str]]]:
    models = list(stage1.keys())
    labels = [chr(65 + i) for i in range(len(models))]   # A, B, C …
    label_to_model = dict(zip(labels, models))
    
    # Store vs_candidates and vs_selected for each model if using verbalized sampling
    vs_parts: dict[str, tuple[str, str]] = {}
    if use_vs:
        for model, response in stage1.items():
            candidates, selected = extract_vs_parts(response)
            vs_parts[model] = (candidates, selected)

    anon_block = "\n\n".join(
        f"Response {lbl}:\n{stage1[m]}"
        for lbl, m in zip(labels, models)
    )
    # Create ranking prompt
    rank_prompt_content = f"""Original question: {query}

Here are {len(models)} responses labeled anonymously:

{anon_block}

Rank these responses from best to worst. Reply ONLY in this exact format:
Ranking: [comma-separated labels, e.g. "B, A, C"]
A: [one sentence on this response]
B: [one sentence]
C: [one sentence]"""

    # Add verbalized sampling if requested
    if use_vs:
        rank_prompt_content = f"{rank_prompt_content}\n\n{VERBALIZED_SAMPLING_BLOCK}"

    tasks = [
        call_model(or_client, ant_client, m, [{"role": "user", "content": rank_prompt_content}], lens_template)
        for m in PANEL_MODELS
    ]
    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=STAGE_TIMEOUT,
    )
    rankings: dict[str, str] = {
        m: (f"[ERROR: {r}]" if isinstance(r, Exception) else r)
        for m, r in zip(PANEL_MODELS, results)
    }

    # Tally: 1st = N pts, 2nd = N-1 pts, …
    scores: dict[str, int] = {m: 0 for m in models}
    for raw in rankings.values():
        if raw.startswith("[ERROR"):
            continue
        try:
            rank_line = next(ln for ln in raw.split("\n") if ln.strip().startswith("Ranking:"))
            ordered = [x.strip().strip("[]\"' ") for x in rank_line.replace("Ranking:", "").split(",")]
            pts = len(ordered)
            for lbl in ordered:
                if lbl in label_to_model:
                    scores[label_to_model[lbl]] += pts
                    pts -= 1
        except Exception:
            logging.warning("Stage 2: could not parse ranking response — skipping.")

    return scores, rankings, label_to_model, labels, vs_parts


# ── Stage 3: Chairman synthesis ────────────────────────────────────────────────

async def stage3_synthesize(
    ant_client: AsyncAnthropic,
    query: str,
    stage1: dict[str, str],
    scores: dict[str, int],
    use_vs: bool = False,
    lens_template: Optional[str] = None,
    vs_parts: Optional[dict[str, tuple[str, str]]] = None,
) -> tuple[str, str]:
    """Synthesize final answer from chairman model based on panel responses and scores.
    Returns (topic_title, body) tuple where topic_title is a 1-4 word summary for the filename."""
    sorted_models = sorted(scores.items(), key=lambda x: -x[1])
    score_summary = "\n".join(
        f"  {MODEL_LABELS.get(m, m)}: {s} pts" for m, s in sorted_models
    )
    answers_block = "\n\n".join(
        f"[{MODEL_LABELS.get(m, m)}]:\n{a}" for m, a in stage1.items()
    )
    base_system = (
        "You are the chairman of an AI model council. Your task is to synthesize "
        "the council's answers into the single best possible response. "
        "Be direct and substantive. Do not hedge excessively."
    )
    
    # Apply lens template to system prompt if provided
    system = base_system
    if lens_template:
        system = f"{base_system}\n\n{lens_template}"

    topic_instruction = "Start your response with a short 1-4 word topic title on its own line prefixed with exactly TOPIC: (e.g., \"TOPIC: Rust Learning Strategy\"). Then leave a blank line before the rest of your response."

    if use_vs and vs_parts:
        # If we have verbalized sampling data, include it in the chairman's input
        vs_candidates_block = "\n\n".join(
            f"[{MODEL_LABELS.get(m, m)}] Candidates:\n{candidates}"
            for m, (candidates, _) in vs_parts.items()
        )
        
        vs_selected_block = "\n\n".join(
            f"[{MODEL_LABELS.get(m, m)}] Selected (lowest probability):\n{selected}"
            for m, (_, selected) in vs_parts.items()
        )
        
        user_msg = f"""Original question: {query}

Council answers:
{answers_block}

Peer ranking scores (higher = rated better by peers):
{score_summary}

Verbalized sampling candidates (probability distributions):
{vs_candidates_block}

Verbalized sampling selected responses (lowest probability):
{vs_selected_block}

Instructions:
1. {topic_instruction}
2. Identify where all models agree — this is the high-confidence ground.
3. Identify where models diverge — surface the disagreement explicitly.
4. Write a final synthesized answer that is better than any individual response.
5. If using verbalized sampling:
   - Note where models AGREED despite different starting distributions
   - Surface any insight that appeared only in low-probability responses
   - Flag when the tail insight changes the conclusion vs. the modal answer
6. End with an **Agreement Map** — one line per model noting their unique contribution or blind spot.

Note: weigh the *content* of answers, not just vote tallies. Verbose responses sometimes outscore concise ones due to length bias, not quality."""
    else:
        user_msg = f"""Original question: {query}

Council answers:
{answers_block}

Peer ranking scores (higher = rated better by peers):
{score_summary}

Instructions:
1. {topic_instruction}
2. Identify where all models agree — this is the high-confidence ground.
3. Identify where models diverge — surface the disagreement explicitly.
4. Write a final synthesized answer that is better than any individual response.
5. End with an **Agreement Map** — one line per model noting their unique contribution or blind spot.

Note: weigh the *content* of answers, not just vote tallies. Verbose responses sometimes outscore concise ones due to length bias, not quality."""

    raw = await _call_anthropic(ant_client, CHAIRMAN_MODEL, [{"role": "user", "content": user_msg}], system)
    
    # Extract TOPIC: line if present
    topic_title = ""
    body = raw
    topic_match = re.match(r'^TOPIC:\s*(.+?)(?:\n|$)', raw.strip())
    if topic_match:
        topic_title = topic_match.group(1).strip()
        # Strip the TOPIC line (and any trailing blank line) from the body
        body = re.sub(r'^TOPIC:\s*.+?\n\s*\n?', '', raw.strip(), count=1).strip()
    
    return topic_title, body


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def run_council(
    query: str, 
    ctx: Any = None, 
    use_vs: bool = False, 
    lens: Optional[str] = None
) -> tuple[str, dict[str, int], str]:
    """
    Run the full 3-stage pipeline.
    `ctx` is an optional FastMCP Context for progress notifications.
    Returns the final synthesis, scores dictionary, and topic title.
    """

    async def progress(msg: str, step: int, total: int = 3) -> None:
        if ctx is not None:
            await ctx.info(msg)
            await ctx.report_progress(progress=step, total=total, message=msg)

    # Load lens template if specified
    lens_template = None
    if lens:
        lens_template = load_lens_template(lens)
        if lens_template and ctx is not None:
            await progress(f"Loaded lens template: {lens}", 0)
        elif ctx is not None and lens:
            await progress(f"Warning: Lens template '{lens}' not found", 0)

    async with _get_httpx_client() as http_client:
        or_client = _or_client(http_client)
        ant_client = _ant_client(http_client)

        await progress("Stage 1: fanning out to panel…", 0)
        stage1 = await stage1_collect(or_client, ant_client, query, use_vs, lens_template)

        await progress("Stage 2: anonymous cross-ranking…", 1)
        scores, rankings, label_to_model, labels, vs_parts = await stage2_rank(
            or_client, ant_client, query, stage1, use_vs, lens_template
        )

        await progress("Stage 3: chairman synthesis…", 2)
        topic_title, final = await stage3_synthesize(
            ant_client, query, stage1, scores, use_vs, lens_template, vs_parts
        )

        return final, scores, topic_title


# ── FastMCP server (optional) ──────────────────────────────────────────────────
# `pip install fastmcp` to enable MCP tool registration.
# Register with Claude Code:
#   claude mcp add model-council -s user \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -e OPENROUTER_API_KEY=sk-or-... \
#     -- python /abs/path/council.py --mcp
#
# MCP tool design — async start/poll pattern:
#   council_start()  → kicks off pipeline in background, returns job_id immediately
#   council_status() → poll for progress / result (never blocks > 1s)
#   council_query()  → convenience wrapper: start + poll loop (each poll < 1s)
#
# This sidesteps the MCP client hard request-timeout entirely.
# No single tool call ever blocks for more than a few seconds.

import uuid as _uuid

# Module-level job store: job_id → state dict
# State keys: status ("running"|"complete"|"error"), stage, result,
#             scores, filepath, error, start_time
_jobs: dict[str, dict] = {}
_JOB_TTL = 3600  # seconds — expire old jobs to prevent memory leaks


def _cleanup_old_jobs() -> None:
    """Remove jobs older than _JOB_TTL seconds."""
    now = time.time()
    expired = [jid for jid, j in _jobs.items() if now - j["start_time"] > _JOB_TTL]
    for jid in expired:
        _jobs.pop(jid, None)


async def _run_council_job(job_id: str, prompt: str, use_vs: bool, lens: Optional[str]) -> None:
    """Background task: run the full pipeline and write results into _jobs[job_id]."""
    job = _jobs[job_id]
    try:
        lens_template = load_lens_template(lens) if lens else None

        async with _get_httpx_client() as http_client:
            or_client = _or_client(http_client)
            ant_client = _ant_client(http_client)

            job["stage"] = "Stage 1/3: panel answering in parallel…"
            stage1 = await stage1_collect(or_client, ant_client, prompt, use_vs, lens_template)

            job["stage"] = "Stage 2/3: anonymous cross-ranking…"
            scores, rankings, label_to_model, labels, vs_parts = await stage2_rank(
                or_client, ant_client, prompt, stage1, use_vs, lens_template
            )

            job["stage"] = "Stage 3/3: chairman synthesizing…"
            topic_title, final = await stage3_synthesize(
                ant_client, prompt, stage1, scores, use_vs, lens_template, vs_parts
            )

        filepath = _save_output(prompt, final, scores, topic_title)
        job.update(status="complete", result=final, scores=scores, filepath=filepath)
    except Exception as exc:
        job.update(status="error", error=str(exc))


try:
    from fastmcp import FastMCP, Context
    from fastmcp.exceptions import ToolError

    mcp = FastMCP("model-council")

    # ── Tool 1: council_start ──────────────────────────────────────────────────
    @mcp.tool()
    async def council_start(
        prompt: str,
        use_vs: bool = False,
        lens: str = "",
    ) -> str:
        """
        Start a Model Council run in the background. Returns a job_id immediately.
        Use council_status(job_id) to poll for progress and retrieve the result.
        """
        _cleanup_old_jobs()
        job_id = str(_uuid.uuid4())[:8]  # short 8-char ID is easier to read/type
        _jobs[job_id] = {
            "status": "running",
            "stage": "Queued",
            "result": None,
            "scores": None,
            "filepath": None,
            "error": None,
            "start_time": time.time(),
            "prompt": prompt,
        }
        # Fire and forget — the task writes back into _jobs[job_id]
        asyncio.create_task(
            _run_council_job(job_id, prompt, use_vs, lens if lens else None)
        )
        return (
            f"Council started. job_id: **{job_id}**\n\n"
            f"Poll with: `council_status(\"{job_id}\")`\n"
            f"The pipeline typically takes 2–5 minutes."
        )

    # ── Tool 2: council_status ─────────────────────────────────────────────────
    @mcp.tool()
    async def council_status(job_id: str) -> str:
        """
        Check the status of a council run started with council_start().
        Returns progress info while running, or the full result when complete.
        """
        job = _jobs.get(job_id)
        if job is None:
            return f"No job found with id '{job_id}'. Jobs expire after 1 hour."

        elapsed = int(time.time() - job["start_time"])
        status = job["status"]

        if status == "running":
            return (
                f"**Status:** Running ⏳\n"
                f"**Stage:** {job['stage']}\n"
                f"**Elapsed:** {elapsed}s\n\n"
                f"Poll again in ~30s with: `council_status(\"{job_id}\")`"
            )
        elif status == "error":
            return (
                f"**Status:** Error ❌\n"
                f"**Error:** {job['error']}\n"
                f"**Elapsed:** {elapsed}s"
            )
        else:  # complete
            filepath = job.get("filepath", "")
            scores = job.get("scores", {})
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1]) if scores else []
            score_str = ", ".join(
                f"{MODEL_LABELS.get(m, m)}: {s}pts" for m, s in sorted_scores
            )
            result = job["result"] or ""
            return (
                f"**Status:** Complete ✅  ({elapsed}s)\n"
                f"**Scores:** {score_str}\n"
                f"**Report saved to:** {filepath}\n\n"
                f"---\n\n{result}"
            )

    # ── Tool 3: council_query (convenience wrapper) ────────────────────────────
    @mcp.tool()
    async def council_query(
        prompt: str,
        ctx: Context,
        use_vs: bool = False,
        lens: str = "",
    ) -> str:
        """
        Run the 3-stage Model Council and return the chairman's synthesized answer.
        Internally uses start/poll so no single call blocks long enough to time out.
        """
        try:
            await ctx.info("Model Council received your query — starting pipeline…")
            await ctx.report_progress(progress=0, total=10, message="Starting…")

            # Start the job
            _cleanup_old_jobs()
            job_id = str(_uuid.uuid4())[:8]
            _jobs[job_id] = {
                "status": "running",
                "stage": "Queued",
                "result": None,
                "scores": None,
                "filepath": None,
                "error": None,
                "start_time": time.time(),
                "prompt": prompt,
            }
            asyncio.create_task(
                _run_council_job(job_id, prompt, use_vs, lens if lens else None)
            )

            # Poll loop — each iteration returns to the event loop quickly
            poll_interval = 15.0  # seconds between polls
            max_wait = 900        # 15 min hard ceiling
            waited = 0
            beat = 1
            while waited < max_wait:
                await asyncio.sleep(poll_interval)
                waited += poll_interval

                job = _jobs.get(job_id, {})
                status = job.get("status", "error")

                if status == "running":
                    elapsed = int(time.time() - job["start_time"])
                    stage = job.get("stage", "…")
                    try:
                        await ctx.report_progress(
                            progress=min(beat, 9),
                            total=10,
                            message=f"{stage} ({elapsed}s elapsed)",
                        )
                    except Exception:
                        pass
                    beat += 1
                    continue

                if status == "error":
                    raise ToolError(f"Council failed: {job.get('error', 'unknown error')}")

                # Complete
                final = job.get("result", "")
                scores = job.get("scores", {})
                filepath = job.get("filepath", "")
                elapsed = int(time.time() - job["start_time"])

                await ctx.report_progress(progress=10, total=10, message="Complete ✅")
                await ctx.info(f"Full report saved to: {filepath}")

                sorted_scores = sorted(scores.items(), key=lambda x: -x[1]) if scores else []
                score_str = ", ".join(
                    f"{MODEL_LABELS.get(m, m)}: {s}pts" for m, s in sorted_scores
                )
                return (
                    f"**Scores:** {score_str}  |  **Time:** {elapsed}s\n"
                    f"**Report:** {filepath}\n\n---\n\n{final}"
                )

            raise ToolError("Council timed out after 15 minutes.")
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Council failed: {exc}") from exc

    _MCP_AVAILABLE = True

except ImportError:
    mcp = None
    _MCP_AVAILABLE = False


# ── Rich CLI ───────────────────────────────────────────────────────────────────

console = Console()


def _save_output(query: str, final: str, scores: dict[str, int], title: str = "") -> str:
    """Save full council synthesis to Case Outputs directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if title:
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_")
        filename = f"{timestamp}_{safe_title}.md"
    else:
        safe_query = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:30].strip("_")
        filename = f"{timestamp}_{safe_query}.md"
    
    # Anchor output dir to the directory containing this script, not cwd.
    # os.getcwd() is unreliable in MCP mode (Claude Desktop may set a different cwd).
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Case Outputs")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.abspath(os.path.join(out_dir, filename))

    sorted_models = sorted(scores.items(), key=lambda x: -x[1])
    score_summary = "\n".join(f"- {MODEL_LABELS.get(m, m)}: {s} pts" for m, s in sorted_models)

    # Strip any filepath footer from final response if it exists (for MCP repeated calls)
    clean_final = final
    if "\n\n---\n**Full Council Report:**" in final:
        clean_final = final.split("\n\n---\n**Full Council Report:**")[0]

    content = f"""# Model Council Synthesis
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Query:** {query}

## Peer Rankings
{score_summary}

## Chairman Synthesis
{clean_final}
"""
    
    # Add explicit file closing and error handling
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Validate the file was actually written
        if not os.path.exists(filepath):
            logging.error(f"Failed to create output file: {filepath}")
        elif os.path.getsize(filepath) == 0:
            logging.error(f"Output file was created but is empty: {filepath}")
        else:
            logging.info(f"Successfully saved council output to: {filepath}")
            
    except Exception as e:
        logging.error(f"Error writing to file {filepath}: {e}")
        raise
        
    return filepath


def _stage_table(stage_idx: int, start: float) -> Table:
    """Live stage-progress table shown during a council run."""
    stage_names = ["Collecting answers", "Cross-ranking peers", "Chairman synthesis"]
    stage_icons = ["⏳", "🔄", "⚗️"]
    elapsed = f"{time.time() - start:.1f}s"
    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    tbl.add_column("Model", style="cyan", min_width=26)
    tbl.add_column("Current stage", min_width=36)
    tbl.add_column("Elapsed", justify="right", min_width=10)
    icon = stage_icons[min(stage_idx, 2)]
    name = stage_names[min(stage_idx, 2)]
    for m in PANEL_MODELS:
        tbl.add_row(MODEL_LABELS.get(m, m), f"{icon}  {name}", elapsed)
    tbl.add_row(
        MODEL_LABELS.get(CHAIRMAN_MODEL, CHAIRMAN_MODEL),
        ("⚗️  Chairman synthesis" if stage_idx >= 2 else "[dim]Waiting…[/dim]"),
        elapsed,
    )
    return tbl


async def run_cli(query: str, use_vs: bool = False, lens: Optional[str] = None) -> None:
    """Run council with a Rich live display. Degrades gracefully for piped output."""
    start = time.time()

    if not console.is_terminal:
        result, scores, topic_title = await run_council(query, use_vs=use_vs, lens=lens)
        filepath = _save_output(query, result, scores, topic_title)
        print(result)
        return

    console.rule("[bold]⚖  Model Council[/bold]")
    console.print(f"[dim]Query:[/dim] {query}\n")
    
    # Show if we're using special features
    if use_vs:
        console.print("[bold cyan]Using verbalized sampling[/bold cyan]")
    if lens:
        lens_template = load_lens_template(lens)
        if lens_template:
            console.print(f"[bold cyan]Using lens template:[/bold cyan] {lens}")
        else:
            console.print(f"[bold red]Warning:[/bold red] Lens template '{lens}' not found")

    async with _get_httpx_client() as http_client:
        or_client = _or_client(http_client)
        ant_client = _ant_client(http_client)

        with Live(_stage_table(0, start), refresh_per_second=4, console=console, vertical_overflow="visible") as live:

            # Stage 1 
            stage1 = await stage1_collect(or_client, ant_client, query, use_vs, 
                                         load_lens_template(lens) if lens else None)
            live.update(_stage_table(1, start))

            console.print()
            for model, answer in stage1.items():
                snippet = answer[:250].replace("\n", " ") + ("…" if len(answer) > 250 else "")
                console.print(f"[bold cyan]{MODEL_LABELS.get(model, model)}[/bold cyan]")
                console.print(f"[dim]{snippet}[/dim]\n")

            # Stage 2
            scores, rankings, label_to_model, labels, vs_parts = await stage2_rank(
                or_client, ant_client, query, stage1, use_vs, 
                load_lens_template(lens) if lens else None
            )
            live.update(_stage_table(2, start))

            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            score_str = "   ".join(
                f"[bold]{MODEL_LABELS.get(m, m)}[/bold] {s}pt{'s' if s != 1 else ''}"
                for m, s in sorted_scores
            )
            console.print(f"[dim]Peer scores:[/dim]  {score_str}\n")

            # Stage 3
            topic_title, final = await stage3_synthesize(
                ant_client, query, stage1, scores, use_vs,
                load_lens_template(lens) if lens else None, vs_parts
            )

    total = time.time() - start
    filepath = _save_output(query, final, scores, topic_title)
    winner_name = MODEL_LABELS.get(sorted_scores[0][0], sorted_scores[0][0])

    console.rule(f"[bold green]Council Complete[/bold green]  •  {total:.1f}s")
    console.print(f"\n[bold green]Winner:[/bold green] {winner_name}")
    console.print(f"[bold cyan]Full Report:[/bold cyan] {filepath}\n")
    console.rule()


# ── Typer app ──────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="council",
    help="Model Council — Karpathy-style 3-stage multi-model council with verbalized sampling and lens templates",
    add_completion=False,
)


@app.command()
def ask(
    prompt: str = typer.Argument(None, help="Your question (omit for interactive REPL)"),
    mcp_server: bool = typer.Option(False, "--mcp", help="Launch as MCP stdio server"),
    http: bool = typer.Option(False, "--http", help="Use HTTP transport instead of stdio (requires --mcp)"),
    vs: bool = typer.Option(False, "--vs", help="Use verbalized sampling to explore model uncertainty"),
    lens: str = typer.Option("", "--lens", help="Apply a lens template from the lenses directory (e.g., 'leap', 'career-bet', 'stalker-strategy')"),
) -> None:
    """
    Ask the council a question, or run in interactive mode.

    \b
    Examples:
      python council.py "What is the best way to learn Rust?"
      python council.py              # interactive REPL
      python council.py --mcp        # MCP stdio server for Claude Code/Desktop
    """
    if mcp_server:
        if not _MCP_AVAILABLE:
            typer.echo("fastmcp not installed.  Run: pip install fastmcp", err=True)
            raise typer.Exit(1)
        transport = "http" if http else "stdio"
        mcp.run(transport=transport)  # type: ignore[union-attr]
        return

    if prompt:
        asyncio.run(run_cli(prompt, use_vs=vs, lens=lens if lens else None))
        return

    # Interactive REPL
    history_path = os.path.expanduser("~/.council_history")
    session: PromptSession = PromptSession(history=FileHistory(history_path))
    console.print("[bold]⚖  Model Council[/bold] — interactive mode  [dim](Ctrl+C to exit)[/dim]\n")

    # Get lens template options for autocomplete
    lens_options = []
    if os.path.exists(LENSES_DIR):
        lens_options = [os.path.splitext(f)[0] for f in os.listdir(LENSES_DIR) 
                       if os.path.isfile(os.path.join(LENSES_DIR, f)) and f.endswith('.md')]
    
    use_vs = False
    current_lens = None
    
    while True:
        try:
            query = session.prompt("Council › ").strip()
            
            # Check for command prefixes
            if query.startswith(":vs"):
                use_vs = not use_vs
                console.print(f"[bold]Verbalized sampling {'enabled' if use_vs else 'disabled'}[/bold]")
                continue
                
            if query.startswith(":lens "):
                lens_name = query[6:].strip()
                if lens_name.lower() == "none":
                    current_lens = None
                    console.print("[bold]Lens template cleared[/bold]")
                elif lens_name in lens_options:
                    current_lens = lens_name
                    console.print(f"[bold]Using lens template:[/bold] [cyan]{lens_name}[/cyan]")
                else:
                    console.print(f"[bold red]Unknown lens:[/bold red] {lens_name}")
                    console.print(f"[dim]Available lenses: {', '.join(lens_options)}[/dim]")
                continue
                
            if query.startswith(":help"):
                console.print("[bold]Commands:[/bold]")
                console.print("  :vs              - Toggle verbalized sampling")
                console.print(f"  :lens <name>     - Set lens template ({', '.join(lens_options)})")
                console.print("  :lens none       - Clear lens template")
                console.print("  :help            - Show this help")
                continue
                
        except (KeyboardInterrupt, EOFError):
            break
            
        if not query:
            continue

        try:
            asyncio.run(run_cli(query, use_vs=use_vs, lens=current_lens))
        except KeyboardInterrupt:
            # Ctrl+C during a running query — don't exit the REPL, just cancel this query
            console.print("\n[yellow]Query cancelled.[/yellow]")
        except Exception as exc:
            # Any other error (network, API, quorum failure, etc.) — show it and continue
            console.print(f"\n[bold red]Error:[/bold red] {exc}\n")

    console.print("\n[dim]Goodbye.[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
