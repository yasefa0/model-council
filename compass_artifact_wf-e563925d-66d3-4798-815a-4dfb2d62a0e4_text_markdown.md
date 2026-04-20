# A hybrid architecture for your upgraded Model Council

**Go hybrid: Anthropic SDK direct for Claude, OpenRouter for everything else; wrap the pipeline in a FastMCP 2.x server with a typer + rich + prompt_toolkit CLI front end; and upgrade your panel to GPT‑5.4 / Gemini 3.1 Pro / Grok 4.20 with Claude Opus 4.7 as chairman.** That single stack change — splitting the router, not unifying it — is worth more than any other optimization because Anthropic's prompt caching, which dominates cost on a council that calls Claude five out of seven times, is demonstrably unreliable when proxied through OpenRouter. Your current all‑OpenRouter, all‑AsyncOpenAI design is clean but leaves roughly **2–5× cost headroom** on the table for long conversations and ships with two generations of stale models. The rest of this report walks through each decision with the reasoning and concrete code patterns you need to rewrite `council.py`.

## The routing decision hinges on prompt caching, not latency

OpenRouter's raw overhead is small — its edge adds roughly **25–70 ms per request** according to its own docs and an independent 1,000‑call benchmark from early 2026, well under the variance of LLM inference itself. The 5.5% credit‑purchase fee is also easy to accept for convenience. What is **not** acceptable is the well‑documented degradation of Anthropic prompt caching when routed through OpenRouter: the `sst/opencode` issue #1245 (July 2025) and the `OpenRouterTeam/ai-sdk-provider` issue #35 both confirm that **OpenRouter caches the system message but silently fails to update cache breakpoints on multi‑turn calls**, making long conversations "an order of magnitude more expensive than Anthropic direct." Anthropic's newer features — 1‑hour cache TTL, structured outputs (GA November 2025), interleaved thinking, and the advisor tool — also ship on the native API first and trickle to OpenRouter with partial support.

For a council where **Claude is invoked five times per query** (three panelist calls if you keep Claude on the panel, plus two synthesis/ranker calls by Opus), those cache misses compound. The answer is a hybrid: keep the `AsyncOpenAI` client pointed at OpenRouter for GPT and Gemini, where cache economics don't dominate and the unified schema is genuinely convenient, and add `AsyncAnthropic` directly for all Claude calls. You trade one extra SDK and auth flow for reliable caching, day‑zero feature access, and no middleman hop on your most expensive calls. Both clients should share tuned `httpx.AsyncClient` instances with **`http2=True`, `max_connections=50`, `keepalive_expiry=30`, and a four‑field timeout** (`connect=5, read=120, write=10, pool=5`). Inject these into each SDK via `http_client=` and set `max_retries=0` on the SDKs themselves so your retry wrapper doesn't get multiplied.

## The 2026 council: swap three of four models

Your current lineup is **stale by two full release cycles**. Karpathy's original Model Council from November 2025 already used GPT‑5.1, Gemini 3 Pro Preview, Sonnet 4.5, and Grok 4 — and the frontier has moved again since. As of April 2026, LMArena shows the top six models (Opus 4.6 Thinking, Gemini 3.1 Pro, Grok 4.20, GPT‑5.4) clustered within a **~20 Elo gap**, effectively tied at the frontier. The right panel optimizes for *diversity of training lineage* rather than leaderboard rank, because peer‑ranking benefits from genuine disagreement.

The recommended panel is **`openai/gpt-5.4` (generalist, 1M ctx, $2.50/$15), `google/gemini-3.1-pro-preview` (reasoning + multimodal, 1M ctx, $2/$12), and `x-ai/grok-4.20` (low‑hallucination, 2M ctx, $2/$6)**. Three completely different RLHF lineages produce genuine disagreement; Grok specifically replaces "another Claude voice" on the panel, which matters because research on LLM‑as‑judge (JudgeBench, RewardBench) shows consistent **self‑enhancement bias** — judges favor models from their own family. For the chairman, upgrade to **`anthropic/claude-opus-4.7`** (released April 16, 2026): same $5/$25 price as your current Opus 4.5 but with a 1M context window (vs. 200K), the longest measured task‑completion horizon of any frontier model per METR, and no self‑bias problem because no Claude variant sits on the panel. For Stage 2 cross‑ranking, keep Karpathy's original design of panelists ranking their anonymized peers; if you need to cut cost, substitute a single `x-ai/grok-4.1-fast` call at $0.20/$0.50 rather than running Haiku, which would re‑introduce same‑family bias under an Opus chairman.

If per‑query cost matters more than peak quality, a **budget alternative** of GPT‑5.4 + Gemini 3 Pro + DeepSeek V3.2 (panel) with Gemini 3.1 Pro as chairman drops you from ~$0.14–0.20 per query to **~$0.04–0.07** with an estimated 90% of the quality. DeepSeek V3.2 at $0.26/$0.42 is genuinely frontier‑competitive on reasoning and brings a completely different (Chinese‑lab, MIT‑licensed) training lineage. One caveat worth coding around: Karpathy observed during his original experiment that cross‑LLM juries **systematically favor GPT‑family verbosity** and rank Claude lowest, so your chairman prompt should explicitly be told to weigh panel content, not peer votes, when synthesizing.

## FastMCP 2.x is the right server framework, and stdio is the right transport

The MCP ecosystem consolidated in 2025: **FastMCP 1.0 was merged into Anthropic's official Python SDK** (what you get from `pip install mcp`), while **FastMCP 2.x/3.x remained standalone and actively maintained** at `PrefectHQ/fastmcp` with ~1M daily downloads and FastMCP 3.0 released January 19, 2026. Use the standalone `fastmcp` package — it's a strict superset of the 1.x API, adds `ctx.report_progress()` and `ctx.info()` notifications that the official SDK lags on, ships an `fastmcp install claude-desktop` / `claude-code` auto‑registrar, and has first‑class OpenAPI proxying if you ever want to expose your council over HTTP to non‑Claude clients.

For a locally‑invoked council called from Claude Desktop and Claude Code, **use stdio transport, not HTTP**. Claude Desktop can't speak HTTP directly anyway (it requires an `npx mcp-remote` proxy), and stdio has the smallest footprint. The only stdio gotcha is absolute and non‑negotiable: **never write to stdout** from your tool or any imported library — it corrupts the JSON‑RPC framing. Route all logging to stderr via `logging.basicConfig(stream=sys.stderr)`, and if you run under Docker set `PYTHONUNBUFFERED=1`. The deprecated SSE transport (replaced by Streamable HTTP in spec 2025‑03‑26) is only relevant if you later want to host the council remotely.

Crucially for your UX, MCP's `notifications/progress` mechanism lets long‑running tools stream status back to the client. A tool signature like `async def council_query(prompt: str, ctx: Context) -> str:` gets a `Context` injected automatically when FastMCP sees the type hint, and `await ctx.report_progress(progress=1, total=3, message="Stage 1 complete")` will surface "Stage 1 complete, running cross‑ranking…" in Claude Code's status line. Progress notifications also **reset the client's activity timer**, which solves the 60‑second timeout that would otherwise kill your 30–60 second pipeline — emit a progress update at minimum every 10 seconds. Raise `fastmcp.exceptions.ToolError` for user‑actionable failures (bad input, known outage) so they return as `isError: true` with a clean message rather than a protocol crash.

## The dual-mode file pattern is a single-argparse switch

The cleanest way to ship one file that runs as both CLI and MCP server is the pattern used by FastMCP's own examples and the Strava MCP reference server: put your pipeline in a pure async function that takes `ctx: Context | None = None`, wrap it with an `@mcp.tool` decorator, and dispatch in `main()`:

```python
# council.py — structural skeleton
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
import argparse, asyncio

mcp = FastMCP("model-council")

async def run_council(prompt: str, ctx: Context | None = None) -> str:
    async def progress(msg, step, total=3):
        if ctx:
            await ctx.info(msg)
            await ctx.report_progress(progress=step, total=total, message=msg)
    await progress("Stage 1: fan-out to panel", 0)
    answers = await stage1_collect(prompt)     # asyncio.gather with return_exceptions
    await progress("Stage 2: anonymous cross-ranking", 1)
    rankings = await stage2_rank(answers)
    await progress("Stage 3: chairman synthesis", 2)
    try:
        return await stage3_synthesize(prompt, answers, rankings)
    except Exception as e:
        raise ToolError(f"Synthesis failed: {e}") from e

@mcp.tool
async def council_query(prompt: str, ctx: Context) -> str:
    """Run the 3-stage Model Council and return the synthesized answer."""
    return await run_council(prompt, ctx)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?")
    p.add_argument("--mcp", action="store_true")
    p.add_argument("--http", action="store_true")
    args = p.parse_args()
    if args.mcp:
        mcp.run(transport="http" if args.http else "stdio")
    elif args.prompt:
        print(asyncio.run(run_council(args.prompt)))
```

Registration is a one‑liner per client. For Claude Code: `claude mcp add model-council -s user -e ANTHROPIC_API_KEY=... -e OPENROUTER_API_KEY=... -- python /abs/path/council.py --mcp`. For Claude Desktop, add an entry to `~/Library/Application Support/Claude/claude_desktop_config.json` with an **absolute path** to your Python interpreter (Claude Desktop does not inherit shell PATH, which is the #1 cause of "server failed to start" reports). If you're using `uv`, the cleanest pattern is `"command": "uv", "args": ["run", "--with", "fastmcp", "python", "/abs/path/council.py", "--mcp"]` — no venv management required.

## Async fan-out: gather with return_exceptions, not TaskGroup

The single most common mistake when migrating parallel LLM code to 3.11+ is reaching for `asyncio.TaskGroup` because it's "newer." **Don't.** TaskGroup cancels all siblings on the first exception, which is correct for all‑or‑nothing workflows but catastrophically wrong for a council: one transient 429 on Claude would kill the GPT and Gemini calls mid‑flight. Use `asyncio.gather(*coros, return_exceptions=True)` so partial failures come back as exception objects in index order, wrap the whole fan‑out in an `asyncio.wait_for(..., timeout=180)` outer guard, and filter afterward — require N of M panelists to succeed before continuing to Stage 2 rather than aborting on any single failure.

For retries, **use `stamina` (by Hynek) over `tenacity`** as the 2026 default: it has sane exponential‑backoff‑with‑jitter presets, first‑class async support, honors `Retry-After` headers, and ships a test‑mode toggle to disable retries in CI. Retry only on `RateLimitError`, `APITimeoutError`, `httpx.TransportError`, and `httpx.PoolTimeout` — never on 400/401/403 or context‑length errors. Set SDK `max_retries=0` so stamina doesn't compound with built‑in retries. One subtle but pervasive bug to know about: `openai-python` issue #763 confirms that **streaming responses leak connections back into the pool unless explicitly closed**, so always use `async with client.chat.completions.stream(...) as stream:` rather than a bare `await`. Over hours of uptime, leaked streams produce the dreaded `httpx.PoolTimeout` (httpx #2556), which is why a short `pool=5` timeout is important — you want to fail fast rather than queue behind dead connections.

## The CLI stack: typer + rich + prompt_toolkit

For the user‑facing CLI, **use `typer` for argument parsing, `rich` for rendering, and `prompt_toolkit` for the interactive REPL**. Skip `textual` — it's a full‑screen TUI framework and Claude Code itself is actually a line‑oriented CLI using `rich.live.Live` regions, not a TUI app. Skip `argparse` for anything beyond the dual‑mode switch; typer's type‑hint‑driven API cuts boilerplate to near zero. Skip `questionary` for the chat loop (it's for one‑shot wizards) in favor of `PromptSession.prompt_async()`, which gives you file history, reverse search, bracketed paste, and multi‑line input for free.

Streaming markdown to the terminal has one canonical pattern, used by `pydantic-ai`'s `stream_markdown.py` example and `gianlucatruda/richify`: accumulate incoming tokens into a buffer and re‑render `Markdown(buffer)` inside a `rich.Live` region at `refresh_per_second=10`. Partial markdown self‑heals — an unclosed triple‑backtick renders as plain text until the closer arrives — so don't try to parse tokens incrementally. Set `vertical_overflow="visible"` to avoid the ellipsis truncation that bites long LLM outputs. For the 3‑stage pipeline display, use a `rich.Table` inside a single `Live` region with one row per model, showing a spinner icon, token count, and elapsed time — this is cleaner than `rich.progress.Progress`, which assumes homogeneous tasks with known totals. The two highest‑leverage UX touches from Claude Code worth replicating are the **rotating spinner verbs** ("Considering…", "Pondering…", "Ideating…") and the **live telemetry line** showing elapsed time and per‑model token counts.

One cross‑platform concern to bake in from the start: branch on `console.is_terminal` to disable `Live` regions and spinners when stdout is piped, so `council ask "q" > out.md` produces clean markdown and `council ask "q" | jq` works in scripts. Respect `NO_COLOR` and `FORCE_COLOR` environment variables (rich does this automatically). Add a `--output-format text|markdown|json` flag mirroring Claude Code's own, so automation can parse results. For reference implementations to steal patterns from, the best two are **`simonw/llm`** for overall CLI shape (model abstractions, SQLite logging, slash commands, `!multi`/`!edit` hooks) and **`pydantic-ai/examples/stream_markdown.py`** for the 40‑line streaming markdown renderer including the `SimpleCodeBlock` patch that makes fenced code blocks copy‑friendly.

## Putting it all together: the upgrade path

Your next `council.py` has four concrete changes from the current version. First, split the API client: keep `AsyncOpenAI` pointed at OpenRouter for GPT‑5.4, Gemini 3.1 Pro, and Grok 4.20, but add `AsyncAnthropic` for Opus 4.7 synthesis with explicit `cache_control` breakpoints on any large shared rubric — this alone will cut multi‑turn cost materially. Second, replace `asyncio.gather(...)` with `asyncio.gather(..., return_exceptions=True)` wrapped in `asyncio.wait_for(timeout=180)`, require a minimum quorum (e.g. 2 of 3 panelists) before proceeding, and add `stamina.retry` decorators with `max_retries=0` on the SDKs. Third, wrap your pipeline in a FastMCP 2.x `@mcp.tool` with a `Context` parameter and emit `ctx.report_progress` at each stage boundary — this gives you MCP integration, Claude Code status‑line updates, and keeps the client's timeout clock from expiring during long syntheses. Fourth, add a `typer` CLI in front with a rich‑based streaming display and a `prompt_toolkit` REPL, gating rich rendering on `console.is_terminal` so pipes stay clean.

The final architecture is decisively **more capable, moderately cheaper, and substantially more reliable** than what you have today. Total per‑query cost stays in your current $0.14–0.20 band with a frontier‑tier panel; drops to $0.04–0.07 with the DeepSeek budget variant; and, critically, no longer silently loses 60–90% of your Anthropic cache hits on long conversations. The MCP layer costs essentially nothing in code (the core pipeline needs no changes — just the thin `@mcp.tool` wrapper and the `Context` progress calls) but turns your council from a standalone CLI into a tool that Claude itself can invoke recursively, which is the most interesting compound use case this architecture unlocks.