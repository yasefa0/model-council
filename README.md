# Model Council "2026-Frontier" Build

Karpathy-style three-stage multi-model council — upgraded with hybrid routing, hardened connection pooling, automatic Markdown reporting, FastMCP 3.x server mode, and a 2026-frontier panel.

## Architecture

```
Stage 1 — Fan-out     All panel models answer in parallel
Stage 2 — Peer jury   Each panelist ranks the others anonymously; scores tallied
Stage 3 — Synthesis   Chairman (Claude Opus, direct Anthropic) writes final answer + agreement map
```

## 2026-Frontier Council

| Role | Model | Routing |
|---|---|---|
| Panelist | GPT-5.4 | OpenRouter |
| Panelist | Gemini 3.1 Pro | OpenRouter |
| Panelist | Grok 4.20 | OpenRouter |
| Chairman | Claude Opus 4.7 | Anthropic direct |

> **Why Grok instead of another Claude?**
> Claude is the chairman. Keeping Claude off the panel eliminates same-family self-enhancement bias in Stage 2 peer voting. Three different RLHF lineages produce genuine disagreement — which is the whole point.

> **Why Anthropic direct for the chairman?**
> Anthropic prompt caching is unreliable through OpenRouter proxies (~2–5× more expensive on long conversations). The chairman runs direct for reliable cache hits.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Optional: for MCP server mode, also install fastmcp
pip install fastmcp>=3.0.0
# or: pip install ".[mcp]"

# 2. Set API keys
cp .env.example .env
# Edit .env — you need BOTH keys:
#   OPENROUTER_API_KEY  →  https://openrouter.com/keys
#   ANTHROPIC_API_KEY   →  https://console.anthropic.com/keys

# 3. Run
python council.py "your question here"

# Interactive REPL (with history)
python council.py

# Pipe-friendly (auto-detects non-terminal output)
python council.py "your question" > answer.md

# Full Reports
The council automatically exports full Markdown syntheses to the `Case Outputs/` directory.
```

## Verbalized Sampling

Verbalized Sampling (VS) is a technique that explores model uncertainty by analyzing tail distribution responses. When enabled:

1. Each model generates 5 distinct responses with assigned probabilities  
2. The model selects the lowest-probability response as its submission
3. The chairman notes agreement despite different starting distributions
4. The chairman surfaces insights from low-probability responses
5. The chairman flags when tail insights change conclusions

Enable with the `--vs` flag or `:vs` in REPL mode:

```bash
# CLI mode
python council.py --vs "What is the best database for time series data?"

# REPL mode
python council.py
>>> :vs          # toggles verbalized sampling on/off
>>> What is the best database for time series data?
```

## Lens Templates

Lens Templates provide specialized decision frameworks for different types of questions. Templates from the `/lenses` directory are injected into model system prompts.

Available templates:
- `leap`: For evaluating high-risk, high-reward decisions or innovations
- `career-bet`: For career path and professional development decisions 
- `stalker-strategy`: For competitive analysis and market positioning

Enable with the `--lens` flag or `:lens` in REPL mode:

```bash
# CLI mode
python council.py --lens leap "Should I pivot my startup to focus on AI?"

# REPL mode
python council.py
>>> :lens career-bet
>>> Should I change careers from software engineering to AI research?
```

You can create custom lens templates by adding markdown files to the `/lenses` directory.

## MCP Server Mode

Registers the council as a tool that Claude Code or Claude Desktop can call directly. Requires `fastmcp>=3.0.0` (`pip install fastmcp`).

The MCP server exposes three tools:

| Tool | Purpose |
|---|---|
| `council_start` | Starts a pipeline run in the background, returns a job_id immediately |
| `council_status` | Polls a running job for progress; returns the full result when complete |
| `council_query` | Convenience wrapper: starts a job and polls until completion |

This async start/poll pattern ensures no single tool call ever blocks long enough to trigger an MCP client timeout.

```bash
# Register with Claude Code
claude mcp add model-council -s user \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OPENROUTER_API_KEY=sk-or-... \
  -- python /absolute/path/to/council.py --mcp

# Claude Desktop (add to ~/Library/Application Support/Claude/claude_desktop_config.json)
{
  "mcpServers": {
    "model-council": {
      "command": "python",
      "args": ["/absolute/path/to/council.py", "--mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Once registered, ask Claude: *"Use the model council to answer: what's the best database for time-series data?"*

You can also use Verbalized Sampling and Lens Templates with MCP:

```
Use the model council to answer with VS enabled: what's the best database for time-series data?

Use the model council with the career-bet lens: should I switch from software engineering to product management?
```

## Swap Models

Edit the top of `council.py`:

```python
PANEL_MODELS = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.20",
]
CHAIRMAN_MODEL = "claude-opus-4-7"   # always Anthropic direct
```

Any model on OpenRouter works for panel seats. Browse: https://openrouter.ai/models

## Cost

Each council run makes:
- 3 parallel calls (Stage 1)
- 3 parallel calls (Stage 2)
- 1 chairman call (Stage 3, Anthropic direct with caching)
= **7 total API calls**

Approximate cost with default panel: **~$0.10–0.20 per query** depending on answer length.

## What's new in the 2026 Build

| | 2024 Legacy | 2026 Frontier |
|---|---|---|
| API routing | All-meta-router | Hybrid (Anthropic direct for Opus caching) |
| Panel | Same-family bias | Diversified (GPT + Gemini + Grok) |
| Client | Individual calls | Hardened `AsyncClient` Pool (HTTP/2) |
| Error handling | Silent failures | `return_exceptions=True` + Quorum check |
| Retries | Static | `stamina` (Jittered backoff) |
| Reporting | Terminal only | Automatic Markdown Export (`Case Outputs/`) |
| CLI | `argparse` | `typer` + `rich.Live` + `prompt_toolkit` REPL |
| MCP | ✗ | ✓ FastMCP 3.x `@mcp.tool` with async start/poll pattern |
| Uncertainty | ✗ | ✓ Verbalized Sampling (VS) with tail distribution insights |
| Templates | ✗ | ✓ Lens Templates for specialized decision frameworks |