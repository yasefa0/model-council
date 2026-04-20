# Changelog

## v2026.1.2 - 2026-04-24

### Fixed
- **MCP Server Disconnect (Critical)**: `FastMCP("model-council", timeout=900)` crashed on import because FastMCP 3.x removed the `timeout` constructor kwarg. Moved timeout to the tool decorator: `@mcp.tool(timeout=900)`. This caused the "Server disconnected" error in Claude Desktop / Claude Code.
- **`_save_output` wrong path in MCP mode**: Output directory was anchored to `os.getcwd()` (unreliable when Claude Desktop launches the server), now anchored to `os.path.dirname(__file__)` so files always save next to `council.py`.
- **Silent ranking parser failures**: `except Exception: pass` in Stage 2 now emits `logging.warning(...)` instead of swallowing errors silently.

### Improved
- **`load_lens_template` refactored**: Removed redundant `glob.glob()` + iteration loop; now uses direct `os.path.isfile()` path check. Code is shorter and faster.
- **`extract_vs_parts` is no longer `async`**: Function had no `await` expressions — corrected to plain `def`. Removed erroneous `await` call at the call site.
- **Removed unused `import yaml`**: YAML front matter in lens files is stripped with a simple `split("---", 2)` — no YAML parser needed.
- **Removed unused `import glob`**: No longer needed after `load_lens_template` refactor.
- **Explicit `httpx` and `pyyaml` in dependencies**: Both are direct dependencies but were previously missing from `requirements.txt` and `pyproject.toml`.
- **Updated FastMCP minimum version**: `fastmcp>=2.0.0` → `fastmcp>=3.0.0` in all dependency files.

## v2026.1.1 - 2026-04-20

### Fixed
- **MCP Integration**: Fixed issue where Markdown files were not being saved in `Case Outputs` directory when called from Claude Desktop or Claude Code.
- **MCP Output Truncation**: Fixed issue where Model Council output would get cut off in Claude Desktop by including report filepath in the response and increasing token limits.
- **Absolute Paths**: Added absolute path resolution to ensure consistent file access across different execution contexts, especially in MCP mode.
- **File Validation**: Added explicit file validation and error handling for output files to ensure robust file saving.

## v2026.1.0 - 2026-04-20

### Fixed
- **MCP Integration**: Fixed issue where Markdown files were not being saved in `Case Outputs` directory when called from Claude Desktop or Claude Code.
- **Truncation Issue**: Increased `max_tokens` parameter from 1500 to 2500 for Anthropic calls to prevent truncation in longer responses.
- **HTTP Client**: Implemented shared `httpx.AsyncClient` pool with HTTP/2 support for better connection reuse and performance.
- **Code Structure**: Fixed indentation issues in the MCP code section.

### Added
- **Full Reports**: Added automatic Markdown export to `Case Outputs` directory for all council runs.
- **Terminal Improvements**: Updated terminal output to show the winner of the ranking phase and the absolute path to the full report.

### Changed
- **Frontier Models**: Updated to 2026 frontier models:
  - GPT-5.4 (1M context)
  - Gemini 3.1 Pro (1M context)
  - Grok 4.20 (2M context)
  - Claude Opus 4.7 (Chairman)

## v2.0.0 - 2024-10-15

Initial version with:
- Hybrid routing (Anthropic direct for chairman, OpenRouter for panel)
- Panel with different model families (GPT + Gemini + Grok)
- Error handling with quorum check
- FastMCP 2.x integration
- Rich CLI with terminal UI