# AI Coding Agent
<img width="1920" height="1069" alt="image" src="https://github.com/user-attachments/assets/35465f1b-e438-4b26-bdaa-ed535351130f" />

A local Textual coding assistant built with LangChain Deep Agents and OpenRouter.
The manager can delegate implementation, read-only review, and approved checks
to specialist subagents.

## Setup

1. Install Python 3.12+ and [uv](https://docs.astral.sh/uv/).
2. From this repository, run `uv sync`.
3. Copy `.env.example` to `.env` in this repository and set `OPENROUTER_API_KEY`.
4. Set `OPENROUTER_MODEL` to a tool-calling model available in your OpenRouter account.
5. Set `TAVILY_API_KEY` to enable the agent's Tavily web search tool.

The agent loads credentials from this repository's `.env`, not the project being
edited.

With Tavily configured, the agent can search up to five general web results for
current facts, external research, and up-to-date documentation. Search results
are treated as reference material and their links can be included in responses.

## Large project tasks

- For large or unfamiliar repositories, the agent searches and reads relevant
  files directly with its workspace filesystem tools.
- The agent can run a workspace-relative Python script (for example,
  `uv run python sudoku.py`) only after you approve the command. It returns the
  script's output and exit code. Python environment probes are limited to fixed
  read-only checks.
- An implementation request with no file-write action is shown as incomplete,
  rather than marked finished.

## Open a project from VS Code

Open a project's folder in VS Code, then use its integrated terminal. From that
terminal, run:

```powershell
uv run --project "C:\path\to\ai-coding-agent" python "C:\path\to\ai-coding-agent\terminal\main.py" --workspace .
```

Replace both paths with where you cloned this agent. The current folder (`.`)
becomes the agent's workspace. You can also pass an absolute path:

```powershell
uv run --project "C:\path\to\ai-coding-agent" python "C:\path\to\ai-coding-agent\terminal\main.py" --workspace "C:\path\to\your-project"
```

## Safety and controls

- The agent can read and edit files only inside the selected workspace.
- `.env` files, `.git`, and `.venv` paths are blocked from agent filesystem access.
- Every file write, deletion, and allowed command pauses for review. Use the
  visible **Approve** and **Reject** buttons, or enter `/approve` and `/reject`.
- Commands first pass a safety allowlist that blocks unknown executables,
  shell operators, and paths outside the selected workspace. Every command that
  passes is shown for your approval before it runs; use the Approve button or
  type `/approve` or `accept`. Rejection or no approval means it does not run.
  Approved commands run with your Windows account
  permissions. File writes and later commands are separate actions, so a
  command requested after an approved write receives its own safety check and
  approval prompt. Virtual `/workspace/...` script paths are resolved to
  project-relative paths in the selected workspace before execution.
- The agent can add or sync project dependencies with `uv`, `npm`, `pnpm`, `cargo`,
  `go`, `.NET`, or Composer. Python projects without `pyproject.toml` use an
  explicit project `.venv`. Installs require approval; npm/pnpm scripts and
  Composer scripts/plugins are disabled.
- The process row animates while working; the chat shows the manager's plan,
  tool activity, and specialist-agent progress.
- A persistent process row shows `WORKING`, `WAITING FOR APPROVAL`, `FINISHED`,
  `STOPPED`, or `FAILED`, so the current run state stays visible while the chat scrolls.
- The prompt sends with `Enter`; press `Shift+Enter` to add a paragraph line.
- Press **Stop** to cancel the active response and command. Completed file
  changes are kept. Sending a new message while the agent is busy stops the
  current run and starts handling the new request. A new message also declines
  any pending, unapproved action.
- `Ctrl+L` clears the visible feed, `Shift+Tab` cycles the displayed mode, and
  `Ctrl+C` exits.

Conversation checkpoints are held in memory and last only for the running app
session.
