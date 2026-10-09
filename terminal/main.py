from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static, TextArea
from textual.widget import Widget
from textual.binding import Binding
from rich.markup import escape
from random import choice
from typing import Iterable
import argparse
import asyncio
from pathlib import Path
from time import monotonic
import shlex
import sqlite3
from datetime import datetime, timezone

if __package__:
    from .agent import CodingAgent, validate_command
else:
    from agent import CodingAgent, validate_command


ACTIVITY_STATES = {
    "running": ("RUNNING", "#ff7a45"),
    "stopped": ("STOPPED", "#ff7a45"),
    "done": ("DONE", "#ffffff"),
    "success": ("DONE", "#ffffff"),
    "failed": ("FAILED", "#ff7a45"),
    "error": ("FAILED", "#ff7a45"),
    "pending": ("PENDING", "#ffffff"),
}


def activity_state(status: str) -> tuple[str, str]:
    """Return a readable label and palette color for an activity status."""
    return ACTIVITY_STATES.get(status.lower(), (status.upper(), "#ffffff"))


class Message(Static):
    """A user or agent chat message."""

    def __init__(self, role: str, message: str, *, color: str = "#ffffff") -> None:
        self.role = role
        self.message = message
        self.color = color
        super().__init__(classes=f"{role.lower()}-message")
        self.refresh_content()

    def append_text(self, text: str) -> None:
        self.message += text
        self.refresh_content()

    def refresh_content(self) -> None:
        safe_role = escape(self.role.upper())
        self.update(f"[bold {self.color}]{safe_role}[/]\n{escape(self.message)}")


class ToolActivity(Static):
    """A tool invocation whose status can be updated as it runs."""

    def __init__(
        self,
        tool_name: str,
        detail: str = "",
        *,
        status: str = "running",
    ) -> None:
        self.tool_name = tool_name
        self.detail = detail
        self.status = status
        super().__init__(classes="tool")
        self.refresh_content()

    def set_status(self, status: str, detail: str | None = None) -> None:
        """Update the activity after the tool starts, finishes, or fails."""
        self.status = status
        if detail is not None:
            self.detail = detail
        self.refresh_content()

    def refresh_content(self) -> None:
        label, color = activity_state(self.status)
        detail = f"  {escape(self.detail)}" if self.detail else ""
        self.update(
            f"[bold #ff7a45]>[/] [bold #ffffff]{escape(self.tool_name)}[/] "
            f"[{color}]{label}[/]{detail}"
        )


class StatusActivity(Static):
    """A short progress or lifecycle update from the agent."""

    def __init__(self, message: str, *, status: str = "running") -> None:
        self.message = message
        self.status = status
        super().__init__(classes="status-activity")
        self.refresh_content()

    def set_status(self, status: str, message: str | None = None) -> None:
        self.status = status
        if message is not None:
            self.message = message
        self.refresh_content()

    def refresh_content(self) -> None:
        label, color = activity_state(self.status)
        self.update(f"[{color}]{label}[/]  {escape(self.message)}")


class CommandActivity(StatusActivity):
    """A shell command and its current outcome."""

    def __init__(
        self,
        command: str,
        *,
        status: str = "running",
        output: str = "",
    ) -> None:
        self.command = command
        self.output = output
        super().__init__(command, status=status)
        self.refresh_content()

    def set_status(self, status: str, output: str | None = None) -> None:
        self.status = status
        if output is not None:
            self.output = output
        self.refresh_content()

    def refresh_content(self) -> None:
        label, color = activity_state(self.status)
        content = f"[{color}]{label}[/]  [bold #ffffff]$ {escape(self.command)}[/]"
        if self.output:
            content += f"\n  {escape(self.output)}"
        self.update(content)


class FileActivity(StatusActivity):
    """A file read, create, edit, or delete activity."""

    def __init__(self, action: str, path: str, *, status: str = "done") -> None:
        self.action = action
        self.path = path
        super().__init__(path, status=status)
        self.refresh_content()

    def refresh_content(self) -> None:
        label, color = activity_state(self.status)
        self.update(
            f"[{color}]{label}[/]  [bold #ffffff]{escape(self.action.upper())}[/] "
            f"{escape(self.path)}"
        )


class PlanActivity(Static):
    """A compact checklist for an agent plan."""

    def __init__(self, steps: Iterable[tuple[str, bool]]) -> None:
        self.steps = list(steps)
        super().__init__(classes="plan")
        self.refresh_content()

    def set_steps(self, steps: Iterable[tuple[str, bool]]) -> None:
        self.steps = list(steps)
        self.refresh_content()

    def refresh_content(self) -> None:
        rows = [
            f"[{'bold #ff7a45' if complete else '#ffffff'}]"
            f"{'[x]' if complete else '[ ]'}[/]  {escape(step)}"
            for step, complete in self.steps
        ]
        self.update("[bold #ff7a45]PLAN[/]\n" + "\n".join(rows))


class TurnSeparator(Static):
    """Separates one conversation turn from the next."""


class ProcessIndicator(Static):
    """Persistent run state that stays visible while the chat scrolls."""

    COLORS = {
        "READY": "#ffffff",
        "WORKING": "#ff7a45",
        "WAITING FOR APPROVAL": "#ff7a45",
        "FINISHED": "#ffffff",
        "FAILED": "#ff7a45",
        "STOPPED": "#ff7a45",
    }

    def __init__(self) -> None:
        self.state = "READY"
        self.detail = "Enter a task to start"
        self.progress_position = 0
        self.started_at: float | None = None
        self.elapsed_seconds = 0.0
        super().__init__(id="process-status")
        self.refresh_content()

    def set_status(self, state: str, detail: str) -> None:
        if state == "WORKING" and self.state != "WORKING":
            self.started_at = monotonic()
            self.elapsed_seconds = 0.0
            self.progress_position = 0
        elif state != "WORKING" and self.state == "WORKING":
            self.elapsed_seconds = (
                monotonic() - self.started_at if self.started_at is not None else 0.0
            )
            self.started_at = None
        self.state = state
        self.detail = detail
        self.refresh_content()

    def advance_animation(self) -> None:
        if self.state != "WORKING":
            return
        self.progress_position = (self.progress_position + 1) % 12
        self.refresh_content()

    def refresh_content(self) -> None:
        color = self.COLORS.get(self.state, "#ffffff")
        if self.state == "WORKING":
            track = ["-"] * 12
            track[self.progress_position] = ">"
            elapsed = monotonic() - self.started_at if self.started_at else 0.0
            minutes, seconds = divmod(int(elapsed), 60)
            self.update(
                f"[bold {color}]RUNNING[/]  [{color}]{''.join(track)}[/]  "
                f"{minutes:02}:{seconds:02}  {escape(self.detail)}"
            )
            return

        icon = "!" if self.state in {"WAITING FOR APPROVAL", "FAILED", "STOPPED"} else ">"
        elapsed = ""
        if self.elapsed_seconds:
            minutes, seconds = divmod(int(self.elapsed_seconds), 60)
            elapsed = f"  ({minutes:02}:{seconds:02})"
        self.update(
            f"[bold {color}]{icon} {escape(self.state)}[/]{elapsed}  "
            f"{escape(self.detail)}"
        )


class PromptTextArea(TextArea):
    """Multiline prompt: Enter sends, Shift+Enter inserts a line break."""

    BINDINGS = [
        Binding("enter", "submit_prompt", "Send message", priority=True),
        Binding("shift+enter", "insert_newline", "New line", priority=True),
    ]

    def action_submit_prompt(self) -> None:
        self.app.run_worker(
            self.app.action_submit_prompt(),
            group="prompt-submit",
            exclusive=False,
        )

    def action_insert_newline(self) -> None:
        self.insert("\n")


class ActivityFeed(VerticalScroll):
    """Append-only feed for chat messages and agent activity events."""

    async def append(self, item: Widget) -> Widget:
        await self.mount(item)
        self.scroll_end(animate=False)
        return item

    async def add_tool(
        self,
        tool_name: str,
        detail: str = "",
        *,
        status: str = "running",
    ) -> ToolActivity:
        item = ToolActivity(tool_name, detail, status=status)
        return await self.append(item)  # type: ignore[return-value]

    async def add_status(self, message: str, *, status: str = "running") -> StatusActivity:
        item = StatusActivity(message, status=status)
        return await self.append(item)  # type: ignore[return-value]

    async def add_command(
        self,
        command: str,
        *,
        status: str = "running",
        output: str = "",
    ) -> CommandActivity:
        item = CommandActivity(command, status=status, output=output)
        return await self.append(item)  # type: ignore[return-value]

    async def add_file_change(
        self,
        action: str,
        path: str,
        *,
        status: str = "done",
    ) -> FileActivity:
        item = FileActivity(action, path, status=status)
        return await self.append(item)  # type: ignore[return-value]

    async def add_plan(self, steps: Iterable[tuple[str, bool]]) -> PlanActivity:
        item = PlanActivity(steps)
        return await self.append(item)  # type: ignore[return-value]


class CodingAgentApp(App):
    """A compact, Claude Code-inspired terminal interface."""

    TITLE = "Coding Agent"
    SUB_TITLE = "~/git/terminal"

    CSS = """
    Screen {
        background: #000000;
        color: #ffffff;
        border: none;
        scrollbar-color: #ff7a45;
        scrollbar-color-hover: #ffffff;
        scrollbar-color-active: #ff7a45;
        scrollbar-background: #000000;
    }

    #topbar {
        height: 1;
        background: #000000;
        color: #ffffff;
        border-bottom: solid #ff7a45;
        padding: 0 2;
        content-align: left middle;
        text-style: bold;
    }

    #brand-mark {
        width: 4;
        color: #ff7a45;
        text-style: bold;
        content-align: left middle;
    }

    #app-title {
        color: #ffffff;
        text-style: bold;
        content-align: left middle;
    }

    #welcome {
        height: 4;
        padding: 0 2;
        background: #000000;
    }

    #process-status {
        height: 2;
        padding: 0 2;
        border-top: solid #555555;
        border-bottom: solid #555555;
        color: #ffffff;
        content-align: left middle;
    }

    #welcome-logo {
        width: 6;
        height: 3;
        margin: 0 1 0 0;
        border: round #ff7a45;
        background: #ff7a45;
        color: #000000;
        text-style: bold;
        content-align: center middle;
    }

    #identity { width: 1fr; padding: 0 1; }
    #identity-title { color: #ffffff; text-style: bold; }
    #identity-meta { color: #ffffff; }
    #identity-path { color: #ff7a45; }

    #conversation {
        height: 1fr;
        padding: 1 3;
        scrollbar-size: 1 1;
    }

    .user-message {
        margin: 1 0;
        padding: 0 1;
        color: #ffffff;
    }
    .accent-orange { border-left: tall #ff7a45; }
    .accent-cyan { border-left: tall #30d5c8; }
    .accent-violet { border-left: tall #b69cff; }
    .accent-blue { border-left: tall #78aef7; }
    .agent-message {
        margin: 1 0;
        padding: 0 1;
        border-left: tall #ffffff;
        color: #ffffff;
    }
    .turn-separator {
        height: 1;
        margin: 1 0;
        border-top: solid #808080;
    }
    .tool {
        margin: 0 0 0 3;
        padding: 0 1;
        border-left: tall #ff7a45;
        color: #ff7a45;
    }
    .status-activity {
        margin: 0 0 1 3;
        padding: 0 1;
        border-left: tall #ff7a45;
        color: #ffffff;
    }
    .plan {
        margin: 1 0;
        padding: 0 1;
        border-left: tall #ff7a45;
        color: #ffffff;
    }
    .success { color: #ffffff; }
    .error { color: #ff7a45; }

    #input-area {
        height: auto;
        min-height: 6;
        max-height: 8;
        padding: 0 1;
        border-top: solid #ff7a45;
        background: #000000;
    }

    #prompt-line { height: 4; }
    #stop-button {
        height: 3;
        min-width: 10;
        margin: 0 1;
        background: #000000;
        color: #ff7a45;
        border: solid #ff7a45;
        text-style: bold;
    }
    #stop-button.hidden { display: none; }
    #prompt-symbol {
        width: 2;
        color: #ff7a45;
        text-style: bold;
        content-align: left top;
    }

    #prompt {
        height: 4;
        border: none;
        background: #000000;
        color: #ffffff;
        padding: 0;
        scrollbar-size: 0 0;
    }

    #prompt:focus { border: none; }

    #approval-actions {
        height: 3;
        padding: 0 2;
        align-horizontal: right;
    }
    #approval-actions.hidden { display: none; }
    #approval-hint { color: #ffffff; width: 1fr; content-align: left middle; }
    #approve-button {
        background: #ff7a45;
        color: #000000;
        text-style: bold;
        min-width: 12;
        margin-right: 1;
    }
    #reject-button {
        background: #000000;
        color: #ffffff;
        border: solid #ff7a45;
        min-width: 12;
    }

    #statusbar {
        height: 1;
        padding: 0 2;
        color: #ffffff;
        border-top: solid #ff7a45;
        background: #000000;
    }

    #mode { color: #ff7a45; }
    #mode-hint { color: #ffffff; }
    #location { color: #ff7a45; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_screen", "Clear"),
        ("shift+tab", "toggle_plan", "Cycle mode"),
        ("ctrl+enter", "submit_prompt", "Send message"),
    ]

    def __init__(self, workspace: Path | None = None) -> None:
        super().__init__()
        self.plan_mode = True
        self.agent: CodingAgent | None = None
        self.workspace = (workspace or Path.cwd()).expanduser().resolve()
        self.history_path = self.workspace / ".coding_agent_history.sqlite3"
        self.current_chat_id: int | None = None
        self.pending_approval: list[dict] | None = None
        self.active_agent_task = None
        self.active_run_activity: StatusActivity | None = None
        self.agent_message: Message | None = None
        self.plan_activity: PlanActivity | None = None
        self.is_busy = False
        self.activity_phase = "PLANNING"
        self.tool_activities: dict[str, ToolActivity] = {}
        self.user_accents = [
            ("orange", "#ff7a45"),
            ("cyan", "#30d5c8"),
            ("violet", "#b69cff"),
            ("blue", "#78aef7"),
        ]

    def compose(self) -> ComposeResult:
        yield Static("Coding Agent", id="topbar")
        yield Horizontal(
            Static(">_", id="welcome-logo"),
            Vertical(
                Static("v0.1.0", id="identity-title"),
                Static("Ready for your next task", id="identity-meta"),
                Static(str(self.workspace), id="identity-path"),
                id="identity",
            ),
            id="welcome",
        )
        yield ActivityFeed(id="conversation")
        yield ProcessIndicator()
        yield Vertical(
            Horizontal(
                Static(">", id="prompt-symbol"),
                PromptTextArea(
                    placeholder="Ask the agent to help with a task... (Shift+Enter for a new line)",
                    id="prompt",
                    soft_wrap=True,
                    show_line_numbers=False,
                ),
                Button("Stop", id="stop-button", classes="hidden"),
                id="prompt-line",
            ),
            Horizontal(
                Static("Review the requested action", id="approval-hint"),
                Button("Approve", id="approve-button", variant="primary"),
                Button("Reject", id="reject-button", variant="default"),
                id="approval-actions",
                classes="hidden",
            ),
            id="input-area",
        )
        yield Horizontal(
            Static("PLAN MODE", id="mode"),
            Static("  (Enter to send | Shift+Enter for new line)", id="mode-hint"),
            Static("In Terminal", id="location"),
            id="statusbar",
        )

    async def on_mount(self) -> None:
        self.query_one("#prompt", TextArea).focus()
        self.set_interval(0.18, self._tick_process_indicator)
        await self._restore_recent_chats()

    def _open_history(self) -> sqlite3.Connection:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.history_path)
        connection.execute(
            """CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )
        return connection

    async def _restore_recent_chats(self) -> None:
        with self._open_history() as connection:
            rows = connection.execute(
                "SELECT user_message, assistant_message FROM chats ORDER BY id DESC LIMIT 10"
            ).fetchall()
        conversation = self.query_one("#conversation", ActivityFeed)
        for user_message, assistant_message in reversed(rows):
            if conversation.children:
                await conversation.append(TurnSeparator(classes="turn-separator"))
            accent, color = choice(self.user_accents)
            user = Message("user", user_message, color=color)
            user.add_class(f"accent-{accent}")
            await conversation.append(user)
            if assistant_message:
                await conversation.append(Message("agent", assistant_message))

    def _save_user_chat(self, message: str) -> None:
        with self._open_history() as connection:
            cursor = connection.execute(
                "INSERT INTO chats (user_message, created_at) VALUES (?, ?)",
                (message, datetime.now(timezone.utc).isoformat()),
            )
            self.current_chat_id = cursor.lastrowid
            connection.execute(
                "DELETE FROM chats WHERE id NOT IN (SELECT id FROM chats ORDER BY id DESC LIMIT 10)"
            )

    def _save_assistant_chat(self, message: str) -> None:
        if self.current_chat_id is None:
            return
        with self._open_history() as connection:
            connection.execute(
                "UPDATE chats SET assistant_message = ? WHERE id = ?",
                (message, self.current_chat_id),
            )


    def _tick_process_indicator(self) -> None:
        if self.is_busy:
            self.query_one(ProcessIndicator).advance_animation()

    async def action_submit_prompt(self) -> None:
        prompt = self.query_one("#prompt", TextArea)
        user_input = prompt.text.strip()
        if not user_input:
            return

        if self.pending_approval is not None:
            command, _, reason = user_input.partition(" ")
            decision = command.lower().removeprefix("/")
            if decision == "stop":
                prompt.load_text("")
                await self.action_stop()
                return
            if decision not in {"approve", "accept", "reject", "decline"}:
                prompt.load_text("")
                self.pending_approval = None
                self.agent = None
                self.query_one("#approval-actions").add_class("hidden")
                self.query_one("#stop-button").add_class("hidden")
                if self.active_run_activity is not None:
                    self.active_run_activity.set_status("stopped", "Canceled by your new request")
                for item in self.tool_activities.values():
                    if item.status in {"running", "pending"}:
                        item.set_status("stopped", "Canceled by your new request")
                await self.add_status(
                    "Your new message canceled the unapproved action. Starting the new request.",
                    status="stopped",
                )
                await self.add_user_message(user_input)
                self._save_user_chat(user_input)
                await self.run_agent(prompt=user_input)
                return
            prompt.load_text("")
            await self.resolve_approval(
                approved=decision in {"approve", "accept"},
                reason=reason.strip(),
            )
            return

        if self.is_busy:
            if user_input.lower() in {"/stop", "stop"}:
                prompt.load_text("")
                await self.action_stop()
            else:
                prompt.load_text("")
                await self.add_user_message(user_input)
                self._save_user_chat(user_input)
                await self.add_status(
                    "New request received. Stopping the current run before handling it."
                )
                await self._cancel_active_run()
                await self.run_agent(prompt=user_input)
            return

        prompt.load_text("")
        await self.add_user_message(user_input)
        self._save_user_chat(user_input)
        await self.run_agent(prompt=user_input)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-button":
            await self.action_stop()
            return
        if self.pending_approval is None:
            return
        if event.button.id == "approve-button":
            await self.resolve_approval(approved=True)
        elif event.button.id == "reject-button":
            await self.resolve_approval(approved=False)

    async def action_stop(self) -> None:
        if self.pending_approval is not None:
            self.pending_approval = None
            self.agent = None
            self.query_one("#approval-actions").add_class("hidden")
            self.query_one("#stop-button").add_class("hidden")
            if self.active_run_activity is not None:
                self.active_run_activity.set_status("stopped", "Stopped by you")
            for item in self.tool_activities.values():
                if item.status in {"running", "pending"}:
                    item.set_status("stopped", "Stopped by you; action not approved")
            self.query_one(ProcessIndicator).set_status("STOPPED", "Stopped by you")
            await self.add_status(
                "Stopped. The pending action was declined; completed file changes were kept.",
                status="stopped",
            )
            return
        task = self.active_agent_task
        if task is not None and not task.done():
            await self.add_status("Stopping the agent and any active command.")
            task.cancel()

    async def _cancel_active_run(self) -> None:
        task = self.active_agent_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def resolve_approval(self, *, approved: bool, reason: str = "") -> None:
        requests = self.pending_approval
        if requests is None:
            return
        self.pending_approval = None
        self.query_one("#approval-actions").add_class("hidden")
        self.query_one(ProcessIndicator).set_status("WORKING", "Resuming after your decision")
        for item in self.tool_activities.values():
            if item.status == "pending":
                item.set_status("running", "Resuming after approval")
        if approved:
            decisions = [{"type": "approve"} for _ in requests]
            await self.add_user_message("Approved the requested action")
        else:
            message = reason or "The user rejected this action. Do not retry unless asked."
            decisions = [{"type": "reject", "message": message} for _ in requests]
            await self.add_user_message("Rejected the requested action")
        await self.run_agent(decisions=decisions)

    async def run_agent(
        self,
        *,
        prompt: str | None = None,
        decisions: list[dict] | None = None,
    ) -> None:
        self.active_agent_task = asyncio.current_task()
        self.is_busy = True
        self.query_one("#stop-button").remove_class("hidden")
        activity = await self.add_status("Connecting to the coding agent")
        self.active_run_activity = activity
        self.agent_message = None
        self.activity_phase = "PLANNING"
        self.tool_activities = {}
        process_status = self.query_one(ProcessIndicator)
        process_status.set_status("WORKING", "Planning and inspecting the project")
        self._tick_process_indicator()
        try:
            if self.agent is None:
                self.agent = CodingAgent(self.workspace)
            activity.set_status("running", "Working through the request")
            async for event in self.agent.stream(prompt, decisions=decisions):
                kind = event.get("type")
                if kind == "status":
                    activity.set_status("running", event["message"])
                    message = event["message"].lower()
                    if "delegat" in message:
                        self.activity_phase = "DELEGATING"
                    elif "review" in message:
                        self.activity_phase = "REVIEWING"
                    elif "check" in message or "command" in message:
                        self.activity_phase = "CHECKING"
                    elif "plan" in message:
                        self.activity_phase = "PLANNING"
                    else:
                        self.activity_phase = "WORKING"
                    process_status.set_status("WORKING", event["message"])
                elif kind == "token":
                    if self.agent_message is None:
                        self.agent_message = Message("agent", "")
                        await self.query_one("#conversation", ActivityFeed).append(
                            self.agent_message
                        )
                    self.agent_message.append_text(event["text"])
                    self._save_assistant_chat(self.agent_message.message)
                elif kind == "tool":
                    raw_tool_name = event.get("name", "tool")
                    tool_name = {
                        "ls": "List project files",
                        "read_file": "Read file",
                        "glob": "Find files",
                        "grep": "Search files",
                        "write_file": "Write file",
                        "edit_file": "Edit file",
                        "delete_file": "Delete file",
                        "run_command": "Run approved command",
                        "tavily_search": "Search the web",
                    }.get(raw_tool_name, raw_tool_name)
                    tool_status = event.get("status", "running")
                    if tool_status == "running":
                        operation = event.get("operation")
                        friendly = {
                            "ls": "Inspecting project folders",
                            "read_file": "Reading a project file",
                            "glob": "Finding project files",
                            "grep": "Searching project files",
                            "write_file": "Preparing a file change",
                            "edit_file": "Preparing a file edit",
                            "delete_file": "Preparing a file deletion",
                            "run_command": (
                                "Installing approved project packages"
                                if operation == "package-install"
                                else "Running an approved project check"
                            ),
                            "tavily_search": "Searching the web with Tavily",
                        }.get(raw_tool_name, f"Working: {tool_name}")
                        process_status.set_status("WORKING", friendly)
                    else:
                        process_status.set_status(
                            "WORKING", f"Finished {tool_name}; continuing the task"
                        )
                    call_id = event.get("call_id")
                    if call_id and call_id in self.tool_activities:
                        self.tool_activities[call_id].set_status(
                            tool_status, event.get("detail", "")
                        )
                    else:
                        item = await self.add_tool(
                            tool_name,
                            event.get("detail", ""),
                            status=tool_status,
                        )
                        if call_id:
                            self.tool_activities[call_id] = item
                elif kind == "plan":
                    steps = event.get("steps", [])
                    if self.plan_activity is None:
                        self.plan_activity = await self.add_plan(steps)
                    else:
                        self.plan_activity.set_steps(steps)
                elif kind == "approval":
                    self.pending_approval = event.get("requests", [])
                    self.query_one("#approval-actions").remove_class("hidden")
                    activity.set_status("pending", "Waiting for your approval")
                    for item in self.tool_activities.values():
                        if item.status == "running":
                            item.set_status("pending", "Waiting for your approval")
                    process_status.set_status(
                        "WAITING FOR APPROVAL",
                        "Review the requested action below",
                    )
                    command_review = False
                    package_review = False
                    for request in self.pending_approval:
                        name = request.get("name", "file operation")
                        args = request.get("args", {})
                        if name == "run_command":
                            command_review = True
                            command = str(args.get("command", ""))
                            validated, _error = validate_command(command)
                            display_command = shlex.join(validated) if validated else command
                            workspace_path_note = (
                                " (virtual /workspace path resolved inside the selected project)"
                                if "/workspace/" in command or command.endswith("/workspace")
                                else ""
                            )
                            package_review = any(
                                command.startswith(prefix)
                                for prefix in (
                                    "uv add", "uv sync", "uv venv", "uv pip install",
                                    "npm install", "npm i",
                                    "pnpm add", "pnpm install", "cargo add", "go get",
                                    "dotnet add package", "composer require", "composer install",
                                )
                            )
                            action_type = "Package command" if package_review else "Command"
                            notes = []
                            if package_review:
                                notes.append("This may update dependency files or the project environment.")
                            if ".py" in display_command.lower():
                                notes.append("A Python script can run project code and modify files.")
                            notes.append("It runs with your Windows account permissions.")
                            await self.add_status(
                                "Safety precheck passed: allowlisted command, executed without a shell, "
                                f"with command-line paths restricted to this project. {action_type} to run: "
                                f"{display_command}{workspace_path_note}. {' '.join(notes)}",
                                status="pending",
                            )
                        else:
                            await self.add_status(
                                f"Approval needed: {name} {args}", status="pending"
                            )
                    await self.add_status(
                        "Use Approve, /approve, or accept to run/apply; use Reject, /reject, or decline to cancel."
                        if command_review
                        else "Use Approve, /approve, or accept to apply; use Reject, /reject, or decline to cancel.",
                        status="pending",
                    )
                elif kind == "complete":
                    activity.set_status("done", "Request complete")
                    for item in self.tool_activities.values():
                        if item.status == "running":
                            item.set_status("done", "Finished")
                    process_status.set_status("FINISHED", "Task complete - agent response is ready")
                elif kind == "incomplete":
                    activity.set_status("failed", "No implementation changes made")
                    process_status.set_status("FAILED", "Implementation did not change any files")
                    await self.add_status(str(event.get("message", "Implementation did not finish.")), status="failed")
        except asyncio.CancelledError:
            self.agent = None
            activity.set_status("stopped", "Stopped by you")
            for item in self.tool_activities.values():
                if item.status in {"running", "pending"}:
                    item.set_status("stopped", "Stopped by you")
            process_status.set_status("STOPPED", "Stopped by you")
            await self.add_status(
                "The active response and command were stopped. Completed file changes were kept.",
                status="stopped",
            )
        except Exception as error:
            activity.set_status("failed", "Request failed")
            for item in self.tool_activities.values():
                if item.status in {"running", "pending"}:
                    item.set_status("failed", "Stopped because the request failed")
            process_status.set_status("FAILED", "Task stopped - see the error message below")
            await self.add_agent_message(f"I couldn't get a response: {error}")
        finally:
            self.is_busy = False
            if self.active_agent_task is asyncio.current_task():
                self.active_agent_task = None
            if self.pending_approval is None:
                self.query_one("#stop-button").add_class("hidden")

    async def add_user_message(self, message: str) -> None:
        conversation = self.query_one("#conversation", ActivityFeed)
        if conversation.children:
            await conversation.append(TurnSeparator(classes="turn-separator"))
        accent, color = choice(self.user_accents)
        item = Message("user", message, color=color)
        item.add_class(f"accent-{accent}")
        await conversation.append(item)

    async def add_agent_message(self, message: str) -> None:
        conversation = self.query_one("#conversation", ActivityFeed)
        await conversation.append(Message("agent", message))
        self._save_assistant_chat(message)

    async def add_tool(self, tool_name: str, detail: str = "", *, status: str = "running") -> ToolActivity:
        """Add a tool event and return it so the agent can later call set_status."""
        conversation = self.query_one("#conversation", ActivityFeed)
        return await conversation.add_tool(tool_name, detail, status=status)

    async def add_status(self, message: str, *, status: str = "running") -> StatusActivity:
        """Add a progress update, such as planning or inspecting the project."""
        return await self.query_one("#conversation", ActivityFeed).add_status(
            message, status=status
        )

    async def add_command(
        self,
        command: str,
        *,
        status: str = "running",
        output: str = "",
    ) -> CommandActivity:
        """Display a shell command; update its returned widget when it exits."""
        return await self.query_one("#conversation", ActivityFeed).add_command(
            command, status=status, output=output
        )

    async def add_file_change(
        self,
        action: str,
        path: str,
        *,
        status: str = "done",
    ) -> FileActivity:
        """Display a file read or edit event in the activity feed."""
        return await self.query_one("#conversation", ActivityFeed).add_file_change(
            action, path, status=status
        )

    async def add_plan(self, steps: Iterable[tuple[str, bool]]) -> PlanActivity:
        """Display plan steps as (description, completed) pairs."""
        return await self.query_one("#conversation", ActivityFeed).add_plan(steps)

    def action_toggle_plan(self) -> None:
        self.plan_mode = not self.plan_mode
        label = "PLAN MODE" if self.plan_mode else "EXECUTE MODE"
        self.query_one("#mode", Static).update(label)

    def action_clear_screen(self) -> None:
        self.query_one("#conversation", ActivityFeed).remove_children()
        self.plan_activity = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Coding Agent in a project.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project folder the agent may inspect and edit (default: current folder).",
    )
    app = CodingAgentApp(workspace=parser.parse_args().workspace)
    app.run()
