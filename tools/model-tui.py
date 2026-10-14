#!/usr/bin/env python3
"""model-tui.py — elegant model manager for the 2-node GX10 cluster.

Lists the available models (deepseek / glm / qwen38), highlights the active one,
and drives switch / stop / restart through the boot orchestration on the head
node. Under the hood every action shells out to `/usr/local/bin/swap-model.sh`
so the TUI is a convenience layer — the CLI remains fully usable on its own.

Runs on the HEAD node (leo) and needs root (swap-model.sh writes /etc/vllm/current_model,
/etc/motd and sudo -u wfoster into the rootless podman containers).

Model metadata is read from /usr/local/bin/swap-model.sh's known lanes; the
served model name comes from `swap-model.sh --health`.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, LoadingIndicator, Log
from textual.screen import ModalScreen

SWAP = "/usr/local/bin/swap-model.sh"

# lane_key -> (display name, description). Keep in sync with swap-model.sh.
MODELS = [
    ("deepseek", "DeepSeek-V4-Flash-Vision", "vLLM (b12x), 1M ctx, vision"),
    ("glm", "GLM-5.3-Flash", "vLLM (dflash2), 1M ctx"),
    ("qwen38", "Qwen3.8-Flash-Next-Uncensored", "SGLang (tp2), 262K ctx"),
]

# lane_key -> rootless podman container name (for `podman logs`). Sync with swap-model.sh.
CONTAINERS = {
    "deepseek": "b12x-vllm-node",
    "glm": "vllm_glm53",
    "qwen38": "qwen38fn-sglang",
}


def run(args: list[str], timeout: int = 1200, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a swap-model.sh subcommand. `timeout` is large because swaps block on load."""
    cmd = [SWAP, *args]
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, check=False)


STATE_FILE = "/etc/vllm/current_model"


def active_model() -> str | None:
    """Return the currently-deployed model lane key.

    Reads `/etc/vllm/current_model` (the same file `swap-model.sh` uses for the
    default = last-used model) directly, no subprocess. Falls back to a `--health`
    probe only if the file is unreadable. The file is what the boot ensure path
    guarantees is serving, so it is authoritative for "active".
    """
    try:
        with open(STATE_FILE) as fh:
            return fh.read().strip() or None
    except OSError:
        pass
    try:
        p = run(["--health"], timeout=10)
        for line in p.stdout.splitlines():
            if line.startswith("current_model:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


class ActionScreen(ModalScreen[str | None]):
    """Sub-menu of available actions for a selected model (arrow-select + Enter)."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, model_label: str, actions: list[str]):
        super().__init__()
        self.model_label = model_label
        self.actions = actions

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Label(f"  {self.model_label}", id="confirm-msg")
            yield Label("  (↑/↓ select, Enter choose, Esc cancel)", id="confirm-sub")
            yield ListView(*[ListItem(Label(a), id=a) for a in self.actions + ["cancel"]], id="action-list")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#action-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.id)  # type: ignore[attr-defined]


class ConfirmScreen(ModalScreen[bool]):
    """Yes / No confirmation dialogue (arrow-select + Enter)."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, message: str, danger: bool = False):
        super().__init__()
        self.message = message
        self.danger = danger

    def action_cancel(self) -> None:
        self.dismiss(False)

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Label("⚠ " + self.message, id="confirm-msg")
            yield ListView(
                ListItem(Label("No"), id="no"),
                ListItem(Label("Yes"), id="yes"),
                id="action-list",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#action-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.id == "yes")  # type: ignore[attr-defined]


class ProgressScreen(ModalScreen[None]):
    """Informative process dialogue shown while a swap/stop/restart runs.

    Tails the lane container's `podman logs -f` into a collapsible window so the
    user can watch the model actually load (weights, CUDA graph capture, server
    started). `l` toggles the log pane; anything else scrolls the log.
    """

    BINDINGS = [("escape", "cancel", "Cancel"), ("l", "toggle_log", "Show/Hide Log")]

    def __init__(self, heading: str, container: str | None):
        super().__init__()
        self.heading = heading
        self.container = container
        self._log_proc: subprocess.Popen | None = None
        self._log_on = False

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="progress"):
            yield LoadingIndicator()
            yield Label(self.heading, id="progress-msg")
            with Vertical(id="log-pane"):
                yield Log(id="log")
        yield Footer()

    def action_toggle_log(self) -> None:
        self._log_on = not self._log_on
        self.query_one("#log-pane").display = self._log_on
        if self._log_on and not self._log_proc and self.container:
            self._start_tail()

    def _start_tail(self) -> None:
        # `sudo -u wfoster` because the serving containers are rootless podman.
        if os.geteuid() == 0:
            cmd = ["sudo", "-u", "wfoster", "podman", "logs", "-f", self.container]
        else:
            cmd = ["podman", "logs", "-f", self.container]
        self._log_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._log_task = asyncio.create_task(self._pump_log())

    async def _pump_log(self) -> None:
        proc = self._log_proc
        if proc is None or proc.stdout is None:
            return
        log = self.query_one("#log", Log)
        # Read the pipe on a worker thread so a stalled `podman logs -f` never
        # blocks the event loop; the read is what yields, not us.
        reader = proc.stdout
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, reader.readline)
            if not line:
                break
            log.write_line(line.rstrip())
            if len(log._lines) > 300:  # type: ignore[attr-defined]
                del log._lines[: len(log._lines) - 300]  # type: ignore[attr-defined]

    def on_unmount(self) -> None:
        if self._log_proc:
            self._log_proc.kill()

    def tick(self) -> None:
        pass


class ModelTui(App):
    TITLE = "GX10 Model Manager"
    BINDINGS = [("q", "quit", "Quit")]
    CSS = """
    #list-screen {
        padding: 1 2;
    }
    #title {
        text-style: bold;
        margin-bottom: 1;
    }
    ListView {
        height: auto;
        margin-bottom: 1;
    }
    ListView > ListItem {
        padding: 0 1;
    }
    ListView > ListItem.--highlight {
        background: $accent;
        color: $text;
    }
    #hint {
        color: $text-muted;
    }
    #confirm {
        width: 60;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    #confirm-msg {
        text-style: bold;
        margin-bottom: 1;
    }
    #progress {
        width: 80;
        padding: 2 3;
        border: thick $primary;
        background: $surface;
    }
    #log-pane {
        height: 60%;
        margin-top: 1;
        display: none;
        border: solid $primary;
    }
    #log {
        height: 100%;
        background: $surface;
    }
    #progress-msg {
        text-style: bold;
        margin-top: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.active: str | None = None
        self.selected: str = MODELS[0][0]
        self._suppress_next_select = False

    def action_quit(self) -> None:
        self.exit()

    def compose(self) -> ComposeResult:
        # Compute active at compose time so the rows are built in compose() and
        # actually render. (In Textual 8.2.8, ListItems appended in on_mount()
        # do not paint.)
        try:
            self.active = active_model()
        except Exception:
            self.active = None
        yield Header()
        yield Label("GX10 Models (↑/↓ select, Enter action, q quit)", id="title")
        yield ListView(*[self._item(m) for m in MODELS], id="models")
        yield Label("", id="hint")
        yield Footer()

    def _item(self, m: tuple[str, str, str]) -> ListItem:
        key, name, desc = m
        is_active = key == self.active
        marker = "●" if is_active else "○"
        active_tag = "  [b]ACTIVE[/b]" if is_active else ""
        return ListItem(Label(f"{marker}  {name}{active_tag}  —  {desc}"), id=key)

    def on_mount(self) -> None:
        # Focus the list and pre-highlight the active model row.
        lv = self.query_one("#models", ListView)
        if self.active:
            idx = next((i for i, (k, _, _) in enumerate(MODELS) if k == self.active), None)
            if idx is not None:
                lv.index = idx
        lv.focus()

    def refresh_list(self) -> None:
        # Rebuild via recompose() so the rows actually re-render after a swap.
        self.recompose()

    def _rebuild_list(self) -> None:
        self.refresh_list()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Only open the action menu when no modal is on top (the main list is the
        # active screen). A trailing keypress after a modal dismisses can bubble
        # here; ignore it so we never re-open a menu for a stale selection.
        if self._suppress_next_select:
            self._suppress_next_select = False
            return
        if isinstance(self.screen, (ActionScreen, ConfirmScreen, ProgressScreen)):
            return
        key = event.item.id  # type: ignore[attr-defined]
        self.selected = key
        self._open_action_menu()

    def _open_action_menu(self) -> None:
        key = self.selected
        label = next((f"{n} ({key})" for k, n, _ in MODELS if k == key), key)
        if key == self.active:
            actions = ["restart", "stop"]
        else:
            actions = ["switch"]
        self.push_screen(ActionScreen(label, actions), self._on_action_chosen)

    def _on_action_chosen(self, choice: str | None) -> None:
        if not choice or choice == "cancel":
            # Swallow the trailing Enter that bubbles to the main list after the
            # menu dismisses; otherwise it re-opens the menu immediately.
            self._suppress_next_select = True
            return
        key = self.selected
        msg = {
            "switch": f"Switch to '{key}'? This stops the current model and loads {key} (several minutes).",
            "restart": f"Restart the active model '{key}'? Clean stop, then reload. This is a few minutes of downtime.",
            "stop": f"Stop the active model '{key}'? :8000 will be freed; boot will bring back your last-used model.",
        }[choice]
        # stop has no destructive-reload risk but is a deliberate downtime; all are danger-ish.
        self.push_screen(ConfirmScreen(msg, danger=(choice != "switch")), lambda c: self._on_action_confirmed(c, key, choice))

    def _on_action_confirmed(self, confirmed: bool, key: str, choice: str) -> None:
        if not confirmed:
            # Swallow the trailing Enter that bubbles to the main list after the
            # confirm dismisses; otherwise it re-opens the action menu.
            self._suppress_next_select = True
            return
        args = {"switch": ["--swap", key], "restart": ["restart"], "stop": ["--stop"]}[choice]
        self._run_action(args, f"{choice.capitalize()} {key}...", CONTAINERS.get(key))

    def _run_action(self, args: list[str], heading: str, container: str | None = None) -> None:
        self.push_screen(ProgressScreen(heading, container))
        # Run the model action off the event loop so the UI (LoadingIndicator)
        # stays live during the multi-minute load. asyncio.to_thread keeps the
        # blocking subprocess off the loop; everything else stays on the app
        # thread (no cross-thread UI calls needed).
        self.run_worker(self._action_worker(args, heading), exclusive=True)

    async def _action_worker(self, args: list[str], heading: str):
        proc: subprocess.CompletedProcess | None = None
        err: str | None = None
        try:
            proc = await asyncio.to_thread(run, args)
            self.active = active_model()
        except subprocess.TimeoutExpired:
            err = f"timed out after {args}; the model is still loading. Run 'swap-model.sh --health' to check."
        except Exception as e:  # noqa: BLE001
            err = f"error: {e}"
        finally:
            self._finish_action(proc, err, args)

    def _finish_action(self, proc: subprocess.CompletedProcess | None, err: str | None, args: list[str]) -> None:
        self.pop_screen()
        self._rebuild_list()
        if err:
            status = "FAILED"
            tail = err
        elif proc is not None:
            status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
            tail = (proc.stdout.strip() + "\n" + proc.stderr.strip()).strip()[-500:]
        else:
            status = "?"
            tail = ""
        self.query_one("#hint", Label).update(f"{status}: swap-model.sh {shlex.join(args)}\n{tail}")
        self.refresh_bindings()


if __name__ == "__main__":
    if Path(SWAP).exists():
        ModelTui().run()
    else:
        print(f"error: {SWAP} not found. Run on the head node as root.", file=sys.stderr)
        sys.exit(1)
