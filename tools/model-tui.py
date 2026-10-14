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

import shlex
import subprocess
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, LoadingIndicator, Static
from textual.screen import ModalScreen

SWAP = "/usr/local/bin/swap-model.sh"

# lane_key -> (display name, description). Keep in sync with swap-model.sh.
MODELS = [
    ("deepseek", "DeepSeek-V4-Flash-Vision", "vLLM (b12x), 1M ctx, vision"),
    ("glm", "GLM-5.3-Flash", "vLLM (dflash2), 262K ctx"),
    ("qwen38", "Qwen3.8-Flash-Next-Uncensored", "SGLang (tp2), 262K ctx"),
]


def run(args: list[str], timeout: int = 1200, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a swap-model.sh subcommand. `timeout` is large because swaps block on load."""
    cmd = [SWAP, *args]
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, check=False)


def active_model() -> str | None:
    """Ask swap-model.sh --health for current_model. Returns the lane key or None."""
    try:
        p = run(["--health"], timeout=20)
        for line in p.stdout.splitlines():
            if line.startswith("current_model:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


class ActionScreen(ModalScreen[str | None]):
    """Sub-menu of available actions for a selected model."""

    def __init__(self, model_label: str, actions: list[str]):
        super().__init__()
        self.model_label = model_label
        self.actions = actions

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Label(f"  {self.model_label}", id="confirm-msg")
            yield Label("", id="confirm-sub")
            for a in self.actions:
                yield Button(a, id=a)
            yield Button("Cancel", id="cancel", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)


class ConfirmScreen(ModalScreen[bool]):
    """Yes / No confirmation dialogue."""

    def __init__(self, message: str, danger: bool = False):
        super().__init__()
        self.message = message
        self.danger = danger

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Label("⚠ " + self.message, id="confirm-msg")
            with Horizontal(id="confirm-btns"):
                yield Button("No", id="no", variant="primary")
                yield Button("Yes", id="yes", variant="error" if self.danger else "success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class ProgressScreen(ModalScreen[None]):
    """Informative process dialogue shown while a swap/stop/restart runs."""

    def __init__(self, heading: str):
        super().__init__()
        self.heading = heading

    def compose(self) -> ComposeResult:
        with Vertical(id="progress"):
            yield LoadingIndicator()
            yield Label(self.heading, id="progress-msg")
            yield Label("", id="progress-log")

    def on_mount(self) -> None:
        self.set_interval(1.0, self.tick)

    def tick(self) -> None:
        # cheap liveness: re-pull the container state summary from the tail of the test
        # log is overkill; instead just reassure the user we are alive.
        self.query_one("#progress-log", Label).update(self.heading)


class ModelList(ListView):
    """ListView with an extra marker column for the active model."""


class ModelTui(App):
    TITLE = "GX10 Model Manager"
    CSS = """
    Screen {
        align: center middle;
    }
    #list-screen {
        width: 100%;
        height: 100%;
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
    ListItem {
        padding: 1 2;
    }
    ListItem.--highlight {
        background: $accent;
    }
    #hint {
        color: $text-muted;
    }
    #confirm {
        width: 60;
        padding: 2 3;
        border: thick $primary;
        background: $surface;
    }
    #confirm-msg {
        text-style: bold;
        margin-bottom: 2;
    }
    #confirm-btns {
        height: auto;
    }
    #progress {
        width: 80;
        padding: 2 3;
        border: thick $primary;
        background: $surface;
        align: center middle;
    }
    #progress-msg {
        text-style: bold;
        margin-top: 1;
    }
    #progress-log {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.active: str | None = None
        self.selected: str = MODELS[0][0]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="list-screen"):
            yield Label("GX10 Models (↑/↓ select, Enter action, q quit)", id="title")
            yield ModelList(*[self._item(m) for m in MODELS], id="models")
            yield Label("", id="hint")
        yield Footer()

    def _item(self, m: tuple[str, str, str]) -> ListItem:
        key, name, desc = m
        is_active = key == self.active
        marker = "●" if is_active else " "
        return ListItem(
            Horizontal(
                Label(f"{marker}  {name}", classes="name"),
                Label("  ACTIVE" if is_active else "        ", classes="active"),
                Label(f"  {desc}", classes="desc"),
            ),
            id=key,
        )

    def on_mount(self) -> None:
        self.active = active_model()
        self.refresh_list()
        self.query_one("#models", ListView).focus()

    def refresh_list(self) -> None:
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#models", ListView)
        lv.clear()
        for m in MODELS:
            lv.append(self._item(m))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
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
            return
        args = {"switch": ["--swap", key], "restart": ["restart"], "stop": ["--stop"]}[choice]
        self._run_action(args, f"{choice.capitalize()} {key}...")

    def _run_action(self, args: list[str], heading: str) -> None:
        self.push_screen(ProgressScreen(heading))
        proc: subprocess.CompletedProcess | None = None
        err = None
        try:
            proc = run(args)
            self.active = active_model()
        except subprocess.TimeoutExpired:
            err = f"timed out after {args}; the model is still loading. Run 'swap-model.sh --health' to check."
        except Exception as e:  # noqa: BLE001
            err = f"error: {e}"
        finally:
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
