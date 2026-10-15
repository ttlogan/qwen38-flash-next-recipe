"""Regression check for the GX10 model-manager TUI Esc-during-load crash.

Bug: pressing Esc on the ProgressScreen dismisses (pops) it, but the still
-running swap subprocess later calls _finish_action -> pop_screen() again on the
base screen, raising Textual's `ScreenStackError: Can't pop screen`. Fix guards
the pop with `isinstance(self.screen, ProgressScreen)`.

Run from the leo venv (which has textual), next to model-tui.py:
    /home/wfoster/model-tui/venv/bin/python test_model_tui_esc.py
Exits non-zero on regression.
"""
import asyncio
import importlib.util
import time
from pathlib import Path

HERE = Path(__file__).parent


def load_app():
    spec = importlib.util.spec_from_file_location("mt", str(HERE / "model-tui.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.active_model = lambda: "qwen38"

    class FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    m.run = lambda args, *a, **k: (time.sleep(2.0), FakeProc())[1]
    return m


async def drive() -> None:
    m = load_app()
    app = m.ModelTui()
    crashed: list[Exception] = []

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("enter")  # main list -> ActionScreen
        await pilot.pause(0.4)
        await pilot.press("enter")  # action -> ConfirmScreen
        await pilot.pause(0.4)
        await pilot.press("down")  # -> Yes
        await pilot.pause(0.2)
        await pilot.press("enter")  # Yes -> ProgressScreen (subprocess runs 2s)
        await pilot.pause(0.4)
        assert type(app.screen).__name__ == "ProgressScreen", type(app.screen).__name__

        await pilot.press("escape")  # Esc pops ProgressScreen mid-load
        await pilot.pause(0.4)

        try:
            await pilot.pause(3.0)  # let the worker finish -> _finish_action
        except Exception as exc:  # noqa: BLE001
            crashed.append(exc)

    assert not crashed, f"worker crashed: {crashed[0]!r}"
    # The fix is "no ScreenStackError from a double pop"; the exact screen after
    # a cancelling Esc may differ by Textual version, so only require no crash.


if __name__ == "__main__":
    asyncio.run(drive())
    print("PASS: Esc during load does not crash")
