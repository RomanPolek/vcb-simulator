from __future__ import annotations

from runner.native import NativeRuntime
from runner.renderer import PygameRenderer
from runner.simulation import SimulationDriver


def run_prepared_runtime(runtime: NativeRuntime, *, frame_hz: int = 60) -> None:
    """Run the native simulation worker and its interactive renderer."""
    driver = SimulationDriver(runtime, ticks_per_second=None, frame_hz=frame_hz)
    renderer = PygameRenderer(runtime, driver)
    driver.start()
    try:
        renderer.run()
    finally:
        driver.stop()
