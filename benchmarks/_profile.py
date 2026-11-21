"""Profile a script with pyinstrument and cProfile.

The two profilers answer different questions, so they run one after the other, never together
(each one slows the other down and bends its numbers):

* ``pyinstrument`` samples the call stack and shows where the wall-clock time goes.
* ``cProfile`` counts every call, which shows how often a function runs, for example once per
  iteration of a training loop.

The first run also pays the cost of the imports, the second one does not.
"""

import cProfile
import os
import pstats
import runpy
import sys
from pathlib import Path

from pyinstrument import Profiler

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Rows of the cProfile tables that belong to this tool (the code that starts the script), not to
# the script. The pattern matches every other row.
_NOT_TOOL = r"^(?!.*(<frozen runpy>|_profile\.py|builtins\.exec)).*$"


def run_script(path: Path) -> None:
    """Run a script the way ``python path`` does.

    The folder of the script is put first on ``sys.path`` and ``sys.argv`` holds only the
    script, as in a normal run. The matplotlib backend is set to ``Agg`` (unless it is already
    set), so that examples do not open windows.

    Parameters
    ----------
    path : Path
        Python file to run.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    saved_argv = sys.argv
    saved_path = sys.path[:]
    sys.argv = [str(path)]
    sys.path.insert(0, str(path.parent))
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path


def profile_time(path: Path, interval: float, html: bool) -> Path | None:
    """Print the pyinstrument call tree of a script and, optionally, save it as HTML.

    Parameters
    ----------
    path : Path
        Python file to run.
    interval : float
        Seconds between two samples of the call stack.
    html : bool
        If True, also write the interactive HTML report into the results folder.

    Returns
    -------
    Path or None
        The HTML report, or None when ``html`` is False.
    """
    profiler = Profiler(interval=interval)
    profiler.start()
    try:
        run_script(path)
    finally:
        profiler.stop()
    profiler.print()

    if not html:
        return None
    RESULTS_DIR.mkdir(exist_ok=True)
    report = RESULTS_DIR / f"{path.stem}-pyinstrument.html"
    profiler.write_html(report)
    return report


def profile_calls(path: Path, top: int) -> None:
    """Print the cProfile tables of a script, by cumulative time and by number of calls.

    Parameters
    ----------
    path : Path
        Python file to run.
    top : int
        Number of rows in each table.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_script(path)
    finally:
        profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs()
    print(f"\ncProfile: top {top} by cumulative time")
    stats.sort_stats("cumulative").print_stats(_NOT_TOOL, top)
    print(f"cProfile: top {top} by number of calls")
    stats.sort_stats("ncalls").print_stats(_NOT_TOOL, top)


def profile_script(path: Path, top: int, interval: float, html: bool) -> None:
    """Profile a script with pyinstrument first and with cProfile second.

    Parameters
    ----------
    path : Path
        Python file to run.
    top : int
        Number of rows in each cProfile table.
    interval : float
        Seconds between two pyinstrument samples.
    html : bool
        If True, also write the pyinstrument HTML report into the results folder.
    """
    print(f"pyinstrument: {path}")
    report = profile_time(path, interval, html)
    if report is not None:
        print(f"HTML report: {report}")
    profile_calls(path, top)
