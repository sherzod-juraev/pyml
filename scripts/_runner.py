"""Shared step-runner used by all check_*.py scripts."""

import subprocess


def run_steps(steps: list[tuple[str, list[str]]]) -> int:
    """Execute a sequence of check commands and print a summary.

    Parameters
    ----------
    steps : list of (str, list of str)
        Pairs of a human-readable step name and the command to run.

    Returns
    -------
    int
        0 if all steps passed, 1 if any failed.
    """
    results: list[tuple[str, bool]] = []
    for name, command in steps:
        print(f"\n=== {name} ===")
        result = subprocess.run(command)
        results.append((name, result.returncode == 0))

    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1
