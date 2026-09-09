"""Master runner to execute all project quality checks sequentially.

Runs check_pyml, check_docs, and check_tests and returns a unified status.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    """Run all three quality check scripts."""
    scripts = [
        "check_pyml.py",
        "check_docs.py",
        "check_tests.py",
    ]

    all_passed = True
    for script in scripts:
        print("\n=========================================")
        print(f"RUNNING: {script}")
        print("=========================================")

        result = subprocess.run([sys.executable, str(SCRIPT_DIR / script)])
        if result.returncode != 0:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
