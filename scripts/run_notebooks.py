#!/usr/bin/env python3
"""Execute every notebook top to bottom. Any failing cell fails the run.

    python scripts/run_notebooks.py            check only, files stay untouched
    python scripts/run_notebooks.py --write    also save the fresh outputs

Runs offline unless LLM_PROVIDER is already set, so it needs no key and no
network. Outputs are also scanned so a saved notebook can't leak a local path
or an API key.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
LEAK = re.compile(r"/Users/|/home/|/private/|gsk_[A-Za-z0-9]{8}|\bsk-[A-Za-z0-9_-]{16}")


def outputs_text(notebook) -> str:
    parts = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            parts.append(str(output.get("text", "")))
            parts.append(str(output.get("data", "")))
            parts.append(str(output.get("traceback", "")))
    return "\n".join(parts)


def run(path: Path, write: bool, timeout: int) -> bool:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    )
    started = time.time()
    try:
        client.execute()
    except CellExecutionError as error:
        print(f"FAIL  {path.name}\n{error}")
        return False
    leak = LEAK.search(outputs_text(notebook))
    if leak:
        print(f"FAIL  {path.name}: output looks like a local path or key: {leak.group()!r}")
        return False
    if write:
        nbformat.write(notebook, path)
    print(f"PASS  {path.name} ({time.time() - started:.1f}s)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="save fresh outputs into the notebooks")
    parser.add_argument("--timeout", type=int, default=600, help="seconds allowed per cell")
    args = parser.parse_args()

    paths = sorted(NOTEBOOKS.glob("[0-9]*.ipynb"))
    if not paths:
        print("No notebooks found.")
        return 1
    results = [run(path, args.write, args.timeout) for path in paths]
    print(f"{sum(results)}/{len(results)} notebooks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
