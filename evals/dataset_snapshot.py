"""Export or restore a LangSmith dataset to or from a committed JSON snapshot.

Pick the dataset by alias - `golden`, `hallucinations`, `pii`, `guardrail`, or
`all`:

    # recreate a dataset in LangSmith from its committed snapshot
    python evals/dataset_snapshot.py restore golden
    python evals/dataset_snapshot.py restore golden --reset   # overwrite if it exists
    python evals/dataset_snapshot.py restore all --reset      # recreate every dataset

    # pull the live dataset back down into its snapshot, for example after
    # Engine adds examples, or after editing them in LangSmith
    python evals/dataset_snapshot.py export hallucinations
    python evals/dataset_snapshot.py export all

The snapshots are committed, so the datasets survive a deletion and give a
reproducible baseline for demo practice. Nobody has to wait on an Engine scan.

Dataset NAMES carry the presenter suffix, because presenters share one
workspace. The suffix is applied here, from DEMO_PRESENTER, and not stored in
the JSON. The `name` field inside each snapshot is the unscoped base name and
is informational only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client  # noqa: E402

from evals.dataset import DEMO_PRESENTER  # noqa: E402

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path


def _scoped(base: str) -> str:
    return f"{base}-{DEMO_PRESENTER}"


DATASETS = {
    "golden": DatasetConfig(_scoped("chat-lc-lite-golden"), HERE / "dataset_golden.json"),
    "hallucinations": DatasetConfig(
        _scoped("chat-lc-lite-hallucinations"), HERE / "dataset_hallucinations.json"
    ),
    "pii": DatasetConfig(_scoped("chat-lc-lite-pii"), HERE / "dataset_pii.json"),
    "guardrail": DatasetConfig(_scoped("chat-lc-lite-guardrail"), HERE / "dataset_guardrail.json"),
}

EXAMPLES = """\
examples:
  python evals/dataset_snapshot.py restore all --reset       # recreate every dataset
  python evals/dataset_snapshot.py restore golden --reset    # recreate one dataset
  python evals/dataset_snapshot.py restore pii               # skip if it already exists
  python evals/dataset_snapshot.py export guardrail          # pull live dataset into its snapshot
"""


class _ExamplesParser(argparse.ArgumentParser):
    """Show usage plus examples instead of a terse usage line on error."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        sys.stderr.write(f"\nerror: {message}\n\n{EXAMPLES}")
        raise SystemExit(2)


def export_dataset(name: str, path: Path) -> None:
    client = Client()
    ds = client.read_dataset(dataset_name=name)
    examples = list(client.list_examples(dataset_id=ds.id))

    # Keep the unscoped base name in the file, so a snapshot is portable
    # between presenters.
    base = ds.name
    suffix = f"-{DEMO_PRESENTER}"
    if base.endswith(suffix):
        base = base[: -len(suffix)]

    payload = {
        "name": base,
        "description": ds.description or "",
        "examples": [
            {"inputs": ex.inputs, "outputs": ex.outputs, "metadata": ex.metadata or {}}
            for ex in examples
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Exported {len(examples)} examples from {ds.name!r} to {path}")


def restore_dataset(name: str, path: Path, reset: bool) -> None:
    if not path.is_file():
        raise SystemExit(f"Snapshot not found at {path}.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    description = payload.get("description") or ""
    examples = payload.get("examples", [])

    client = Client()
    if list(client.list_datasets(dataset_name=name)):
        if not reset:
            print(f"Dataset {name!r} already exists. Pass --reset to delete and recreate it.")
            return
        print(f"Deleting existing dataset {name!r}...")
        client.delete_dataset(dataset_name=name)

    ds = client.create_dataset(dataset_name=name, description=description)
    print(f"Created dataset {ds.name} ({ds.id})")

    if not examples:
        print("Snapshot has no examples; dataset created empty.")
        return

    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": ex.get("inputs", {}),
                "outputs": ex.get("outputs", {}),
                "metadata": ex.get("metadata", {}) or {},
            }
            for ex in examples
        ],
    )
    print(f"Restored {len(examples)} examples.")


def main() -> None:
    parser = _ExamplesParser(
        description=__doc__,
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("action", choices=("export", "restore"))
    parser.add_argument("dataset_choice", choices=[*sorted(DATASETS), "all"])
    parser.add_argument("--name", default=None, help="Override the dataset name (single dataset).")
    parser.add_argument("--path", default=None, help="Override the snapshot path (single dataset).")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="On restore, delete and recreate a dataset that already exists.",
    )
    args = parser.parse_args()

    if args.dataset_choice == "all":
        if args.name or args.path:
            raise SystemExit("--name/--path target a single dataset; omit them with 'all'.")
        targets = [DATASETS[c] for c in sorted(DATASETS)]
    else:
        config = DATASETS[args.dataset_choice]
        targets = [
            DatasetConfig(args.name or config.name, Path(args.path) if args.path else config.path)
        ]

    for target in targets:
        if args.action == "export":
            export_dataset(target.name, target.path)
        else:
            restore_dataset(target.name, target.path, args.reset)


if __name__ == "__main__":
    main()
