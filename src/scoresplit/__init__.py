"""
scoresplit: split sheet-music PDFs into one PDF per instrument part.

Start reading at cli.py -> api.py -> pipeline.py.
To use scoresplit from other Python code: ``from scoresplit import api``.

The ``main`` function below is what runs when you type ``scoresplit`` in
the terminal (see [project.scripts] in pyproject.toml).
"""

__version__ = "0.1.0"


def main() -> None:
    """Entry point of the `scoresplit` command."""
    # Imported inside the function so that `import scoresplit` stays cheap
    # (it doesn't load the whole CLI unless you actually run it).
    import sys

    from .cli import run

    # sys.argv = what was typed: ["scoresplit", "score.pdf", ...]. [1:]
    # leaves out the program's own name. See cli.run for what happens next.
    run(sys.argv[1:])
