Different ways to run Python files:
- python3 -m src.evaluator this runs the file within a module
- python3 src/evaluator.py this runs it with src as the root, instead of root as the actual root.

Python Modules & Packages

When a project uses package-style imports such as:

    from src.classifier import classify_email
    from src.models import ClassificationOutput

run the module from the PROJECT ROOT:

    python3 -m src.evaluator

Not:

    python3 src/evaluator.py

Why:
- `python3 src/evaluator.py` executes the file as a standalone script and puts `src/` on the import path.
- `python3 -m src.evaluator` executes `evaluator.py` as the `src.evaluator` module and runs it with the project root on the import path.
- This allows Python to correctly resolve imports such as `src.classifier` and `src.models`.

Key distinction:
    python3 src/evaluator.py
        → "run this file"

    python3 -m src.evaluator
        → "run this module as part of the package"

Production Python projects generally favor package/module execution (`python -m ...`) and proper package structure over manually modifying `sys.path`.