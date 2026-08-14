"""tactile_recollect: inject 32x32 piezo tactile into DexMimicGen Fourier-hand
demos (Path B = replay-based obs re-extraction) and expose it for online policy
inference.

Importing this package removes the repo root from sys.path so the editable
robosuite install wins over the repo-root `robosuite/` *namespace* shadow.
PathFinder resolves that namespace portion before robosuite's meta_path finder
whenever the repo root sits on sys.path (e.g. `python -m tactile_recollect...`
run from the repo root). Submodules still load via this package's __path__, so
dropping the repo root is safe. Must run before any `import robosuite`.
"""
import os as _os
import sys as _sys

_repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path
                if _os.path.abspath(p or ".") != _repo]
