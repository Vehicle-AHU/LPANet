"""Backward-compatible validation entry point.

Historically LPANet used ``valtest.py`` for standalone evaluation while
``val.py`` was called from training.  The public release keeps this filename so
existing commands continue to work, but the implementation now lives in one
place: :mod:`val`.
"""

from val import main, parse_opt, run

__all__ = ['run', 'parse_opt', 'main']


if __name__ == '__main__':
    main(parse_opt())
