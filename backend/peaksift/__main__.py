"""``python -m peaksift`` — same as the ``peaksift`` console script.

Useful where the generated ``peaksift.exe`` launcher is blocked (Windows Smart
App Control blocks unsigned launchers) or not on PATH.
"""

from .cli import app

app()
