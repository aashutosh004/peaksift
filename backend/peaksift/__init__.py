"""PeakSift — open-source, local-first AI video clipper."""

from . import portable as _portable

# Must run before anything imports huggingface_hub, yt-dlp or keyring, all of
# which read their cache/config locations from the environment at import time.
_portable.apply()

__version__ = "0.1.0"

__all__ = ["__version__"]
