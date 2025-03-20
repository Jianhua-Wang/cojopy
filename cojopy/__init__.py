"""Top-level package for cojopy."""

import logging

from rich.logging import RichHandler

from .cojopy import COJO

__author__ = """Jianhua Wang"""
__email__ = "jianhua.mert@gmail.com"
__version__ = "0.0.4"


logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
