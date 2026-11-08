"""Post-freeze, research-only strict admission boundary for CGC-LAO v4.

This package is intentionally additive: historical v1-v4 frozen files remain unchanged.
"""

from .strict_host import StrictResearchHost, strict_admit

__all__ = ["StrictResearchHost", "strict_admit"]
