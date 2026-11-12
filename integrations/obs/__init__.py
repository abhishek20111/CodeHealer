"""OBS (log observability) Integration Package.

Query logs from a log-observability service using the Proxy Service API.
"""

from .query_client import ObsQueryClient

__version__ = "1.0.0"
__all__ = ["ObsQueryClient"]
