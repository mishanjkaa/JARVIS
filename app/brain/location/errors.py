from __future__ import annotations


class LocationError(ValueError):
    """Base class for safe Location/Navigation runtime failures."""


class LocationDisabledError(LocationError):
    """Raised when the Location runtime is disabled."""


class LocationUnavailableError(LocationError):
    """Raised when no freshest point has been received from the phone yet."""


class LocationPlaceNotFoundError(LocationError):
    """Raised when a named place has not been saved by the owner."""


class LocationDestinationError(LocationError):
    """Raised when a navigation destination cannot be resolved to coordinates."""


class LocationSessionNotFoundError(LocationError):
    """Raised when a navigation session id does not refer to an active session."""


class LocationGeocodeError(LocationError):
    """Raised when a geocoding provider request fails or returns no result."""


class LocationRoutingError(LocationError):
    """Raised when a routing provider request fails or returns no route."""


class LocationBindError(LocationError):
    """Raised when the configured Tailscale bind address is missing or unusable."""
