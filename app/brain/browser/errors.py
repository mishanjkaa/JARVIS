from __future__ import annotations


class BrowserError(RuntimeError):
    """Base browser runtime error."""


class BrowserDisabledError(BrowserError):
    pass


class BrowserUnavailableError(BrowserError):
    pass


class BrowserPolicyError(BrowserError):
    pass


class BrowserSessionError(BrowserError):
    pass


class BrowserOperationError(BrowserError):
    pass


class BrowserTimeoutError(BrowserOperationError):
    pass


class BrowserCancelledError(BrowserOperationError):
    pass

