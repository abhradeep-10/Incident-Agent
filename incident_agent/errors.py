class ToolError(Exception):
    """Base class for tool failures that are reported back to the agent."""
    transient = False


class ToolArgError(ToolError):
    """Arguments failed validation. The tool was NOT executed."""


class ToolTransientError(ToolError):
    """Temporary backend failure (e.g. HTTP 503). Safe to retry for idempotent tools."""
    transient = True


class ToolTimeoutError(ToolTransientError):
    """Backend did not answer in time."""


class ToolMalformedError(ToolError):
    """Backend answered with an empty or structurally invalid payload."""
