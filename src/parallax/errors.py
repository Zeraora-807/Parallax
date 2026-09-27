"""Typed, user-safe errors produced by the runtime."""


class ParallaxError(Exception):
    """Base class for errors that can be returned without a traceback."""

    code = "parallax_error"
    retryable = False
    execution_state = "not_started"
    recovery_hint: str | None = None


class ToolNotFoundError(ParallaxError):
    code = "tool_not_found"


class ToolValidationError(ParallaxError):
    code = "invalid_arguments"


class PermissionDeniedError(ParallaxError):
    code = "permission_denied"


class WorkspaceViolationError(ParallaxError):
    code = "workspace_violation"


class ToolUnavailableError(ParallaxError):
    code = "tool_unavailable"
    retryable = True
    recovery_hint = "Select an available tool or enable the required plugin."


class ExecutionTimeoutError(ParallaxError):
    code = "execution_timeout"
    retryable = True
    execution_state = "interrupted"
    recovery_hint = "Reduce the analysis scope or choose a lighter tool."


class OutputLimitExceededError(ParallaxError):
    code = "output_limit_exceeded"
    execution_state = "interrupted"
    recovery_hint = "Request a narrower result or use an output-filtering tool."


class BudgetExceededError(ParallaxError):
    code = "budget_exceeded"
    recovery_hint = "Stop the run or start a new run with an explicitly larger budget."


class ArtifactAccessError(ParallaxError):
    code = "artifact_access_denied"


class PluginValidationError(ParallaxError):
    code = "plugin_validation_error"


class EvolutionError(ParallaxError):
    code = "evolution_error"


class ModelProtocolError(ParallaxError):
    code = "model_protocol_error"


class InternalRuntimeError(ParallaxError):
    code = "internal_runtime_error"
