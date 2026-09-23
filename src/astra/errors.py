"""Exception hierarchy for the ASTRA control plane."""


class AstraError(Exception):
    """Base class for all expected, user-reportable ASTRA failures."""


class CommandError(AstraError):
    """An external command (nvidia-smi, dmesg, ...) failed or is missing."""


class ParseError(AstraError):
    """Output from a hardware tool could not be interpreted."""


class ConfigError(AstraError):
    """The ASTRA configuration file is missing required values or is malformed."""


class PlanningError(AstraError):
    """A workload cannot be placed on the available GPU pool."""
