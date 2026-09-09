"""Package setup information.

Note that this module is NOT public API nor are any of its contents.
Stability is NOT guaranteed.
This module exposes package setup information for the `spexial` package.

"""

__all__: tuple[str, ...] = ()

import contextlib
import os
from collections.abc import Sequence
from typing import Any, Final, Literal

from jaxtyping import install_import_hook as _install_import_hook

_env = os.getenv("SPEXIAL_ENABLE_RUNTIME_TYPECHECKING", "False")
RUNTIME_TYPECHECKER: Final[str | Literal[False] | None] = {
    "False": False,
    "None": None,
}.get(_env, _env)
"""Runtime type checking variable "SPEXIAL_ENABLE_RUNTIME_TYPECHECKING".

Set to "False" to disable runtime typechecking (default).
Set to "None" to only enable typechecking for `@jaxtyped`-decorated functions.
Set to "beartype.beartype" to enable runtime typechecking.

See https://docs.kidger.site/jaxtyping/api/runtime-type-checking for more
information on options.

"""


def install_import_hook(
    modules: str | Sequence[str], /
) -> contextlib.AbstractContextManager[Any, None]:
    """Install the jaxtyping import hook for the given modules.

    Parameters
    ----------
    modules
        Module name or sequence of module names to install the import hook for.

    Returns
    -------
    contextlib.AbstractContextManager
        Context manager that installs the import hook on entry and removes it on
        exit.

    """
    return (
        _install_import_hook(modules, RUNTIME_TYPECHECKER)
        if RUNTIME_TYPECHECKER is not False
        else contextlib.nullcontext()
    )
