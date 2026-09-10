"""Package setup: the runtime type-checking hook and the variable that arms it.

Mostly internal, with two deliberate exceptions -- so this module does not
carry the blanket "NOT public API" note the rest of `spexial._src` does, which
would contradict both a user-facing document and a test.

`SPEXIAL_ENABLE_RUNTIME_TYPECHECKING` is **supported**. It is documented for
users in ``docs/reference/errors.md``, which tells them to set it to
``beartype.beartype`` to get argument checking, so its name and its three
accepted values are a contract.

`install_import_hook` is a **development aid**, supported at this location and
only at this one. `spexial/__init__.py` imports it to wrap the package's own
imports and then deletes the binding, precisely so that
``spexial.install_import_hook`` does not become a name anyone depends on by
accident; `tests/unit/test_setup_package.py` pins both halves of that.

Everything else here -- `RUNTIME_TYPECHECKER` as a name, the parsing of the
environment variable -- is internal, and stability is not guaranteed for it.

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
