"""Special functions."""

__all__ = [
    # Gegenbauer
    "eval_gegenbauer",
    "eval_gegenbauers",  # NOTE: not in scipy.special
    "comb",
    "gamma",
    "K0",
    "K1",
    "K2",
    "Li",  # NOTE: not in scipy.special
    "zeta",
]

from ._src.comb import comb
from ._src.gamma import gamma
from ._src.gegenbauer import eval_gegenbauer, eval_gegenbauers
from ._src.kn import K0, K1, K2
from ._src.polylog import Li
from ._src.zeta import Riemann_zeta as zeta
from ._version import version as __version__  # noqa: F401
