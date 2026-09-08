from jax import lax, numpy as jnp


def gamma(x):
    """Gamma function using the Lanczos approximation. Differs from the
    jax.scipy.special implementation in that it supports imaginary
    arguments.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html#scipy.special.gamma

    Parameters
    ----------
    x : array_lke
        Real or complex valued argument.

    Returns
    -------
    array_like
        Value(s) of the gamma function

    """

    def lanczos(x):
        g = 7

        p_vals = jnp.array(
            [
                0.99999999999980993,
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7,
            ]
        )

        eps = 1e-06

        z = x
        z_fill = z - 1
        denom = z_fill + jnp.arange(1, len(p_vals))
        x_fill = p_vals[0] + jnp.sum(p_vals[1:] / denom)
        t = z_fill + g + 0.5

        result1 = jnp.sqrt(2 * jnp.pi) * t ** (z_fill + 0.5) * jnp.exp(-t) * x_fill
        result2 = jnp.where(jnp.abs(result1.imag) <= eps, result1.real, result1)
        return result2

    def reflect(x):
        return jnp.pi / jnp.sin(jnp.pi * x) * lanczos(1.0 - x)

    return lax.cond(x < 0.5, reflect, lanczos, x)
