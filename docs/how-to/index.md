# How-to guides

Directions for specific tasks. These assume you already know what you want to do and are looking for the shortest path to it. If you are still orienting yourself, start with the [tutorial](../tutorials/index.md) instead.

- [How to enable double precision](enable-double-precision.md) — the first thing to do in any project using `spexial`, and the ways to get it wrong.
- [How to use spexial with jit, vmap and grad](use-with-jit-vmap-and-grad.md) — broadcasting, static integer parameters, the `polylog` exception, and the one gradient you should not trust.
- [How to port code from `scipy.special`](port-from-scipy.md) — what swaps cleanly, what is less accurate, and what has no counterpart.

- [How to choose between `spexial` and `jax.scipy.special`](choose-between-spexial-and-jax.md) — the four functions where both libraries have something, and which to reach for.

For the exact domain and tolerance of any function, see [Accuracy and domains](../reference/accuracy-and-domains.md).
