# Explanation

Background on why `spexial` behaves the way it does. These pages are for reading away from the keyboard; nothing here is needed to get work done, and everything here is needed to predict what the library will do at its edges.

- [About precision](precision.md) — why double precision is not optional for these functions, why it is a global switch, and how a computation silently ends up in float32 anyway.
- [About domain edges, `nan`, and gradients that lie](edges.md) — why traced code returns `nan` instead of raising, why accuracy degrades near a pole, and why a gradient can be finite and still wrong.
- [Why `spexial` exists alongside `jax.scipy.special`](why-not-upstream.md) — what upstream can and cannot accept, the version-floor problem, and how this package is designed to shrink.
- [About the algorithms](algorithms.md) — series orders, cross-over points, why the limit is usually cancellation rather than truncation, and what machine precision would actually cost.
