# About the algorithms

Why each function is computed the way it is, where the accuracy limits come from, and what buying more accuracy would actually cost. The measured numbers themselves live in [Accuracy and domains](../reference/accuracy-and-domains.md); this page is the reasoning behind them.

The recurring theme: for most of these functions the limit is **cancellation, not truncation**. That distinction matters, because adding terms fixes truncation and does nothing at all for cancellation.

## Modified Bessel `K` — two series and a cross-over

`K0` uses a 30-term ascending series below $z = 9$ and a 10-term asymptotic expansion above it, following Zhang and Jin. `K1` comes from the Wronskian, `K2` from the recurrence.

The worst error, $2\times10^{-7}$, sits almost exactly at the hand-off. That is not a coincidence and not a bug: it is where both approximations are simultaneously at their worst, and the cross-over is placed there deliberately, because moving it in either direction makes one branch worse faster than it makes the other better.

**Why the ascending series cannot simply take more terms.** It computes

$$K_0(z) = -\left(\log\tfrac{z}{2} + \gamma\right) I_0(z) + \sum_k \frac{H_k}{(k!)^2}\left(\tfrac{z}{2}\right)^{2k},$$

and at $z = 9$ those two pieces are each of size $I_0(9) \approx 1.09\times10^{3}$ while their difference is $K_0(9) \approx 5.09\times10^{-5}$. The ratio is $2.1\times10^{7}$, so about **7.3 decimal digits are destroyed by cancellation** before the result is formed. float64 starts with 16 and finishes with roughly 9. Adding terms to a sum whose leading digits are already cancelling buys nothing.

This is also why the float32 cross-over is 4.65 rather than 9: the loss grows as $2z/\ln 10$ digits, and float32 has only 7 to spend. Applying the float64 constant at float32 made `K0(8.5)` come out **negative**.

**Why the asymptotic series cannot either.** Asymptotic expansions diverge. Past some optimal term count the partial sums get worse, not better, and that optimum depends on $z$. Ten terms is near it at the cross-over; more would help at large $z$ and hurt at small.

**What machine precision would cost.** SciPy reaches it with Cephes' rational Chebyshev approximations — separate hard-coded coefficient sets per interval, fitted offline. That is the right answer for a C library shipping a fixed set of functions, and the wrong one here: it is several hundred magic constants per function, it must be refitted for each precision, and it is not obviously differentiable in a form JAX can use. The current approach is about forty lines, works at every dtype, and differentiates analytically. Trading $10^{-16}$ for $10^{-7}$ buys that.

**The scaled forms exist to dodge a different limit entirely.** $K_n(z)$ underflows to zero above $z \approx 705.5$ — not from any weakness in the series, but because the true value is smaller than the smallest normal double and XLA flushes it. $e^z K_n(z)$ decays only as $1/\sqrt{z}$, so `K0e`/`K1e`/`K2e` stay accurate to `DBL_MAX`. Writing them against `i0e` rather than `i0` also removed an overflow ceiling the unscaled code used to have.

## `spence` — three series and a removable singularity

Translated from SciPy's Cython implementation: a series about $z = 0$ for $|z| \le 1/2$, an accelerated series about $z = 1$, and a reflected form for $|1 - z| > 1$.

The accelerated series ends by dividing by $1 + 4t + t^2$ with $t = 1 - z$. That quadratic **vanishes** at $t = -2 + \sqrt3$, and so does the numerator — a $0/0$ that produced a 59%-wrong value at Spence argument $3 - \sqrt3$, and again at $3 + \sqrt3$, which the reflected branch maps onto the same root.

The fix is not a patch at the singular point but a change of formula near it: for small denominators the plain defining series $\sum t^n/n^2$ is used instead, which is exact there because $|t| \le 0.42$ in that region and sixty terms reach $10^{-23}$. Both branches are evaluated under a `jnp.where`, so this costs about 7% on the forward path — the price of being right at two points where the alternative was catastrophically wrong.

This bug is worth dwelling on because of _how it survived_: **SciPy's complex `spence` has the same defect**, being the same code. A parity test against SciPy agreed on the wrong answer. Only `mpmath`, which shares no ancestry, could see it.

## `Li` — three branches and a table

The polylogarithm uses the defining sum for $|z| \le 1/2$, a Hurwitz-zeta expansion in $\log z$ for $1/2 < |z| < 2$, and the inversion formula for $|z| \ge 2$.

The inversion branch needs Bernoulli numbers, and the table stops at $B_{60}$. So the practical bound on the order is **branch-dependent**: $n \le 60$ for $|z| \ge 2$, and $n \le 170$ elsewhere, where $\Gamma(n+1)$ overflows. Past the table the result is `nan` rather than a silently clamped index — an earlier version reused $B_{60}$ for every higher order and returned `Li(62, 3) = 0.979` against a true `3.0`.

The derivative uses $\mathrm{Li}_n'(z) = \mathrm{Li}_{n-1}(z)/z$, which is $0/0$ at the origin. Here too the fix is a change of formula rather than a guard: near zero the ratio is evaluated as the series it equals, $\sum_j z^j/(j+1)^{n-1}$, by Horner. A guard would have been a constant, and a constant differentiates to zero — giving the right first derivative and a wrong second.

## `zeta` — delegate, extend, and know when to stop

For $n > 1$ this is `jax.scipy.special.zeta`, unchanged. Above $n = 54$ it is the constant `1.0`, which is not an approximation: $\zeta(n) - 1 \approx 2^{-n}$ falls below half an eps of 1 once $n > 53$, so every double from there up _is_ exactly 1. Taking the constant also sidesteps upstream returning `nan` above $n \approx 10^{15}$.

The negative half-line uses the functional equation $\zeta(-k) = (-1)^k B_{k+1}/(k+1)$, which is why it covers only the negative _integers_: the equation needs $(-1)^{-n}$, undefined otherwise. The critical strip is not implemented at all.

**The Bernoulli numbers are computed from exact `fractions.Fraction` arithmetic**, not from `jax.scipy.special.bernoulli`, which loses about seven digits on $B_4$ — enough to produce $10^{-6}$ errors in `Li(4, ·)` and `zeta(-3)`. Exact rational arithmetic once at import beats a floating-point recurrence every time here, because the table is small and fixed.

A gradient on the negative line is finite and **is not** $\zeta'$: the value comes from a table lookup, which carries no information about how $\zeta$ varies between the integers.

## Gegenbauer — a recurrence, and what that implies

`eval_gegenbauer` runs the three-term recurrence

$$n\,C_n^{(\alpha)} = 2(n + \alpha - 1)\,x\,C_{n-1}^{(\alpha)} - (n + 2\alpha - 2)\,C_{n-2}^{(\alpha)}$$

through `lax.scan`. `eval_gegenbauers` returns the whole scan output instead of its last element, which is free.

The consequence worth understanding: **a recurrence's accuracy is set by its own working magnitude, not by the size of its answer.** At $n = 20$, $\alpha = 10$ the intermediate values peak at $9.6\times10^{7}$ on the way to a result of $1.3\times10^{-2}$. An absolute error of $1.5\times10^{-7}$ there is $1.6\times10^{-15}$ of the peak — backward-stable, and 13× better than SciPy at the same point — but it is not $10^{-15}$ of the _answer_, and no reformulation of a recurrence makes it so. The test tolerances scale with that peak for exactly this reason.

There is no custom derivative rule. $\partial C/\partial x$ has the closed form $2\alpha C_{n-1}^{(\alpha+1)}$, but $\alpha$ is traced too and $\partial C/\partial\alpha$ has no closed form, so a rule supplying only the $x$-tangent would silently break `grad` with respect to `alpha`. Letting JAX differentiate the scan is correct in both arguments.

## `comb` and `gamma` — the boring ones, deliberately

`comb` is $\exp(\ln\Gamma(N+1) - \ln\Gamma(k+1) - \ln\Gamma(N-k+1))$, masked so out-of-range pairs return 0 rather than a plausible wrong number. The $N \le 170$ domain is a real ceiling: the difference of log-gammas cancels catastrophically for large $N$, and `comb(1e10, 2)` is already off by $3.7\times10^{-6}$.

`gamma` computes nothing. It calls JAX and supplies a derivative. That is the whole implementation, and it is the right one: a value that delegates cannot drift from upstream, and the analytic $\Gamma\psi$ keeps 3× less residual memory than differentiating through JAX's own.

## The pattern worth taking away

Two of these functions had the same class of bug, and it was reported four times across three separate rounds of review: a removable singularity guarded by a constant inside a `jnp.where`. The guard makes the value right and every derivative wrong, because a constant differentiates to zero — and the wrongness reappears one order higher each time it is patched. `spence` at $z = 1$ accounted for three of the four — $-1$, then $1/2$, then $-2/3$, one per round — and `Li` at $z = 0$ was the fourth.

The fix that actually terminates is to change the _formula_ near the singular point to one that is analytic there — a series, evaluated by Horner — so that autodiff differentiates a polynomial. Both are now handled that way, and both are correct to every order rather than to the order most recently complained about.

The trap is easy to fall straight back into. The first attempt at the `Li` fix substituted a constant and reproduced the `spence` bug exactly; the second used $\sum_j z^j c_j$, whose term-by-term derivative $j\,z^{j-1}$ is $0 \times \infty$ at the origin for $j = 0$, so the second derivative came back `nan`. Only Horner evaluation gives an honest polynomial. Neither misstep reached a commit, but both were caught by tests rather than by reasoning about them in advance, which is the more useful half of the lesson.

A genuine pole is a different matter, and the distinction is worth keeping straight: nothing is finite there, so no change of formula avoids the substitution and each order needs its own. `grad`$^3$ of the scaled Bessel family at $z = 0$ is documented rather than fixed for exactly that reason.
