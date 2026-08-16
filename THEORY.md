# Theory: Variance of Window-Correlated Stochastic Rounding in Re-Quantized Accumulators

Draft of the theory section + theory appendix for the ICLR 2027 submission
(PAPER_PLAN.md, "theorem package": L1, L2, P1, P2, N1). All closed forms and
counterexamples in this file were verified by Monte Carlo on 2026-08-15 before
finalization; the verification script and its verbatim output are in Appendix A.
Statements formalize the code at `src/optim/gquant.py` (Site 1) and
`src/muon.py::_mq_quantize` (Site 2).

Role of each statement in the paper:

| statement | role |
|---|---|
| L1 | fixes conventions; unbiasedness of a single SR rounding, and uniqueness of the uniform dither |
| L2 | the crux: the frozen-grid accumulator reduces the window to *fixed* monotone integrands |
| P1 | the guarantee: $\mathrm{Var} \le \mathrm{Var}_{\mathrm{iid}}$ for negatively associated dither (strat/LHS, antithetic) |
| P2 | exact per-mode variance formulas — the quantitative predictions behind the mechanism figure |
| N1 | the shifted lattice (and latperm) have **no** such guarantee — the observed lattice losses are theory-consistent |

---

## 0. Scope and the reduction

All results below concern a single coordinate of a single accumulation window
and are elementary; the intended target is a careful appendix, with L2, P1, and
N1 quoted in the main text. We claim no new sampling theory. The variance
inequality in P1 **is** the classical Latin-hypercube monotonicity theorem of
McKay, Beckman and Conover (1979), reproved here through negative association
(Joag-Dev and Proschan, 1983) so that the antithetic window is covered by the
same three-line argument. The contribution is a *reduction*: a sequentially
re-quantized accumulator whose grid is frozen for the duration of the window
stays exactly on-grid between additions, so the fractional parts that determine
each rounding are **independent of all previous dither draws** (L2). The $m$
roundings of a window therefore collapse to a fixed vector of monotone
indicator integrands evaluated at the window's dither vector, and classical
dithered-quadrature theory applies *verbatim* along the **time axis** of the
accumulator. Correlating quantizer randomness across a population is known on
the *spatial* axis — across clients in distributed mean estimation (Suresh et
al., 2022; see also Panferov et al., 2025) — where the aggregation happens in
full precision at a server. The point of L2 is that a re-quantized accumulator,
which aggregates *in low precision with no error feedback*, nevertheless
supports the identical analysis, because re-quantization onto the frozen grid
restores exact on-grid state before every add.

The closed forms (P2) convert logged threshold statistics into exact predicted
window-error variances per dither mode, which the paper's mechanism figure
tests against the measured window error (fp32-shadow instrumentation). The
counterexample (N1) shows the shifted-lattice construction — and, less
obviously, its permuted-assignment variant latperm — admits threshold profiles
with *positive* step-error covariance and window variance strictly above iid,
so no profile-free guarantee exists for them; among the five constructions run
in the paper, the two with guarantees (strat, antithetic) are exactly the two
that are best-or-tied empirically (RESULTS.md, ordering controls).

---

## 1. Setting and notation

### 1.1 Quantizer

For a grid step $s > 0$ and dither $u \in [0,1)$, the stochastic-rounding
quantizer onto the uniform grid $s\mathbb{Z}$ is

$$Q_{s,u}(x) \;=\; s\,\big\lfloor x/s + u \big\rfloor .$$

The implementations clamp the grid index to $[-q_{\max}, q_{\max}]$ with
$q_{\max} = 2^{b-1}-1$ at $b$ bits (a symmetric integer grid *with* a zero
level). The idealized analysis in Sections 2–6 ignores clamping; Section 7
(C1) states exactly what saturation breaks.

### 1.2 Site 1 — gquant (`src/optim/gquant.py`)

One optimizer step is a window of $m$ micro-steps ($m = \texttt{acc\_steps}$,
default $8$). At micro-step $0$ the per-row step size

$$s \;=\; \texttt{headroom} \cdot m \cdot \mathrm{absmax}_{\mathrm{row}}(g_1) \,/\, q_{\max}$$

is computed and **frozen for the window**. With $G_0 = 0$ the accumulator
evolves as

$$G_k \;=\; Q_{s,u_k}\!\big(G_{k-1} + g_k\big), \qquad k = 1, \dots, m,$$

where $g_k$ is the $k$-th micro-gradient (per coordinate) and $u_k$ the $k$-th
dither draw. There is no error feedback: each rounding error persists in $G$.

### 1.3 Site 2 — mq (`src/muon.py::_mq_quantize`)

The Muon momentum buffer is re-quantized after every momentum update:

$$\mathrm{buf}_{t+1} \;=\; Q_{s_t,\,u_t}\!\big(\beta\,\mathrm{buf}_t + G_t\big),
\qquad \beta = 0.95,$$

with the step size **recomputed every optimizer step from the pre-round
buffer**, $s_t = \texttt{headroom}\cdot \mathrm{absmax}_{\mathrm{row}}(\beta\,
\mathrm{buf}_t + G_t)/q_{\max}$, and window constructions tiled over blocks of
8 consecutive optimizer steps. Because $s_t$ and $\mathrm{buf}_t$ depend on all
earlier dither draws, the thresholds at this site are **adaptive**: L2 fails
and none of P1–P2 formally applies there. Section 7 (C2) states precisely what
survives. Everything in Sections 3–6 is about the fixed-grid Site 1 (and any
future site with the same frozen-grid, on-grid-state structure).

### 1.4 Dither constructions

Per coordinate and per window, the $m$ draws $u_1,\dots,u_m$ are (with
$U, \xi_i \sim \mathcal{U}[0,1)$ fresh and independent where they appear, and
$\pi$ a uniform random permutation of $\{0,\dots,m-1\}$ independent of the
rest):

| mode | construction | per-draw marginal |
|---|---|---|
| iid | $u_i$ i.i.d. $\mathcal{U}[0,1)$ | uniform |
| antithetic ("qmc") | pairs $(u_{2l-1}, u_{2l}) = (U_l,\, 1-U_l)$, $U_l$ i.i.d. | uniform |
| lattice | $u_i = \mathrm{frac}(U + i/m)$, one shared $U$ | uniform |
| vdc | $u_i = \mathrm{frac}(U + \sigma(i)/m)$, $\sigma$ = bit reversal | uniform |
| latperm | $u_i = \mathrm{frac}(U + \pi(i)/m)$, shared $U$, random $\pi$ | uniform |
| strat (LHS) | $u_i = (\pi(i) + \xi_i)/m$, fresh independent $\xi_i$ | uniform |

Every mode has exact $\mathcal{U}[0,1)$ marginals (for the shifted point sets
this is the Cranley–Patterson (1976) randomization property), so by L1 every
*single* rounding is unbiased in every mode; the modes differ only in the
*joint* law across the window.

### 1.5 Notation

| symbol | meaning |
|---|---|
| $s$ | grid step (frozen per window at Site 1; per-row) |
| $Q_{s,u}$ | SR quantizer $x \mapsto s\lfloor x/s + u\rfloor$ |
| $b,\; q_{\max}$ | bit width; $q_{\max} = 2^{b-1}-1$ |
| $m$ | window length (micro-steps at Site 1; block length 8 at Site 2) |
| $g_k$ | $k$-th micro-gradient (one coordinate) |
| $G_k$ | quantized accumulator after $k$ adds, $G_0 = 0$ |
| $t_k$ | pre-round grid coordinate $(G_{k-1}+g_k)/s$ |
| $c_k$ | rounding threshold $\mathrm{frac}(g_k/s) \in [0,1)$ |
| $f_k(u)$ | centered step error $\mathbf{1}\{u \ge 1-c_k\} - c_k$ |
| $S$ | normalized window error $(G_m - \sum_k g_k)/s = \sum_k f_k(u_k)$ |
| $U, \pi, \xi_i$ | shared shift, random permutation, within-stratum jitter |
| $\beta,\ \mathrm{buf}_t,\ s_t$ | momentum coefficient, momentum buffer, adaptive step (Site 2) |
| $\mathcal{F}$ | $\sigma$-field of everything before the window plus its minibatches |
| $\mathrm{Var}_{\mathrm{iid}}$ | $\sum_k c_k(1-c_k)$ |
| $p_{k,s},\ a_{k,s},\ v_{k,s}$ | stratum firing prob., stratum mean, stratum variance of $f_k$ (§5) |
| $\bar a_s$ | column mean $\frac1m\sum_k a_{k,s}$ |
| $\mathrm{ov}(a,b,\delta)$ | circular arc-overlap length (§5, eq. (5.1)) |
| $\mathrm{frac}(x)$ | $x - \lfloor x \rfloor$ |

---

## 2. L1 — Unbiasedness and uniqueness of the uniform dither

**Lemma L1.** *Let $u$ be a random variable with $\Pr(u \in [0,1)) = 1$. Then*

$$\mathbb{E}\big[\lfloor t + u \rfloor\big] = t \quad \text{for all } t \in \mathbb{R}
\qquad \Longleftrightarrow \qquad u \sim \mathcal{U}[0,1).$$

**Proof.** For any $t \in \mathbb{R}$ and any $u \in [0,1)$, write $t =
\lfloor t \rfloor + \mathrm{frac}(t)$. Since $\mathrm{frac}(t) + u \in [0,2)$,

$$\lfloor t + u \rfloor \;=\; \lfloor t \rfloor + \lfloor \mathrm{frac}(t) + u \rfloor
\;=\; \lfloor t \rfloor + \mathbf{1}\{u \ge 1 - \mathrm{frac}(t)\}. \tag{2.1}$$

Taking expectations,

$$\mathbb{E}\big[\lfloor t+u \rfloor\big] \;=\; \lfloor t \rfloor + \Pr\big(u \ge 1 - \mathrm{frac}(t)\big). \tag{2.2}$$

($\Leftarrow$) If $u \sim \mathcal{U}[0,1)$ then $\Pr(u \ge 1-c) = c$ for every
$c \in [0,1)$, so (2.2) equals $\lfloor t \rfloor + \mathrm{frac}(t) = t$.

($\Rightarrow$) If $\mathbb{E}\lfloor t+u\rfloor = t$ for all $t$, then by
(2.2), $\Pr(u \ge 1-c) = c$ for all $c \in [0,1)$; substituting $x = 1-c \in
(0,1]$,

$$\Pr(u < x) = x \qquad \text{for all } x \in (0,1].$$

Hence for $y \in [0,1)$, $\Pr(u \le y) = \lim_{x \downarrow y} \Pr(u < x) =
\lim_{x \downarrow y} x = y$ (continuity of probability along the decreasing
events $\{u < x\}$), and $\Pr(u < 0) = 0$, $\Pr(u \le 1) = 1$. This is the CDF
of $\mathcal{U}[0,1)$. $\blacksquare$

**Remark 2.1 (the support restriction is necessary).** On unrestricted support
the characterization fails: with $V \sim \mathcal{U}[0,1)$, the mixture $u =
V+1$ w.p. $\tfrac12$ and $u = V-1$ w.p. $\tfrac12$ satisfies
$\mathbb{E}\lfloor t+u \rfloor = \tfrac12(1 + \lfloor t+V\rfloor) +
\tfrac12(-1 + \lfloor t+V\rfloor) = t$ for all $t$. In general the identity
holds iff $\mathrm{frac}(u) \sim \mathcal{U}[0,1)$ and $\mathbb{E}[u] =
\tfrac12$. Within the dither class $u \in [0,1)$, the uniform is the unique
unbiased dither.

**Remark 2.2 (provenance and conventions).** This is the classical Schuchman
condition for subtractively-undithered uniform quantization (Schuchman, 1964;
Gray and Stockham, 1993; Zamir and Feder, 1992), specialized to our floor-based
convention. We include the proof to pin the exact boundary conventions used
throughout: half-open $[0,1)$ dither, error indicator $\mathbf{1}\{u \ge
1-c\}$, and the identity (2.1), which is also the computational core of L2.

---

## 3. L2 — Frozen-threshold lemma (the crux)

**Lemma L2.** *Fix $s > 0$, $m \ge 1$, and reals $g_1, \dots, g_m$. Let
$u_1, \dots, u_m$ be $[0,1)$-valued random variables with **arbitrary joint
law**. Define $G_0 = 0$ and $G_k = s\lfloor (G_{k-1}+g_k)/s + u_k \rfloor$
(no clamping). Then:*

*(i) $G_k \in s\mathbb{Z}$ for every $k$ (the accumulator is exactly on-grid).*

*(ii) With $t_k := (G_{k-1}+g_k)/s$, the fractional part satisfies, surely,*

$$\mathrm{frac}(t_k) \;=\; \mathrm{frac}(g_k / s) \;=:\; c_k,$$

*a deterministic constant — independent of $u_1, \dots, u_m$ and in particular
of all previous dither draws.*

*(iii) The normalized window error is an additive sum of centered
Bernoulli step errors with **fixed** thresholds:*

$$S \;:=\; \frac{G_m - \sum_{k=1}^m g_k}{s}
\;=\; \sum_{k=1}^m \Big( \mathbf{1}\{u_k \ge 1 - c_k\} - c_k \Big)
\;=\; \sum_{k=1}^m f_k(u_k),$$

*where each $f_k(u) = \mathbf{1}\{u \ge 1-c_k\} - c_k$ is nondecreasing in
$u$, bounded, and satisfies $\mathbb{E}[f_k(u)] = 0$ for $u \sim
\mathcal{U}[0,1)$.*

**Proof.** (i) Induction: $G_0 = 0 \in s\mathbb{Z}$, and $G_k$ is $s$ times an
integer. (ii) $t_k = G_{k-1}/s + g_k/s$ and $G_{k-1}/s \in \mathbb{Z}$ by (i),
so $\mathrm{frac}(t_k) = \mathrm{frac}(g_k/s)$ on every realization. (iii) By
(2.1), for any real $t$ and $u \in [0,1)$, $\lfloor t + u\rfloor - t =
\mathbf{1}\{u \ge 1-\mathrm{frac}(t)\} - \mathrm{frac}(t)$. Hence the $k$-th
step error is

$$\frac{G_k - (G_{k-1} + g_k)}{s} \;=\; \lfloor t_k + u_k \rfloor - t_k
\;=\; \mathbf{1}\{u_k \ge 1 - c_k\} - c_k \;=\; f_k(u_k),$$

using (ii) to replace $\mathrm{frac}(t_k)$ by the constant $c_k$. Summing the
telescoping identity $G_m - \sum_k g_k = \sum_k \big(G_k - G_{k-1} -
g_k\big)$ and dividing by $s$ gives the claim. Monotonicity, boundedness and
centeredness of $f_k$ are immediate (centeredness is L1's forward
direction). $\blacksquare$

**Corollary 3.1 (window unbiasedness for every mode).** If each $u_k$ is
marginally $\mathcal{U}[0,1)$ — true for all five constructions of §1.4,
regardless of their dependence — then $\mathbb{E}[S] = 0$, i.e.
$\mathbb{E}[G_m] = \sum_k g_k$. Dependence across the window affects only the
variance.

**Remark 3.1 (why this is the crux).** Without re-quantization onto a frozen
grid (e.g. if $s$ were rescaled inside the window, or the accumulator kept in
float), $\mathrm{frac}(t_k)$ would depend on $u_1, \dots, u_{k-1}$: the
integrand seen by draw $k$ would be random and correlated with the very draws
we are trying to correlate, and no off-the-shelf LHS/antithetic theory would
apply. On-grid state is what makes the window a *fixed-integrand* dithered
quadrature problem, so the classical theory (McKay et al., 1979; Stein, 1987;
Owen, 1997) transports verbatim to the time axis. This is also exactly what
fails at Site 2 (§7, C2).

**Remark 3.2 (conditional framing in training).** In the training loop, $g_k$
and $s$ are not constants but are measurable w.r.t.
$\mathcal{F}$ = (weights entering the window, the window's minibatches):
weights are fixed during the window (accumulation touches only `.grad`), and
$s$ is computed from $g_1$. The window's dither draws are generated from
dedicated streams seeded by (step, parameter) and independent of
$\mathcal{F}$. Hence L2 and everything downstream hold *conditionally on
$\mathcal{F}$*, with $c_k$ $\mathcal{F}$-measurable; unconditional
unbiasedness follows by the tower property. All P2 variances are conditional
variances given $\mathcal{F}$ — which is precisely what the fp32-shadow window
instrumentation estimates.

**Remark 3.3 (clamp caveat).** With clamping, $G_k$ is still on-grid (the
clamp maps to $\pm q_{\max} s$, a grid point), so (i)–(ii) survive; but the
step-error decomposition (iii) acquires an additional dither-*dependent*
saturation term whenever $|q| > q_{\max}$. All exact statements below are
therefore claims on the event that no coordinate of the window saturates.
Saturation is counted in code (`GradAccumQuantizer.saturated`,
`Muon.mq_saturated`) and the measured rates are reported alongside the
mechanism figure; at $\texttt{headroom} \ge 1$ the Site-1 scale is sized for
the whole window, making saturation rare by construction.

---

## 4. P1 — Variance domination for negatively associated dither

**Definition 4.1 (Joag-Dev and Proschan, 1983).** Random variables
$X_1, \dots, X_m$ are *negatively associated* (NA) if for every pair of
disjoint index sets $A, B \subset \{1,\dots,m\}$ and every pair of
coordinatewise-nondecreasing functions $\varphi: \mathbb{R}^A \to \mathbb{R}$,
$\psi: \mathbb{R}^B \to \mathbb{R}$ (with the covariance defined),

$$\mathrm{Cov}\big(\varphi(X_i,\, i \in A),\; \psi(X_j,\, j \in B)\big) \;\le\; 0.$$

We use three closure facts from Joag-Dev and Proschan (1983): **(P6)**
coordinatewise-nondecreasing functions of disjoint subsets of NA variables are
NA; **(P7)** the union of independent collections of NA variables is NA; and
their **Theorem 2.11**: a random vector uniformly distributed over the
permutations of a fixed vector of real numbers (sampling without replacement)
is NA. Independent random variables are trivially NA.

**Lemma 4.2 (the antithetic window is NA).** *Let $U_1, \dots, U_{m/2}$ be
i.i.d. $\mathcal{U}[0,1)$ and $u_{2l-1} = U_l$, $u_{2l} = 1 - U_l$. Then
$(u_1, \dots, u_m)$ is NA.*

**Proof.** First the pair $(U, 1-U)$: for nondecreasing $\varphi, \psi$, set
$\tilde\psi(x) := \psi(1-x)$, which is nonincreasing. With $U'$ an independent
copy of $U$,

$$\big(\varphi(U) - \varphi(U')\big)\big(\tilde\psi(U) - \tilde\psi(U')\big) \;\le\; 0
\quad \text{pointwise},$$

because the first factor has the sign of $U - U'$ and the second the opposite
sign. Taking expectations and expanding, $0 \ge \mathbb{E}[\cdot] =
2\,\mathrm{Cov}(\varphi(U), \tilde\psi(U)) = 2\,\mathrm{Cov}(\varphi(u_{2l-1}),
\psi(u_{2l}))$ (Chebyshev's correlation inequality). For a 2-vector the NA
definition only involves the two singletons, so each pair is NA. The window is
the union of the $m/2$ mutually independent NA pairs, hence NA by (P7).
$\blacksquare$

**Lemma 4.3 (the strat/LHS window is NA).** *Let $\pi$ be a uniform random
permutation of $\{0,\dots,m-1\}$, $\xi_1,\dots,\xi_m$ i.i.d.
$\mathcal{U}[0,1)$ independent of $\pi$, and $u_k = (\pi(k) + \xi_k)/m$. Then
$(u_1,\dots,u_m)$ is NA, and each $u_k \sim \mathcal{U}[0,1)$ exactly.*

**Proof.** $(\pi(1), \dots, \pi(m))$ is a permutation distribution, hence NA
by Theorem 2.11 of Joag-Dev–Proschan. The $\xi_k$ are independent of each
other and of $\pi$, so the combined family $(\pi(1),\dots,\pi(m),
\xi_1,\dots,\xi_m)$ is NA by iterated (P7). Each $u_k = (\pi(k)+\xi_k)/m$ is a
nondecreasing function of the disjoint pair $\{\pi(k), \xi_k\}$, so
$(u_1,\dots,u_m)$ is NA by (P6). Marginal: $\pi(k)$ is uniform on
$\{0,\dots,m-1\}$ and independent of $\xi_k$, so $u_k$ is uniform on $[0,1)$.
$\blacksquare$

**Theorem P1 (variance domination).** *Let $c_1, \dots, c_m \in [0,1)$ be
fixed and $f_k(u) = \mathbf{1}\{u \ge 1-c_k\} - c_k$. Let $(u_1,\dots,u_m)$
be any NA vector with $\mathcal{U}[0,1)$ marginals. Then the window error
$S = \sum_k f_k(u_k)$ of Lemma L2 satisfies $\mathbb{E}[S] = 0$ and*

$$\mathrm{Var}(S) \;=\; \sum_{k=1}^m c_k(1-c_k) \;+\; \sum_{j \ne k}
\mathrm{Cov}\big(f_j(u_j), f_k(u_k)\big) \;\le\; \sum_{k=1}^m c_k(1-c_k)
\;=\; \mathrm{Var}_{\mathrm{iid}}(S).$$

*In particular, by Lemmas 4.2 and 4.3, the antithetic and strat windows
satisfy $\mathrm{Var}(S) \le \mathrm{Var}_{\mathrm{iid}}(S)$ for **every**
threshold profile $c_1,\dots,c_m$.*

**Proof.** $\mathbb{E}[S] = 0$ by Corollary 3.1. Expand the variance of the
sum. Each $f_k$ is bounded and nondecreasing, so for $j \ne k$ the NA property
applied to $A = \{j\}$, $B = \{k\}$, $\varphi = f_j$, $\psi = f_k$ gives
$\mathrm{Cov}(f_j(u_j), f_k(u_k)) \le 0$. Each diagonal term is the Bernoulli
variance $c_k(1-c_k)$. $\blacksquare$

**Remark 4.1 (this is a reduction, not a new theorem).** For the strat window,
Theorem P1 is the Latin-hypercube monotonicity theorem of McKay, Beckman and
Conover (1979) — LHS does not increase variance for integrands monotone in
each input — specialized via L2 to the additive integrand
$S(u_1,\dots,u_m) = \sum_k f_k(u_k)$, which is coordinatewise nondecreasing.
(MBC state the homogeneous case, one integrand replicated across runs; the
heterogeneous additive case above is the form used spatially by Suresh et al.
(2022).) We present the NA proof because it is self-contained and covers the
antithetic window with the same stroke. The theorem's value here is entirely
in L2's licensing of it: without frozen thresholds there is no fixed monotone
integrand to which it could be applied.

**Proposition 4.4 (latperm: conditional guarantee only).** *Let $u_k =
\mathrm{frac}(U + \pi(k)/m)$ with $U \sim \mathcal{U}[0,1)$ and $\pi$ an
independent uniform permutation. Then:*

*(i) Conditionally on $U$, $(u_1,\dots,u_m)$ is a uniform random assignment
(permutation distribution) of the point set $P_U = \{\mathrm{frac}(U + s/m):
s = 0,\dots,m-1\}$, hence NA given $U$; consequently
$\mathrm{Cov}(f_j(u_j), f_k(u_k) \mid U) \le 0$ and*

$$\mathrm{Var}(S \mid U) \;\le\; \sum_k \mathrm{Var}\big(f_k(u_k) \mid U\big).$$

*(ii) Unconditionally, with $\mu_k(U) := \mathbb{E}[f_k(u_k) \mid U] =
\frac1m \sum_{s=0}^{m-1} f_k(\mathrm{frac}(U + s/m))$ — the shift-$U$
lattice-rule error of integrand $k$ —*

$$\mathrm{Var}(S) \;\le\; \mathrm{Var}_{\mathrm{iid}}(S) \;+\;
\sum_{j \ne k} \mathrm{Cov}\big(\mu_j(U), \mu_k(U)\big). \tag{4.1}$$

*(iii) The last term of (4.1) is not sign-controlled, and
$\mathrm{Var}(S) > \mathrm{Var}_{\mathrm{iid}}(S)$ occurs: explicit threshold
profiles for $m = 2, 3, 4, 8$ are given in N1. Latperm therefore has **no**
universal $\mathrm{Var} \le \mathrm{Var}_{\mathrm{iid}}$ guarantee.*

**Proof.** (i) Given $U$, the point set $P_U$ is a fixed vector of $m$ reals
and $\pi$ assigns its elements to coordinates uniformly at random: this is
exactly the permutation distribution, NA by Theorem 2.11 of
Joag-Dev–Proschan; the displayed inequality is Theorem P1's argument run
conditionally. (ii) By the law of total variance and (i),

$$\mathrm{Var}(S) = \mathbb{E}_U\,\mathrm{Var}(S \mid U) +
\mathrm{Var}_U\Big(\sum_k \mu_k(U)\Big)
\;\le\; \sum_k \mathbb{E}_U\,\mathrm{Var}(f_k(u_k) \mid U) +
\mathrm{Var}_U\Big(\sum_k \mu_k(U)\Big),$$

and $\mathbb{E}_U \mathrm{Var}(f_k \mid U) = \mathrm{Var}(f_k(u_k)) -
\mathrm{Var}_U(\mu_k(U)) = c_k(1-c_k) - \mathrm{Var}_U(\mu_k)$, again by total
variance. Collecting terms, the diagonal $\mathrm{Var}_U(\mu_k)$ cancel and
the cross-covariances remain, giving (4.1). (iii) See N1. $\blacksquare$

**Remark 4.2 (why strat repairs latperm).** Latperm randomizes the
*assignment* of lattice points to window slots but moves the point set as one
rigid body with the single shift $U$; the shared fluctuation $\mu_k(U)$ — how
many lattice points land in each threshold's arc — survives the permutation
averaging, and its cross-terms in (4.1) can be positive when the thresholds'
arcs align. Strat replaces the rigid shift by an independent within-stratum
jitter $\xi_k$ per draw, which destroys exactly this shared term and restores
unconditional NA (Lemma 4.3). Empirically (RESULTS.md, ordering controls,
jobs 51785-88) latperm tracks lattice while strat is best-or-tied at both
sites — consistent with this account, though the per-mode gaps there
($\le 0.008$) are at single-seed noise floor and we do not lean on them.

**Proposition 4.5 (universal bound for arbitrary integrands; Stein 1987, Owen
1997).** *Let $f_1, \dots, f_m: [0,1) \to \mathbb{R}$ be arbitrary
square-integrable centered integrands (no monotonicity), and let
$(u_1,\dots,u_m)$ be the strat/LHS vector. Then*

$$\mathrm{Var}\Big(\sum_k f_k(u_k)\Big) \;\le\; \frac{m}{m-1}\,
\sum_k \mathrm{Var}(f_k) \;=\; \frac{m}{m-1}\,\mathrm{Var}_{\mathrm{iid}},$$

*with $\frac{m}{m-1} = \frac{8}{7} \approx 1.143$ at the paper's window
$m = 8$. The constant is attained (in the class of arbitrary integrands) by
anti-aligned stratum profiles, and cannot be approached by the monotone
integrands produced by L2, for which the constant improves to $1$ (Theorem
P1).*

**Proof.** Deferred to the end of §5.3, where the required variance
decomposition is established; the attainment and impossibility claims are
Remark 5.2. For the classical homogeneous statement see Owen (1997) (bound
$\mathrm{Var}_{\mathrm{LHS}} \le \frac{n}{n-1}\sigma^2/n$ for one integrand)
and Stein (1987) for the asymptotic decomposition. $\blacksquare$

---

## 5. P2 — Closed-form window-error variance per mode

Throughout this section $c_1, \dots, c_m \in [0,1)$ are the fixed thresholds
of L2 and $S = \sum_k f_k(u_k)$. All formulas were verified by Monte Carlo
before finalization (Appendix A): across three random $m=8$ profiles, all
special profiles of N1, and a direct pairwise check of (5.2), the maximum
discrepancy between formula and MC was $1.6 \times 10^{-3}$ at $N = 4 \times
10^6$ samples — within MC standard error ($\approx \sigma^2\sqrt{2/N}$).

### 5.1 iid

$$\boxed{\;\mathrm{Var}_{\mathrm{iid}}(S) \;=\; \sum_{k=1}^m c_k (1 - c_k)\;}$$

(independent centered Bernoulli($c_k$) errors).

### 5.2 Antithetic pairs

**Proposition P2-a.** *For a pair $(u_j, u_k) = (U, 1-U)$, $U \sim
\mathcal{U}[0,1)$,*

$$\mathrm{Cov}\big(f_j(u_j), f_k(u_k)\big)
\;=\; \max(0,\, c_j + c_k - 1) \;-\; c_j c_k
\;=\; -\min\big(c_j c_k,\; (1-c_j)(1-c_k)\big) \;\le\; 0, \tag{5.2}$$

*and for the antithetic window with pairs $(2l-1, 2l)$,*

$$\boxed{\;\mathrm{Var}_{\mathrm{anti}}(S) \;=\; \sum_{k=1}^m c_k(1-c_k)
\;+\; 2 \sum_{l=1}^{m/2} \Big[ \max(0,\, c_{2l-1} + c_{2l} - 1) - c_{2l-1}
c_{2l} \Big].\;}$$

**Proof.** Since both $f$'s are centered, the covariance is
$\mathbb{E}[\mathbf{1}\{U \ge 1-c_j\}\,\mathbf{1}\{1-U \ge 1-c_k\}] - c_j c_k$.
The second event is $\{U \le c_k\}$, so the product event is $U \in [1-c_j,\,
c_k]$, an interval of length $\max(0, c_k - (1-c_j)) = \max(0, c_j + c_k -1)$
(endpoint inclusion is Lebesgue-null). For the second form: if $c_j + c_k \le
1$ the expression is $-c_jc_k$; otherwise it is $c_j + c_k - 1 - c_jc_k =
-(1-c_j)(1-c_k)$; and $c_jc_k \le (1-c_j)(1-c_k) \iff c_j + c_k \le 1$.
Nonpositivity is then evident — consistent with Lemma 4.2 + Theorem P1. Pairs
are independent across $l$, so covariances across different pairs vanish.
$\blacksquare$

### 5.3 Strat / Latin hypercube

Define, for $k \in \{1..m\}$ and stratum $s \in \{0, \dots, m-1\}$
(i.e. $u \in [s/m, (s+1)/m)$):

$$p_{k,s} \;=\; \mathrm{clip}\big(m c_k - (m-1-s),\; 0,\; 1\big), \qquad
a_{k,s} \;=\; p_{k,s} - c_k, \qquad v_{k,s} \;=\; p_{k,s}(1 - p_{k,s}),$$

the conditional firing probability, stratum mean of $f_k$, and stratum
variance of $f_k$. (Derivation of $p_{k,s}$: in stratum $s$, $u = (s+\xi)/m$
with $\xi \sim \mathcal{U}[0,1)$, and $u \ge 1-c_k \iff \xi \ge m(1-c_k) - s$,
an event of probability $\mathrm{clip}(1 - m(1-c_k) + s, 0, 1)$.) Note
$\frac1m \sum_s p_{k,s} = c_k$ and $\frac1m\sum_s a_{k,s} = 0$ (rows of $a$
are centered), and each row $s \mapsto a_{k,s}$ is nondecreasing (stratum
means of a nondecreasing function).

**Proposition P2-b.** *For the strat window,*

$$\boxed{\;\mathrm{Var}_{\mathrm{strat}}(S) \;=\;
\underbrace{\frac1m \sum_{k=1}^{m}\sum_{s=0}^{m-1} v_{k,s}}_{\text{within-stratum}}
\;+\; \underbrace{\frac{1}{m-1} \sum_{k=1}^{m}\sum_{s=0}^{m-1}
\big(a_{k,s} - \bar a_s\big)^2}_{\text{permutation term (Hoeffding)}}, \qquad
\bar a_s = \frac1m \sum_k a_{k,s}.\;}$$

**Proof.** Condition on $\pi$. Given $\pi$, the $u_k$ are independent (fresh
$\xi_k$), with $u_k$ uniform on stratum $\pi(k)$; hence
$\mathrm{Var}(S \mid \pi) = \sum_k v_{k, \pi(k)}$ and $\mathbb{E}[S \mid \pi]
= \sum_k a_{k,\pi(k)} =: T$. By the law of total variance,

$$\mathrm{Var}(S) = \mathbb{E}_\pi \sum_k v_{k,\pi(k)} + \mathrm{Var}_\pi(T)
= \frac1m \sum_{k,s} v_{k,s} + \mathrm{Var}_\pi(T),$$

since $\pi(k)$ is marginally uniform. For $\mathrm{Var}_\pi(T)$ we prove the
row-centered case of Hoeffding's (1951) permutation-variance formula. Row
centering gives $\mathbb{E}[T] = 0$, and

$$\mathbb{E}[T^2] = \sum_k \mathbb{E}\big[a_{k,\pi(k)}^2\big] + \sum_{k \ne l}
\mathbb{E}\big[a_{k,\pi(k)} a_{l,\pi(l)}\big]
= \frac1m \sum_{k,s} a_{k,s}^2 + \frac{1}{m(m-1)} \sum_{k \ne l}\,
\sum_{s \ne s'} a_{k,s}\, a_{l,s'},$$

because $(\pi(k), \pi(l))$ is uniform over ordered pairs of distinct strata.
Using row-centering, $\sum_{s \ne s'} a_{k,s} a_{l,s'} = \big(\sum_s
a_{k,s}\big)\big(\sum_{s'} a_{l,s'}\big) - \sum_s a_{k,s} a_{l,s} = -\sum_s
a_{k,s} a_{l,s}$. Then

$$\sum_{k \ne l} \Big(-\sum_s a_{k,s}a_{l,s}\Big)
= -\sum_s \Big[\Big(\sum_k a_{k,s}\Big)^2 - \sum_k a_{k,s}^2\Big]
= \sum_{k,s} a_{k,s}^2 - m^2 \sum_s \bar a_s^2,$$

so

$$\mathrm{Var}_\pi(T) = \frac1m \sum_{k,s} a_{k,s}^2 +
\frac{1}{m(m-1)}\Big(\sum_{k,s} a_{k,s}^2 - m^2 \sum_s \bar a_s^2\Big)
= \frac{1}{m-1}\Big(\sum_{k,s} a_{k,s}^2 - m \sum_s \bar a_s^2\Big),$$

which equals $\frac{1}{m-1}\sum_{k,s} (a_{k,s} - \bar a_s)^2$ by expanding the
square (the cross term contributes $-2m\sum_s \bar a_s^2$ and the last term
$+m \sum_s \bar a_s^2$). $\blacksquare$

**Proof of Proposition 4.5.** The derivation above never used the specific
indicator form: for arbitrary square-integrable centered $f_k$ with stratum
arrays $(a_{k,s}, v_{k,s})$,

$$\mathrm{Var}_{\mathrm{strat}}(S) = \frac1m\sum_{k,s} v_{k,s} +
\frac{1}{m-1}\Big(\sum_{k,s}a_{k,s}^2 - m\sum_s \bar a_s^2\Big)
\;\le\; \frac1m\sum_{k,s} v_{k,s} + \frac{1}{m-1}\sum_{k,s}a_{k,s}^2,$$

while the law of total variance (uniform stratum index) gives
$\mathrm{Var}_{\mathrm{iid}} = \sum_k \mathrm{Var}(f_k) = \frac1m\sum_{k,s}
v_{k,s} + \frac1m \sum_{k,s} a_{k,s}^2$. Since $\frac{1}{m-1} =
\frac{m}{m-1}\cdot\frac1m$ and the $v$-term's coefficient $\frac1m \le
\frac{m}{m-1}\cdot\frac1m$,

$$\mathrm{Var}_{\mathrm{strat}}(S) \;\le\; \frac{m}{m-1}\Big(\frac1m\sum v +
\frac1m \sum a^2\Big) = \frac{m}{m-1}\,\mathrm{Var}_{\mathrm{iid}}.
\qquad\blacksquare$$

**Remark 5.2 (attainment; why the monotone class gets constant 1).** Equality
requires $v \equiv 0$ (integrands constant on each stratum) and $\bar a_s = 0$
for all $s$ (columns of the stratum-mean array centered) with $a \not\equiv
0$: *anti-aligned profiles*. Witness for even $m$: take $v \equiv 0$ and give
half the integrands the stratum profile $+\rho$ and half $-\rho$, where $\rho
\in \mathbb{R}^m$ is any nonzero centered vector (realizable by
piecewise-constant $f_k$). Then $\bar a_s = 0$ for every $s$, so
$\mathrm{Var}_{\mathrm{iid}} = \frac1m \sum_{k,s} a_{k,s}^2 = \|\rho\|^2$
while $\mathrm{Var}_{\mathrm{strat}} = \frac{1}{m-1} \sum_{k,s} a_{k,s}^2 =
\frac{m}{m-1}\|\rho\|^2$: the bound is attained exactly. (At $m = 2$:
profiles $(-\alpha, \alpha)$ and $(\alpha, -\alpha)$ give $T \in \{\pm
2\alpha\}$, ratio $2$.) But anti-alignment requires integrands of
opposite monotonicity: after the L2 reduction every $f_k$ is nondecreasing,
so every row $a_{k,\cdot}$ is nondecreasing and centered, hence $a_{k,0} \le
0$ for all $k$; if additionally all column means vanish, then column $0$
consists of nonpositive entries averaging to $0$, so $a_{k,0} = 0$ for all
$k$, and each row is nondecreasing, centered, and starts at $0$ — forcing
$a \equiv 0$. The bound $\frac{m}{m-1}$ is therefore never active for L2's
integrands: the operative constant is $1$ (Theorem P1). At $m = 8$ the
general-integrand worst case would be $\frac{8}{7} \approx 1.143$.

### 5.4 Shifted rank-1 lattice (and any fixed assignment, e.g. vdc)

Define the circular arc-overlap function, for $a, b \in [0,1]$ and $\delta \in
[0,1)$:

$$\mathrm{ov}(a, b, \delta) \;=\; \big|\,[0,a) \cap \big([\delta, \delta+b)
\bmod 1\big)\,\big| \;=\; \max\big(0, \min(a, \delta+b) - \delta\big) +
\max\big(0, \min(a, \delta + b - 1)\big). \tag{5.1}$$

**Proposition P2-c.** *For the lattice window $u_i = \mathrm{frac}(U + i/m)$
and any ordered pair $j \ne k$ with offset $d_{jk} = ((k-j) \bmod m)/m$,*

$$\mathrm{Cov}\big(f_j(u_j), f_k(u_k)\big) \;=\;
\mathrm{ov}\big(c_j,\, c_k,\, \delta_{jk}\big) \;-\; c_j c_k,
\qquad \delta_{jk} = \mathrm{frac}\big(c_j - c_k - d_{jk}\big), \tag{5.3}$$

*and*

$$\boxed{\;\mathrm{Var}_{\mathrm{lat}}(S) \;=\; \sum_k c_k(1-c_k) \;+\;
\sum_{j \ne k} \Big[\mathrm{ov}\big(c_j, c_k, \delta_{jk}\big) - c_j c_k\Big].\;}$$

*More generally, for any deterministic assignment $i \mapsto \sigma(i)$ of the
same point set — vdc is $\sigma = $ bit reversal — the same formulas hold with
$d_{jk} = ((\sigma(k) - \sigma(j)) \bmod m)/m$.*

**Proof.** Set $V := u_j \sim \mathcal{U}[0,1)$; then $u_k =
\mathrm{frac}(V + d_{jk})$. Both $f$'s centered, so the covariance is
$\Pr(A \cap B) - c_jc_k$ with $A = \{V \in [1-c_j, 1)\}$ and $B =
\{\mathrm{frac}(V + d) \in [1-c_k, 1)\} = \{V \in [1 - c_k - d,\, 1-d) \bmod
1\}$, two arcs on the circle $\mathbb{R}/\mathbb{Z}$ of lengths $c_j$ and
$c_k$. Rotate the circle by $c_j - 1$ (rotation preserves Lebesgue measure
mod 1): $A \mapsto [0, c_j)$ and $B \mapsto [\delta, \delta + c_k) \bmod 1$
with $\delta = \mathrm{frac}\big((1 - c_k - d) - (1 - c_j)\big) =
\mathrm{frac}(c_j - c_k - d)$. The overlap of $[0,a)$ with the possibly
wrapping arc $[\delta, \delta+b) \bmod 1$ splits into the main piece
$[\delta, \min(\delta+b, 1))$, contributing $\max(0, \min(a, \delta+b) -
\delta)$, and the wrap piece $[0, \max(0, \delta+b-1))$, contributing
$\max(0, \min(a, \delta+b-1))$ — which is (5.1). The generalization is the
same computation with the assigned offsets. $\blacksquare$

### 5.5 Latperm

**Proposition P2-d.** *For the latperm window $u_i = \mathrm{frac}(U +
\pi(i)/m)$: for any ordered pair $j \ne k$,*

$$\mathrm{Cov}\big(f_j(u_j), f_k(u_k)\big) \;=\; \frac{1}{m-1}
\sum_{r=1}^{m-1} \Big[\mathrm{ov}\big(c_j, c_k,
\mathrm{frac}(c_j - c_k - r/m)\big) - c_j c_k\Big], \tag{5.4}$$

*i.e. the average of the lattice covariances (5.3) over the $m-1$ nonzero
offsets, and*

$$\boxed{\;\mathrm{Var}_{\mathrm{latperm}}(S) \;=\; \sum_k c_k(1-c_k) +
\sum_{j\ne k} \mathrm{Cov}\big(f_j(u_j), f_k(u_k)\big) \text{ with (5.4)}.\;}$$

**Proof.** Condition on $r := (\pi(k) - \pi(j)) \bmod m$. For a uniform
permutation, $(\pi(j), \pi(k))$ is uniform over ordered pairs of distinct
strata, so $r$ is uniform on $\{1, \dots, m-1\}$. Given $\pi$ (hence $r$),
$V := \mathrm{frac}(U + \pi(j)/m)$ is uniform (as $U$ is, independent of
$\pi$) and $u_k = \mathrm{frac}(V + r/m)$: conditionally on $r$, the pair
$(u_j, u_k)$ has exactly the lattice-pair law with offset $d = r/m$, and both
conditional means vanish. Average (5.3) over $r$. $\blacksquare$

**Remark 5.3 (using P2 as predictions).** Given the logged per-coordinate
thresholds $c_k = \mathrm{frac}(g_k/s)$ of a training window (or their
histogram), the five boxed formulas produce exact predicted values of
$\mathrm{Var}(S \mid \mathcal{F})$ per mode, at zero simulation cost. The
mechanism figure compares these predictions against the measured normalized
window error $(G_m - \sum_k g_k)/s$ from the fp32-shadow instrumentation; the
comparison is a direct end-to-end test of the fixed-threshold model
(saturation and fp32 dither edges excluded; §7). Any systematic deviation
falsifies the model rather than the arithmetic — the arithmetic is
MC-verified (Appendix A).

---

## 6. N1 — No variance guarantee for the shifted lattice (or latperm)

**Proposition N1.** *(i) For $m = 2$, the shifted-lattice window is the pair
$(u_1, u_2) = (V, \mathrm{frac}(V + \tfrac12))$, $V \sim \mathcal{U}[0,1)$,
and*

$$\mathrm{Cov}\big(f_1(u_1), f_2(u_2)\big) \;=\;
\begin{cases}
-\,c_1 c_2, & \max(c_1, c_2) \le \tfrac12,\\[2pt]
-\,(1-c_1)(1-c_2), & \min(c_1, c_2) > \tfrac12,\\[2pt]
\min\big(\max(c_1,c_2) - \tfrac12,\; \min(c_1,c_2)\big) - c_1 c_2, & \text{otherwise.}
\end{cases}$$

*(ii) In the third branch the covariance is **strictly positive** on an open
region: for $\tfrac12 < c_1 < 1$ and $0 < c_2 \le c_1 - \tfrac12$,*

$$\mathrm{Cov} = c_2 (1 - c_1) > 0 .$$

*In particular at $(c_1, c_2) = (\tfrac34, \tfrac14)$:
$\mathrm{Cov} = \tfrac1{16}$ and*

$$\mathrm{Var}_{\mathrm{lat}}(S) = \tfrac38 + 2\cdot\tfrac1{16} = \tfrac12
\;=\; \tfrac43 \cdot \mathrm{Var}_{\mathrm{iid}}(S)
\;>\; \mathrm{Var}_{\mathrm{iid}}(S) = \tfrac38 .$$

*So the shifted lattice admits no universal
$\mathrm{Var} \le \mathrm{Var}_{\mathrm{iid}}$ guarantee. (iii) At $m=2$,
latperm coincides with the lattice — the offset $(\pi(2)-\pi(1)) \bmod 2 = 1$
deterministically — so the same profile defeats latperm, proving Proposition
4.4 (iii) at $m = 2$; coordinate-ascent on the closed form (5.4) produces
MC-confirmed latperm counterexamples at $m = 3, 4, 8$ as well (below).*

**Proof.** (i) Both $f$'s centered, so $\mathrm{Cov} = \Pr(A \cap B) - c_1c_2$
with $A = [1-c_1, 1)$ and $B = \{V : \mathrm{frac}(V + \tfrac12) \ge 1-c_2\}$.
If $c_2 \le \tfrac12$, $B = [\tfrac12 - c_2, \tfrac12) \subset [0, \tfrac12)$.
If also $c_1 \le \tfrac12$, then $A \subset [\tfrac12, 1)$ and $\Pr(A\cap B) =
0$: first branch. If $c_1 > \tfrac12 \ge c_2$, $A \cap B = [\max(1-c_1,
\tfrac12 - c_2), \tfrac12)$ of length $\min(c_1 - \tfrac12, c_2)$ (when
positive; it is, since $c_1 > \tfrac12$): third branch with $c_1 =
\max$. If $c_2 > \tfrac12$, $B$ wraps: $B = [0, \tfrac12) \cup [\tfrac32 -
c_2, 1)$; if also $c_1 > \tfrac12$, $A \cap B = [1-c_1, \tfrac12) \cup
[\tfrac32 - c_2, 1)$ of total length $(c_1 - \tfrac12) + (c_2 - \tfrac12) =
c_1 + c_2 - 1$: second branch, $= -(1-c_1)(1-c_2)$ after subtracting
$c_1c_2$. The remaining mixed case ($c_2 > \tfrac12 \ge c_1$) is the third
branch by the symmetric computation. (ii) With $c_1 = \max$: if $c_2 \le c_1 -
\tfrac12$ then $\min(c_1 - \tfrac12, c_2) = c_2$ and $\mathrm{Cov} = c_2 -
c_1c_2 = c_2(1-c_1) > 0$ strictly on the stated open region. At
$(\tfrac34,\tfrac14)$: $\mathrm{Cov} = \tfrac14 \cdot \tfrac14 = \tfrac1{16}$,
and $\mathrm{Var}_{\mathrm{iid}} = \tfrac34\cdot\tfrac14 +
\tfrac14\cdot\tfrac34 = \tfrac38$, so $\mathrm{Var}_{\mathrm{lat}} =
\tfrac38 + 2\cdot\tfrac1{16} = \tfrac12$. (iii) At $m=2$ the permutation either fixes
or swaps the two points; in both cases $\{u_1, u_2\} = \{V',
\mathrm{frac}(V' + \tfrac12)\}$ with $V'$ uniform, and the ordered pair has
offset $\tfrac12$ either way, so the pair law — hence $\mathrm{Var}(S)$ — is
identical to the lattice's. $\blacksquare$

**Remark 6.1 (the failure is not a knife-edge, and the lattice can also be
perfect).** The positive-covariance region of (ii) has positive measure; on
uniformly random $m=8$ threshold profiles the closed forms give
$\mathrm{Var}_{\mathrm{lat}} > \mathrm{Var}_{\mathrm{iid}}$ on $10.3\%$ of
20,000 profiles (versus $0$ of 20,000 for antithetic and strat — as Theorem
P1 requires). The sign is profile-dependent, not universally bad: at $c_1 =
c_2 = \tfrac12$ the same pair gives $\mathrm{Cov} = -\tfrac14$ and $S \equiv
0$ — exactly one of the two indicators fires for every $V$ — i.e. the lattice
is *exactly optimal* there. A construction whose variance ranges from $0$ to
$\tfrac43 \mathrm{Var}_{\mathrm{iid}}$ across threshold profiles, with the
profile set by the data ($c_k = \mathrm{frac}(g_k/s)$), supports no
profile-free guarantee — consistent with the lattice/vdc arms' behavior in
training (RESULTS.md): they recover most of the iid$\to$correlated gain on
average yet trail the two guaranteed modes in the 4-bit cells and flip order
with pipeline.

**Remark 6.2 (worst-case profile at $m = 8$: the comonotone alignment).** Take
$c_k = \mathrm{frac}\big(1 - \tfrac{1}{2m} - \tfrac{k}{m}\big)$, $k = 1,
\dots, m$; at $m=8$: $c = (\tfrac{13}{16}, \tfrac{11}{16}, \tfrac{9}{16},
\tfrac{7}{16}, \tfrac{5}{16}, \tfrac{3}{16}, \tfrac{1}{16}, \tfrac{15}{16})$.
In shift space every event $\{u_k \ge 1-c_k\} = \{U \in [1 - c_k - k/m,\,
1 - k/m) \bmod 1\}$ becomes an arc with the **common left endpoint**
$\tfrac{1}{2m}$: the events are nested, so the $m$ step errors are
*comonotone* — the Fréchet–Hoeffding maximal coupling of their Bernoulli
marginals, the worst dependence structure any dither scheme could produce.
Writing $W = \mathrm{frac}(U - \tfrac1{2m})$, $S_{\mathrm{lat}} = N(W) -
\sum_k c_k$ with $N(W) = \#\{k : W < c_k\}$; at $m=8$, $N$ takes the values
$8, 7, \dots, 1, 0$ with probabilities $\tfrac1{16}, \tfrac2{16}, \dots,
\tfrac2{16}, \tfrac1{16}$, giving $\mathbb{E}N = 4$, $\mathbb{E}N^2 = 21.5$,
and

$$\mathrm{Var}_{\mathrm{lat}} = 5.5 \;=\; 4.09 \times
\mathrm{Var}_{\mathrm{iid}} = 4.09 \times 1.34375,$$

while on the same profile antithetic gives $0.5$, strat $0.75$, and latperm
$1.0$ — all below iid (all five values from the P2 closed forms, MC-confirmed;
Appendix A).

**Remark 6.3 (latperm counterexamples beyond $m=2$).** Coordinate ascent on
the closed form (5.4) (maximizing
$\mathrm{Var}_{\mathrm{latperm}}/\mathrm{Var}_{\mathrm{iid}}$ over $c$)
produced the following MC-confirmed witnesses:

| $m$ | profile $c$ | $\mathrm{Var}_{\mathrm{iid}}$ | $\mathrm{Var}_{\mathrm{latperm}}$ | ratio |
|---|---|---|---|---|
| 2 | $(0.75,\ 0.25)$ | $0.375$ | $0.5$ | $1.333$ |
| 3 | $(0.906,\ 0.232,\ 0.906)$ | $0.3485$ | $0.4181$ | $1.200$ |
| 4 | $(0.952,\ 0.952,\ 0.202,\ 0.952)$ | $0.2983$ | $0.3426$ | $1.149$ |
| 8 | $(0.018, 0.018, 0.018, 0.018, 0.002, 0.002, 0.002, 0.906)$ | $0.1619$ | $0.1717$ | $1.061$ |

So Proposition 4.4 (iii) is not an $m=2$ artifact: adversarial threshold
profiles defeat latperm at the paper's window size $m=8$ as well, though none
of 20,000 *uniformly random* $m=8$ profiles did — latperm's failure region is
thin, the lattice's is not ($10.3\%$), and antithetic/strat have none. This
three-way stratification (guaranteed / thin failure region / broad failure
region) is the theory's ordering prediction for the mode families, to be
compared only against the resolved empirical splits (iid vs. everything,
$\sim 0.03$–$0.05$ in loss) and not against the sub-noise per-mode gaps.

---

## 7. What the theory does and does not cover (caveats)

**C1 — Clamping / saturation.** All exact statements hold on the event that no
rounding in the window saturates the index clamp $[-q_{\max}, q_{\max}]$
(Remark 3.3). On-grid state (L2 (i)-(ii)) survives clamping, so the
*thresholds* remain frozen even after a saturated step; what breaks is the
error decomposition L2 (iii), which acquires a dither-dependent saturation
term. The paper reports measured saturation rates (instrumented counters) with
the mechanism figure; the idealized analysis is a good model exactly when
those rates are near zero.

**C2 — Site 2 (mq) is adaptive: no clean guarantee, and unbiasedness itself
is open.** At Site 2 the grid $s_t$ is recomputed each step from the
*pre-round* buffer, and $\mathrm{buf}_t$ is a function of all earlier dither
draws. Consequently the threshold $c_t = \mathrm{frac}\big((\beta\,
\mathrm{buf}_t + G_t)/s_t\big)$ is a random variable *correlated with the
dither draws of the same block*, L2 fails on both counts (no frozen grid; no
dither-independent thresholds), and **Theorem P1 does not formally apply**.
What survives:

- *iid mode*: $u_t$ is independent of $\mathcal{F}_t$ (the past), so each
  step is conditionally unbiased, $\mathbb{E}[\mathrm{buf}_{t+1} \mid
  \mathcal{F}_t] = \beta\,\mathrm{buf}_t + G_t$ on the no-clamp event: the
  buffer errors form a martingale-difference sequence.
- *window modes*: the **first step of each block** is conditionally unbiased
  (fresh $U$, $\pi$, $\xi$, independent of $\mathcal{F}_t$, marginally
  uniform). At later phases the dither is partially (latperm/strat: the
  remaining strata) or **fully** (lattice: $u_{t+1} = \mathrm{frac}(u_t +
  \tfrac1m)$ is $\mathcal{F}_{t+1}$-measurable) determined by earlier draws,
  while $c_t$ is correlated with those same draws; hence per-step conditional
  bias is generic, and even *window-total* unbiasedness — automatic at Site 1
  (Corollary 3.1) — is unproven at Site 2.

The paper accordingly claims a *measured* near-zero buffer-error mean at the
mq site (logged mean-trace instrumentation), an empirical statement, not a
theorem; and the observed mq-site benefit of correlated dither is motivated
only heuristically by the $\beta$-contraction keeping consecutive pre-round
values close (buffer autocorrelation $\approx \beta^{m-1} = 0.95^7 \approx
0.70$ across a block), which makes the frozen-threshold model an approximation
whose quality degrades as $\beta \to 0$.

**C3 — Single-coordinate scope.** All statements are per coordinate. The
implementations draw independent dither per coordinate, so *conditional on
$\mathcal{F}$* the coordinates' window errors are independent; but thresholds
$c_k$ are correlated across coordinates through the gradients, and the
downstream optimizer (Newton–Schulz orthogonalization) mixes coordinates.
Cross-coordinate structure is not modeled; the theory predicts per-coordinate
error variance, not its effect on the update direction.

**C4 — Variance is not loss.** No claim is made that smaller
$\mathrm{Var}(S)$ implies smaller validation loss. The theory predicts the
window-error variance exactly (P2); the link from that variance to training
outcomes is the paper's *empirical* mechanism claim (predictions vs. logged
variance, then variance vs. loss across modes), stated as such.

**C5 — Finite-precision dither.** The analysis treats $u \in [0,1)$ exactly.
In fp32, the antithetic reflection $1 - u$ produces $u = 1.0$ exactly when
$u = 0.0$ (probability $\approx 2^{-24}$ per draw), adding a full step;
similarly strat's $(m-1+\xi)/m$ can round to $1.0$ (noted in `gquant.py`,
realized bias $\sim 10^{-15}$). These are measure-zero events of the idealized
model, kept in the implementation for bit-compatibility with logged runs.

**C6 — Cross-window claims.** Thresholds are fixed only *within* a window,
conditional on $\mathcal{F}$; across windows the weights adapt to past
rounding errors. The per-window statements chain across windows only through
the tower property (unbiasedness); no claim is made about the covariance of
errors across windows or about long-run error accumulation dynamics.

---

## Appendix A — Monte Carlo verification of P2 and N1

All closed forms and counterexamples above were verified numerically **before**
this file was finalized (run 2026-08-15, `conda` env `quest`, numpy;
$N = 4\times10^6$ samples per cell unless noted, $8\times10^6$ for the latperm
witnesses, $2\times10^6$ for the m=2/aligned-arc profiles). No formula
required correction against MC at the final revision: every derived expression
below agreed with simulation within Monte Carlo standard error
($\max |{\rm formula} - {\rm MC}| = 1.6\times10^{-3}$, attained on the largest
variance with the largest MC standard error). One *claim from the task plan*
was corrected by this process: latperm is **not** NA and has no unconditional
variance guarantee (Proposition 4.4, N1) — the initial "same permutation
argument as strat" is refuted by the $m=2$ profile $(0.75, 0.25)$ and its
$m=3,4,8$ relatives, all MC-confirmed below.

### A.1 Verification script

```python
"""MC verification of the THEORY.md closed forms (L1, P2, N1).

Window error S = sum_k [ floor(c_k + u_k) - c_k ]  (integer parts irrelevant),
c_k in [0,1) fixed thresholds, u_k the window dither vector per mode.
"""
import numpy as np

rng = np.random.default_rng(20260815)
frac = lambda x: x - np.floor(x)

# ---------------- closed forms ----------------
def var_iid(c):
    c = np.asarray(c, float)
    return float(np.sum(c * (1 - c)))

def cov_anti(cj, ck):
    return max(0.0, cj + ck - 1.0) - cj * ck

def var_anti(c):
    c = np.asarray(c, float)
    v = var_iid(c)
    for l in range(len(c) // 2):
        v += 2 * cov_anti(c[2 * l], c[2 * l + 1])
    return float(v)

def ov(a, b, d):
    """length of [0,a) intersect ([d, d+b) mod 1), for a,b in [0,1], d in [0,1)."""
    return max(0.0, min(a, d + b) - d) + max(0.0, min(a, d + b - 1.0))

def cov_lat(cj, ck, d):
    """Cov(f_j(V), f_k(frac(V+d))), V~U[0,1), f threshold indicators centered."""
    return ov(cj, ck, (cj - ck - d) % 1.0) - cj * ck

def var_lat(c, assign=None):
    c = np.asarray(c, float); m = len(c)
    if assign is None:
        assign = np.arange(m)
    v = var_iid(c)
    for j in range(m):
        for k in range(m):
            if j != k:
                v += cov_lat(c[j], c[k], ((assign[k] - assign[j]) % m) / m)
    return float(v)

def cov_latperm(cj, ck, m):
    return float(np.mean([cov_lat(cj, ck, r / m) for r in range(1, m)]))

def var_latperm(c):
    c = np.asarray(c, float); m = len(c)
    v = var_iid(c)
    for j in range(m):
        for k in range(m):
            if j != k:
                v += cov_latperm(c[j], c[k], m)
    return float(v)

def var_strat(c):
    c = np.asarray(c, float); m = len(c); s = np.arange(m)
    p = np.clip(m * c[:, None] - (m - 1 - s[None, :]), 0.0, 1.0)   # (k, s)
    a = p - c[:, None]                                              # stratum means
    within = np.sum(p * (1 - p)) / m
    abar = a.mean(axis=0)                                           # column means
    perm = np.sum((a - abar[None, :]) ** 2) / (m - 1)               # Hoeffding
    return float(within + perm)

# ---------------- MC samplers ----------------
def mc(mode, c, N=4_000_000, chunks=4, seed=None):
    c = np.asarray(c, float); m = len(c)
    r = np.random.default_rng(seed if seed is not None else rng.integers(2**63))
    n = N // chunks
    tot = tot2 = 0.0; cnt = 0
    for _ in range(chunks):
        if mode == "iid":
            u = r.random((n, m))
        elif mode == "anti":
            U = r.random((n, m // 2))
            u = np.empty((n, m)); u[:, 0::2] = U; u[:, 1::2] = 1.0 - U
        elif mode == "lat":
            u = frac(r.random((n, 1)) + np.arange(m)[None, :] / m)
        elif mode == "vdc":
            nb = m.bit_length() - 1
            br = np.array([int(format(i, f"0{nb}b")[::-1], 2) for i in range(m)])
            u = frac(r.random((n, 1)) + br[None, :] / m)
        elif mode == "latperm":
            P = np.argsort(r.random((n, m)), axis=1)
            u = frac(r.random((n, 1)) + P / m)
        elif mode == "strat":
            P = np.argsort(r.random((n, m)), axis=1)
            u = (P + r.random((n, m))) / m
        S = (np.floor(c[None, :] + u) - c[None, :]).sum(axis=1)
        tot += S.sum(); tot2 += (S ** 2).sum(); cnt += n
    mean = tot / cnt
    var = tot2 / cnt - mean ** 2
    return mean, var

# ---------------- L1 ----------------
print("== L1: E[floor(t+u)] = t iff u ~ U[0,1) ==")
for t in [-2.3, -0.5, 0.0, 0.25, 3.9]:
    u = rng.random(4_000_000)
    print(f"  uniform  t={t:+.2f}: E[floor(t+u)] = {np.floor(t+u).mean():+.4f}")
u = rng.beta(2, 5, 4_000_000)   # non-uniform on [0,1)
t = 0.5
print(f"  beta(2,5) t={t:+.2f}: E[floor(t+u)] = {np.floor(t+u).mean():+.4f}  (should MISS t)")

# ---------------- P2 on random threshold profiles ----------------
print("\n== P2 closed forms vs MC, m=8, random c profiles ==")
hdr = f"  {'mode':8s} {'formula':>10s} {'MC var':>10s} {'|diff|':>9s} {'MC mean':>9s}"
for trial in range(3):
    c = np.round(rng.random(8), 3)
    print(f"  c = {c.tolist()}")
    print(hdr)
    for mode, fn in [("iid", var_iid), ("anti", var_anti), ("lat", var_lat),
                     ("latperm", var_latperm), ("strat", var_strat)]:
        f = fn(c); mean, v = mc(mode, c)
        print(f"  {mode:8s} {f:10.5f} {v:10.5f} {abs(f-v):9.5f} {mean:+9.5f}")

# vdc = lattice with bit-reversed assignment
c = np.round(rng.random(8), 3)
br = np.array([int(format(i, '03b')[::-1], 2) for i in range(8)])
f = var_lat(c, assign=br); mean, v = mc("vdc", c)
print(f"  vdc assign check, c={c.tolist()}: formula {f:.5f}  MC {v:.5f}  |diff| {abs(f-v):.5f}")

# ---------------- N1 profiles ----------------
print("\n== N1: m=2 profiles ==")
for c in ([0.5, 0.5], [0.75, 0.25]):
    print(f"  c = {c}: iid {var_iid(c):.4f}  anti {var_anti(c):.4f}  "
          f"strat {var_strat(c):.4f}  lat {var_lat(c):.4f}  latperm {var_latperm(c):.4f}")
    print(f"    cov_lat(d=1/2) = {cov_lat(c[0], c[1], 0.5):+.4f}")
    for mode in ("iid", "anti", "strat", "lat", "latperm"):
        mean, v = mc(mode, c, N=2_000_000)
        print(f"    MC {mode:8s} var {v:.4f} mean {mean:+.5f}")

print("\n== N1: m=8 aligned-arc profile (all U-space arcs share a left endpoint) ==")
m = 8
c = np.array([frac(1 - 1/16 - k/m) for k in range(1, m + 1)])
print(f"  c = {np.round(c,4).tolist()}")
for mode, fn in [("iid", var_iid), ("anti", var_anti), ("lat", var_lat),
                 ("latperm", var_latperm), ("strat", var_strat)]:
    f = fn(np.array(c)); mean, v = mc(mode, np.array(c), N=2_000_000)
    print(f"  {mode:8s} formula {f:10.5f}  MC {v:10.5f}  |diff| {abs(f-v):.5f}")

# ---------------- pairwise antithetic covariance, direct MC ----------------
print("\n== antithetic pair covariance formula, direct MC ==")
for _ in range(4):
    cj, ck = rng.random(2)
    u = rng.random(3_000_000)
    fj = np.floor(cj + u) - cj
    fk = np.floor(ck + (1.0 - u)) - ck
    emp = float(np.mean(fj * fk) - fj.mean() * fk.mean())
    print(f"  c=({cj:.3f},{ck:.3f}): formula {cov_anti(cj,ck):+.5f}  MC {emp:+.5f}")

# ---------------- N1: adversarial latperm witnesses (coordinate-ascent) ----------------
print("\n== N1: latperm counterexamples at m = 3, 4, 8 (found by coordinate ascent) ==")
for c in ([0.906, 0.232, 0.906],
          [0.952, 0.952, 0.202, 0.952],
          [0.018, 0.018, 0.018, 0.018, 0.002, 0.002, 0.002, 0.906]):
    f_lp, f_i = var_latperm(c), var_iid(c)
    mean, v = mc("latperm", c, N=8_000_000)
    print(f"  m={len(c)} c={c}: iid {f_i:.5f}  latperm formula {f_lp:.5f} "
          f"(ratio {f_lp/f_i:.4f})  MC {v:.5f}  mean {mean:+.6f}")

# ---------------- guarantee sweep (closed forms only) ----------------
print("\n== ratio sweep over 20000 random m=8 profiles (closed forms) ==")
worst = {k: (0.0, None) for k in ("anti", "strat", "lat", "latperm")}
r2 = np.random.default_rng(7)
for _ in range(20000):
    c = r2.random(8)
    vi = var_iid(c)
    if vi < 1e-6:
        continue
    for k, fn in (("anti", var_anti), ("strat", var_strat),
                  ("lat", var_lat), ("latperm", var_latperm)):
        ratio = fn(c) / vi
        if ratio > worst[k][0]:
            worst[k] = (ratio, np.round(c, 3))
print(f"  m/(m-1) = 8/7 = {8/7:.4f}")
for k, (ratio, c) in worst.items():
    print(f"  max Var_{k}/Var_iid = {ratio:.4f}  at c = {c.tolist() if c is not None else None}")
```

### A.2 Output (verbatim, 2026-08-15)

```text
== L1: E[floor(t+u)] = t iff u ~ U[0,1) ==
  uniform  t=-2.30: E[floor(t+u)] = -2.3002
  uniform  t=-0.50: E[floor(t+u)] = -0.5002
  uniform  t=+0.00: E[floor(t+u)] = +0.0000
  uniform  t=+0.25: E[floor(t+u)] = +0.2497
  uniform  t=+3.90: E[floor(t+u)] = +3.9000
  beta(2,5) t=+0.50: E[floor(t+u)] = +0.1094  (should MISS t)

== P2 closed forms vs MC, m=8, random c profiles ==
  c = [0.206, 0.384, 0.948, 0.096, 0.586, 0.705, 0.66, 0.868]
  mode        formula     MC var    |diff|   MC mean
  iid         1.32574    1.32615   0.00040  +0.00029
  anti        0.73950    0.74040   0.00090  -0.00049
  lat         0.86979    0.86898   0.00081  +0.00014
  latperm     0.82693    0.82707   0.00014  -0.00042
  strat       0.73534    0.73553   0.00019  -0.00022
  c = [0.867, 0.808, 0.819, 0.852, 0.926, 0.916, 0.175, 0.138]
  mode        formula     MC var    |diff|   MC mean
  iid         0.95358    0.95272   0.00086  +0.00002
  anti        0.78820    0.78787   0.00033  +0.00010
  lat         0.74200    0.74205   0.00005  +0.00042
  latperm     0.57543    0.57533   0.00009  +0.00043
  strat       0.50069    0.50060   0.00009  -0.00052
  c = [0.255, 0.947, 0.317, 0.428, 0.561, 0.57, 0.756, 0.329]
  mode        formula     MC var    |diff|   MC mean
  iid         1.59810    1.59967   0.00157  -0.00047
  anti        0.54279    0.54302   0.00023  -0.00030
  lat         0.70643    0.70642   0.00001  -0.00006
  latperm     0.79243    0.79224   0.00019  -0.00014
  strat       0.68823    0.68789   0.00035  -0.00122
  vdc assign check, c=[0.62, 0.692, 0.679, 0.644, 0.788, 0.558, 0.412, 0.725]: formula 0.37208  MC 0.37219  |diff| 0.00011

== N1: m=2 profiles ==
  c = [0.5, 0.5]: iid 0.5000  anti 0.0000  strat 0.0000  lat 0.0000  latperm 0.0000
    cov_lat(d=1/2) = -0.2500
    MC iid      var 0.4998 mean -0.00038
    MC anti     var 0.0000 mean +0.00000
    MC strat    var 0.0000 mean +0.00000
    MC lat      var 0.0000 mean +0.00000
    MC latperm  var 0.0000 mean +0.00000
  c = [0.75, 0.25]: iid 0.3750  anti 0.0000  strat 0.2500  lat 0.5000  latperm 0.5000
    cov_lat(d=1/2) = +0.0625
    MC iid      var 0.3751 mean +0.00011
    MC anti     var 0.0000 mean +0.00000
    MC strat    var 0.2502 mean -0.00009
    MC lat      var 0.4998 mean -0.00078
    MC latperm  var 0.5005 mean -0.00066

== N1: m=8 aligned-arc profile (all U-space arcs share a left endpoint) ==
  c = [0.8125, 0.6875, 0.5625, 0.4375, 0.3125, 0.1875, 0.0625, 0.9375]
  iid      formula    1.34375  MC    1.34377  |diff| 0.00002
  anti     formula    0.50000  MC    0.49965  |diff| 0.00035
  lat      formula    5.50000  MC    5.50050  |diff| 0.00050
  latperm  formula    1.00000  MC    1.00095  |diff| 0.00095
  strat    formula    0.75000  MC    0.75088  |diff| 0.00088

== antithetic pair covariance formula, direct MC ==
  c=(0.780,0.572): formula -0.09406  MC -0.09417
  c=(0.469,0.844): formula -0.08310  MC -0.08314
  c=(0.074,0.080): formula -0.00591  MC -0.00591
  c=(0.847,0.526): formula -0.07264  MC -0.07261

== N1: latperm counterexamples at m = 3, 4, 8 (found by coordinate ascent) ==
  m=3 c=[0.906, 0.232, 0.906]: iid 0.34850  latperm formula 0.41806 (ratio 1.1996)  MC 0.41839  mean -0.000311
  m=4 c=[0.952, 0.952, 0.202, 0.952]: iid 0.29828  latperm formula 0.34264 (ratio 1.1487)  MC 0.34243  mean -0.000162
  m=8 c=[0.018, 0.018, 0.018, 0.018, 0.002, 0.002, 0.002, 0.906]: iid 0.16186  latperm formula 0.17174 (ratio 1.0611)  MC 0.17178  mean -0.000133

== ratio sweep over 20000 random m=8 profiles (closed forms) ==
  m/(m-1) = 8/7 = 1.1429
  max Var_anti/Var_iid = 0.9568  at c = [0.973, 0.911, 0.012, 0.407, 0.998, 0.192, 0.263, 0.027]
  max Var_strat/Var_iid = 0.9012  at c = [0.989, 0.018, 0.831, 0.015, 0.031, 0.006, 0.391, 0.013]
  max Var_lat/Var_iid = 2.9831  at c = [0.054, 0.069, 0.964, 0.89, 0.842, 0.552, 0.462, 0.345]
  max Var_latperm/Var_iid = 0.9657  at c = [0.989, 0.018, 0.831, 0.015, 0.031, 0.006, 0.391, 0.013]
```

### A.3 Addendum — exceedance frequency on random profiles (Remark 6.1)

```python
import numpy as np
# (uses var_iid/var_anti/var_lat/var_latperm/var_strat from A.1)
r = np.random.default_rng(11)
n = 20000
frac_exceed = {k: 0 for k in ("anti", "strat", "lat", "latperm")}
ratios = {k: [] for k in frac_exceed}
for _ in range(n):
    c = r.random(8)
    vi = var_iid(c)
    for k, fn in (("anti", var_anti), ("strat", var_strat),
                  ("lat", var_lat), ("latperm", var_latperm)):
        rat = fn(c) / vi
        ratios[k].append(rat)
        frac_exceed[k] += rat > 1.0
for k in ratios:
    a = np.array(ratios[k])
    print(f"{k:8s} median ratio {np.median(a):.3f}  mean {a.mean():.3f}  "
          f"frac>1: {frac_exceed[k]/n:.4f}")
```

```text
anti     median ratio 0.510  mean 0.513  frac>1: 0.0000
strat    median ratio 0.522  mean 0.521  frac>1: 0.0000
lat      median ratio 0.494  mean 0.578  frac>1: 0.1027
latperm  median ratio 0.577  mean 0.577  frac>1: 0.0000
```

---

## References

- Cranley, R. and Patterson, T. N. L. (1976). Randomization of number
  theoretic methods for multiple integration. *SIAM Journal on Numerical
  Analysis*, 13(6):904-914.
- Croci, M., Fasi, M., Higham, N. J., Mary, T., and Mikaitis, M. (2022).
  Stochastic rounding: implementation, error analysis and applications.
  *Royal Society Open Science*, 9(3):211631.
- Gray, R. M. and Stockham, T. G. (1993). Dithered quantizers. *IEEE
  Transactions on Information Theory*, 39(3):805-812.
- Gupta, S., Agrawal, A., Gopalakrishnan, K., and Narayanan, P. (2015). Deep
  learning with limited numerical precision. *ICML 2015*.
- Hoeffding, W. (1951). A combinatorial central limit theorem. *The Annals of
  Mathematical Statistics*, 22(4):558-566.
- Joag-Dev, K. and Proschan, F. (1983). Negative association of random
  variables, with applications. *The Annals of Statistics*, 11(1):286-295.
- McKay, M. D., Beckman, R. J., and Conover, W. J. (1979). A comparison of
  three methods for selecting values of input variables in the analysis of
  output from a computer code. *Technometrics*, 21(2):239-245.
- Owen, A. B. (1997). Monte Carlo variance of scrambled net quadrature.
  *SIAM Journal on Numerical Analysis*, 34(5):1884-1910.
- Panferov, A., et al. (2025). Correlated quantizers in distributed
  optimization. *UAI 2025*. [TODO: exact title/authors from the related-work
  scan before submission.]
- Schuchman, L. (1964). Dither signals and their effect on quantization
  noise. *IEEE Transactions on Communication Technology*, 12(4):162-165.
- Stein, M. (1987). Large sample properties of simulations using Latin
  hypercube sampling. *Technometrics*, 29(2):143-151.
- Suresh, A. T., Sun, Z., Ro, J., and Yu, F. (2022). Correlated quantization
  for distributed mean estimation and optimization. *ICML 2022*, PMLR 162.
- Zamir, R. and Feder, M. (1992). On universal quantization by randomized
  uniform/lattice quantizers. *IEEE Transactions on Information Theory*,
  38(2):428-436.
