"""Log-space discrete hidden Markov model library.

Provides, for a discrete HMM with 1..12 hidden states and 1..16 observation
symbols (single sequence up to 2000 emissions):

- forward/backward in the log domain with log-sum-exp: log-likelihood,
  posterior state probabilities (gamma) and posterior transition
  probabilities (xi);
- Viterbi decoding with exact-float tie-breaking towards the
  lexicographically smallest full state sequence;
- a single Baum-Welch (EM) parameter update over up to 16 independent
  training sequences of at most 64 emissions each.

Pure Python 3.14 standard library; no third-party dependencies.
All probabilities are validated (finite, non-negative, row sums within
1e-12 of 1) and rows are normalized once at construction.  Zero
probabilities map to -inf in the log domain; impossible sequences yield
loglik=-inf with gamma/xi=None and never produce NaN.
"""

from __future__ import annotations

import math
from collections import namedtuple

__all__ = [
    "HMM",
    "ForwardBackwardResult",
    "ViterbiResult",
    "MAX_STATES",
    "MAX_SYMBOLS",
    "MAX_SEQUENCE_LENGTH",
    "MAX_TRAIN_SEQUENCES",
    "MAX_TRAIN_LENGTH",
    "NEG_INF",
]

NEG_INF = float("-inf")

MAX_STATES = 12
MAX_SYMBOLS = 16
MAX_SEQUENCE_LENGTH = 2000
MAX_TRAIN_SEQUENCES = 16
MAX_TRAIN_LENGTH = 64

_ROW_SUM_TOL = 1e-12

#: loglik: float; gamma: T x N list of rows (None if impossible);
#: xi: (T-1) x N x N list of matrices (None if impossible).
ForwardBackwardResult = namedtuple("ForwardBackwardResult", ("loglik", "gamma", "xi"))
#: path: list of state indices (None if impossible); log_score: float.
ViterbiResult = namedtuple("ViterbiResult", ("path", "log_score"))


def _log(x):
    return math.log(x) if x > 0.0 else NEG_INF


def _logsumexp(values):
    m = max(values)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(math.fsum(math.exp(v - m) for v in values))


def _probability_row(row, name):
    try:
        values = list(row)
    except TypeError:
        raise ValueError(f"{name}: expected a sequence of probabilities") from None
    if not values:
        raise ValueError(f"{name}: distribution must not be empty")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name}: probabilities must be real numbers, got {v!r}")
        v = float(v)
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(
                f"{name}: probabilities must be finite and non-negative, got {v!r}"
            )
        out.append(v)
    total = math.fsum(out)
    if abs(total - 1.0) > _ROW_SUM_TOL:
        raise ValueError(
            f"{name}: row sums must be within 1e-12 of 1.0 (got {total!r})"
        )
    return [v / total for v in out]


def _rectangular(rows, n_rows, n_cols, name):
    try:
        rows = [list(r) for r in rows]
    except TypeError:
        raise ValueError(f"{name}: expected a matrix (sequence of rows)") from None
    if len(rows) != n_rows:
        raise ValueError(f"{name}: expected {n_rows} rows, got {len(rows)}")
    width = None
    for i, r in enumerate(rows):
        if not r:
            raise ValueError(f"{name}[{i}]: row must not be empty")
        if width is None:
            width = len(r)
        elif len(r) != width:
            raise ValueError(f"{name}: rows must all have the same length")
        if n_cols is not None and len(r) != n_cols:
            raise ValueError(f"{name}[{i}]: expected {n_cols} columns, got {len(r)}")
    return rows


class HMM:
    """Discrete hidden Markov model with log-space inference.

    ``pi`` is the initial state distribution (length N), ``A`` the N x N
    transition matrix and ``B`` the N x M emission matrix.  Parameters are
    validated and row-normalized at construction; the instance is
    effectively immutable afterwards (matrices are stored as tuples and
    accessors return fresh lists).
    """

    def __init__(self, pi, A, B):
        pi_row = _probability_row(pi, "pi")
        n = len(pi_row)
        if n > MAX_STATES:
            raise ValueError(f"pi: 1..{MAX_STATES} states allowed, got {n}")
        a_rows = _rectangular(A, n, n, "A")
        b_rows = _rectangular(B, n, None, "B")
        m = len(b_rows[0])
        if m > MAX_SYMBOLS:
            raise ValueError(f"B: 1..{MAX_SYMBOLS} observation symbols allowed, got {m}")
        a_rows = [_probability_row(r, f"A[{i}]") for i, r in enumerate(a_rows)]
        b_rows = [_probability_row(r, f"B[{i}]") for i, r in enumerate(b_rows)]
        self._pi = tuple(pi_row)
        self._A = tuple(tuple(r) for r in a_rows)
        self._B = tuple(tuple(r) for r in b_rows)
        self._logpi = tuple(_log(p) for p in self._pi)
        self._logA = tuple(tuple(_log(p) for p in r) for r in self._A)
        self._logB = tuple(tuple(_log(p) for p in r) for r in self._B)

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    @property
    def n_states(self):
        return len(self._pi)

    @property
    def n_symbols(self):
        return len(self._B[0])

    @property
    def pi(self):
        return list(self._pi)

    @property
    def A(self):
        return [list(r) for r in self._A]

    @property
    def B(self):
        return [list(r) for r in self._B]

    # ------------------------------------------------------------------
    # input checking
    # ------------------------------------------------------------------
    def _check_observations(self, obs, max_length):
        if isinstance(obs, (str, bytes)):
            raise ValueError(
                "observations must be a sequence of integer symbol indices"
            )
        try:
            seq = list(obs)
        except TypeError:
            raise ValueError(
                "observations must be a sequence of integer symbol indices"
            ) from None
        if len(seq) > max_length:
            raise ValueError(
                f"sequence length {len(seq)} exceeds limit {max_length}"
            )
        m = self.n_symbols
        for o in seq:
            if isinstance(o, bool) or not isinstance(o, int):
                raise ValueError(f"observation {o!r} is not an integer symbol index")
            if not 0 <= o < m:
                raise ValueError(f"observation {o} out of range 0..{m - 1}")
        return seq

    # ------------------------------------------------------------------
    # forward-backward
    # ------------------------------------------------------------------
    def forward_backward(self, obs):
        """Log-domain forward-backward.

        Returns ForwardBackwardResult(loglik, gamma, xi).  An empty
        sequence yields loglik 0.0 with empty gamma/xi.  A sequence with
        zero probability under the model yields loglik -inf and
        gamma/xi None (never NaN).
        """
        obs = self._check_observations(obs, MAX_SEQUENCE_LENGTH)
        T = len(obs)
        if T == 0:
            return ForwardBackwardResult(0.0, [], [])
        n = self.n_states
        logpi, logA, logB = self._logpi, self._logA, self._logB

        alpha = [[0.0] * n for _ in range(T)]
        o0 = obs[0]
        for j in range(n):
            alpha[0][j] = logpi[j] + logB[j][o0]
        for t in range(1, T):
            o = obs[t]
            prev = alpha[t - 1]
            cur = alpha[t]
            for j in range(n):
                cur[j] = _logsumexp([prev[i] + logA[i][j] for i in range(n)]) + logB[j][o]

        loglik = _logsumexp(alpha[T - 1])
        if loglik == NEG_INF:
            return ForwardBackwardResult(NEG_INF, None, None)

        beta = [[0.0] * n for _ in range(T)]
        for t in range(T - 2, -1, -1):
            o_next = obs[t + 1]
            nxt = beta[t + 1]
            cur = beta[t]
            for i in range(n):
                cur[i] = _logsumexp(
                    [logA[i][j] + logB[j][o_next] + nxt[j] for j in range(n)]
                )

        gamma = []
        for t in range(T):
            row = [math.exp(alpha[t][j] + beta[t][j] - loglik) for j in range(n)]
            s = math.fsum(row)
            if s > 0.0:
                row = [g / s for g in row]
            gamma.append(row)

        xi = []
        for t in range(T - 1):
            o_next = obs[t + 1]
            nxt = beta[t + 1]
            mat = []
            for i in range(n):
                a = alpha[t][i]
                mat.append(
                    [
                        math.exp(a + logA[i][j] + logB[j][o_next] + nxt[j] - loglik)
                        for j in range(n)
                    ]
                )
            xi.append(mat)

        return ForwardBackwardResult(loglik, gamma, xi)

    # ------------------------------------------------------------------
    # Viterbi
    # ------------------------------------------------------------------
    def viterbi(self, obs):
        """Most likely state path and its log score.

        Ties between exactly equal floating-point scores are resolved in
        favour of the lexicographically smallest *full* state sequence
        (not merely by final state).  An empty sequence yields ([], 0.0);
        an impossible sequence yields (None, -inf).
        """
        obs = self._check_observations(obs, MAX_SEQUENCE_LENGTH)
        T = len(obs)
        if T == 0:
            return ViterbiResult([], 0.0)
        n = self.n_states
        logpi, logA, logB = self._logpi, self._logA, self._logB

        o0 = obs[0]
        delta = [logpi[j] + logB[j][o0] for j in range(n)]
        # lex_rank[j] ranks the best path ending in state j at the current
        # time among all such paths, in lexicographic order.  At t=0 the
        # paths are the singletons (j,), so the rank is the state itself.
        lex_rank = list(range(n))
        preds = []
        for t in range(1, T):
            o = obs[t]
            new_delta = [0.0] * n
            new_pred = [0] * n
            for j in range(n):
                best = NEG_INF
                best_i = -1
                for i in range(n):
                    cand = delta[i] + logA[i][j]
                    if (
                        best_i < 0
                        or cand > best
                        or (cand == best and lex_rank[i] < lex_rank[best_i])
                    ):
                        best = cand
                        best_i = i
                new_pred[j] = best_i
                new_delta[j] = best + logB[j][o]
            # The best path to (t, j) is bestpath(t-1, pred) ++ [j]; its
            # lexicographic rank is determined by (rank of prefix, j).
            order = sorted(range(n), key=lambda j: (lex_rank[new_pred[j]], j))
            new_rank = [0] * n
            for r, j in enumerate(order):
                new_rank[j] = r
            preds.append(new_pred)
            delta = new_delta
            lex_rank = new_rank

        best_score = max(delta)
        if best_score == NEG_INF:
            return ViterbiResult(None, NEG_INF)
        best_j = min(
            (j for j in range(n) if delta[j] == best_score),
            key=lambda j: lex_rank[j],
        )
        path = [0] * T
        path[T - 1] = best_j
        for t in range(T - 1, 0, -1):
            best_j = preds[t - 1][best_j]
            path[t - 1] = best_j
        return ViterbiResult(path, best_score)

    # ------------------------------------------------------------------
    # one Baum-Welch (EM) update
    # ------------------------------------------------------------------
    def baum_welch_step(self, sequences):
        """One Baum-Welch update over independent training sequences.

        At most 16 sequences of at most 64 emissions each.  Empty
        sequences contribute no counts; if every sequence is empty the
        model is returned unchanged.  Parameter rows with zero occupancy
        keep their previous values (no implicit smoothing).  If any
        sequence is impossible under the model the whole update is
        rejected with ValueError and this model is left unchanged.

        Returns a new HMM; ``self`` is never modified.
        """
        if isinstance(sequences, (str, bytes)):
            raise ValueError("training data must be a sequence of sequences")
        try:
            seqs = list(sequences)
        except TypeError:
            raise ValueError("training data must be a sequence of sequences") from None
        if len(seqs) > MAX_TRAIN_SEQUENCES:
            raise ValueError(
                f"at most {MAX_TRAIN_SEQUENCES} training sequences, got {len(seqs)}"
            )
        seqs = [self._check_observations(s, MAX_TRAIN_LENGTH) for s in seqs]
        nonempty = [s for s in seqs if s]
        if not nonempty:
            return HMM(self.pi, self.A, self.B)

        results = [self.forward_backward(s) for s in nonempty]
        for r in results:
            if r.loglik == NEG_INF:
                raise ValueError(
                    "update rejected: a training sequence has zero "
                    "probability under the model"
                )

        n = self.n_states
        m = self.n_symbols
        k = len(nonempty)
        pi_new = [0.0] * n
        a_num = [[0.0] * n for _ in range(n)]
        a_den = [0.0] * n
        b_num = [[0.0] * m for _ in range(n)]
        b_den = [0.0] * n
        for seq, r in zip(nonempty, results):
            gamma, xi = r.gamma, r.xi
            T = len(seq)
            for j in range(n):
                pi_new[j] += gamma[0][j]
            for t in range(T - 1):
                for i in range(n):
                    a_den[i] += gamma[t][i]
                    xrow = xi[t][i]
                    for j in range(n):
                        a_num[i][j] += xrow[j]
            for t in range(T):
                o = seq[t]
                for j in range(n):
                    g = gamma[t][j]
                    b_den[j] += g
                    b_num[j][o] += g

        pi_new = [p / k for p in pi_new]
        a_new = []
        for i in range(n):
            if a_den[i] > 0.0:
                a_new.append([v / a_den[i] for v in a_num[i]])
            else:
                a_new.append(list(self._A[i]))
        b_new = []
        for j in range(n):
            if b_den[j] > 0.0:
                b_new.append([v / b_den[j] for v in b_num[j]])
            else:
                b_new.append(list(self._B[j]))
        return HMM(pi_new, a_new, b_new)
