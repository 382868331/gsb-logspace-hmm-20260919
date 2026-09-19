"""Tests for the log-space HMM library.

Small cases (states <= 3, length <= 5) are verified against exhaustive
enumeration of all paths.  Numeric comparisons use an explicit tolerance;
Viterbi tie-breaking is checked with exactly equal floating-point scores.
"""

import itertools
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmm import (
    HMM,
    NEG_INF,
    MAX_SEQUENCE_LENGTH,
    MAX_TRAIN_LENGTH,
    MAX_TRAIN_SEQUENCES,
)

TOL = 1e-9


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * (1.0 + abs(b))


def brute_force(model, obs):
    """Exhaustive reference: enumerate every state path.

    Returns (loglik, gamma, xi, best_path, best_score).  Path log scores
    are accumulated left-to-right in the same association order as the
    dynamic program, so exact float equality matches the library's
    tie-breaking semantics.
    """
    n = model.n_states
    pi, A, B = model.pi, model.A, model.B
    T = len(obs)
    total = 0.0
    gamma = [[0.0] * n for _ in range(T)]
    xi = [[[0.0] * n for _ in range(n)] for _ in range(max(0, T - 1))]
    best_score = None
    best_path = None
    for path in itertools.product(range(n), repeat=T):
        prob = pi[path[0]] * B[path[0]][obs[0]]
        for t in range(1, T):
            prob *= A[path[t - 1]][path[t]] * B[path[t]][obs[t]]
        total += prob
        for t in range(T):
            gamma[t][path[t]] += prob
        for t in range(T - 1):
            xi[t][path[t]][path[t + 1]] += prob
        if pi[path[0]] == 0.0 or B[path[0]][obs[0]] == 0.0:
            score = NEG_INF
        else:
            score = math.log(pi[path[0]]) + math.log(B[path[0]][obs[0]])
        for t in range(1, T):
            a = A[path[t - 1]][path[t]]
            b = B[path[t]][obs[t]]
            if score == NEG_INF or a == 0.0 or b == 0.0:
                score = NEG_INF
            else:
                score = score + math.log(a) + math.log(b)
        if (
            best_score is None
            or score > best_score
            or (score == best_score and path < best_path)
        ):
            best_score = score
            best_path = path
    loglik = math.log(total) if total > 0.0 else NEG_INF
    if total > 0.0:
        gamma = [[g / total for g in row] for row in gamma]
        xi = [[[v / total for v in row] for row in mat] for mat in xi]
    return loglik, gamma, xi, best_path, best_score


def random_row(rng, length, zero_prob=0.15):
    while True:
        row = [0.0 if rng.random() < zero_prob else rng.random() + 0.05
               for _ in range(length)]
        s = sum(row)
        if s > 0.0:
            return [x / s for x in row]


def random_model(rng, n, m, zero_prob=0.15):
    pi = random_row(rng, n, zero_prob)
    A = [random_row(rng, n, zero_prob) for _ in range(n)]
    B = [random_row(rng, m, zero_prob) for _ in range(n)]
    return HMM(pi, A, B)


def total_loglik(model, seqs):
    return math.fsum(model.forward_backward(s).loglik for s in seqs if s)


class TestValidation(unittest.TestCase):
    def test_row_sum_tolerance_and_normalized(self):
        # within 1e-12 of 1: accepted and renormalized
        model = HMM([0.5 + 5e-13, 0.5], [[0.5, 0.5], [0.5, 0.5]],
                    [[1.0], [1.0]])
        for row in [model.pi] + model.A + model.B:
            self.assertTrue(close(math.fsum(row), 1.0, 1e-12))

    def test_bad_row_sum_rejected(self):
        with self.assertRaises(ValueError):
            HMM([0.5, 0.500001], [[0.5, 0.5], [0.5, 0.5]], [[1.0], [1.0]])
        with self.assertRaises(ValueError):
            HMM([0.5, 0.5], [[0.9, 0.9], [0.5, 0.5]], [[1.0], [1.0]])

    def test_negative_nan_inf_rejected(self):
        base = dict(A=[[0.5, 0.5], [0.5, 0.5]], B=[[1.0], [1.0]])
        with self.assertRaises(ValueError):
            HMM([1.5, -0.5], **base)
        with self.assertRaises(ValueError):
            HMM([float("nan"), 1.0], **base)
        with self.assertRaises(ValueError):
            HMM([float("inf"), 0.0], **base)
        with self.assertRaises(ValueError):
            HMM([0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]],
                [[float("nan")], [1.0]])

    def test_shape_and_size_limits(self):
        with self.assertRaises(ValueError):  # 13 states
            HMM([1.0 / 13] * 13, [[1.0 / 13] * 13] * 13, [[1.0]] * 13)
        with self.assertRaises(ValueError):  # 17 symbols
            HMM([1.0], [[1.0]], [[1.0 / 17] * 17])
        with self.assertRaises(ValueError):  # A not N x N
            HMM([0.5, 0.5], [[0.5, 0.5]], [[1.0], [1.0]])
        with self.assertRaises(ValueError):  # ragged B
            HMM([0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [1.0]])
        # 12 states / 16 symbols accepted
        HMM([1.0 / 12] * 12, [[1.0 / 12] * 12] * 12,
            [[1.0 / 16] * 16] * 12)

    def test_observation_checking(self):
        model = HMM([1.0], [[1.0]], [[0.5, 0.5]])
        with self.assertRaises(ValueError):
            model.forward_backward([0, 2])  # symbol out of range
        with self.assertRaises(ValueError):
            model.forward_backward([0, -1])
        with self.assertRaises(ValueError):
            model.forward_backward([0.5])  # non-integer
        with self.assertRaises(ValueError):
            model.viterbi([True])  # bool is not a valid symbol index
        with self.assertRaises(ValueError):
            model.forward_backward([0] * (MAX_SEQUENCE_LENGTH + 1))
        # exactly at the limit is fine
        r = model.forward_backward([0] * MAX_SEQUENCE_LENGTH)
        self.assertTrue(math.isfinite(r.loglik))


class TestForwardBackward(unittest.TestCase):
    def test_exhaustive_random_small(self):
        rng = __import__("random").Random(20260920)
        for _ in range(40):
            n = rng.randint(1, 3)
            m = rng.randint(1, 3)
            T = rng.randint(1, 5)
            model = random_model(rng, n, m)
            obs = [rng.randrange(m) for _ in range(T)]
            ref_ll, ref_gamma, ref_xi, _, _ = brute_force(model, obs)
            r = model.forward_backward(obs)
            if ref_ll == NEG_INF:
                self.assertEqual(r.loglik, NEG_INF)
                self.assertIsNone(r.gamma)
                self.assertIsNone(r.xi)
                continue
            self.assertTrue(close(r.loglik, ref_ll), (r.loglik, ref_ll))
            for t in range(T):
                for j in range(n):
                    self.assertTrue(close(r.gamma[t][j], ref_gamma[t][j]))
            for t in range(T - 1):
                for i in range(n):
                    for j in range(n):
                        self.assertTrue(close(r.xi[t][i][j], ref_xi[t][i][j]))

    def test_empty_sequence(self):
        model = HMM([0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], [[1.0], [1.0]])
        r = model.forward_backward([])
        self.assertEqual(r.loglik, 0.0)
        self.assertEqual(r.gamma, [])
        self.assertEqual(r.xi, [])

    def test_impossible_sequence(self):
        # state 1 is the only emitter of symbol 1 but is unreachable
        model = HMM([1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]],
                    [[1.0, 0.0], [0.5, 0.5]])
        r = model.forward_backward([0, 1])
        self.assertEqual(r.loglik, NEG_INF)
        self.assertIsNone(r.gamma)
        self.assertIsNone(r.xi)

    def test_zero_probability_and_unreachable_state(self):
        # state 1 unreachable (pi=0, no incoming edges); B has zeros
        model = HMM([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
                    [[0.5, 0.5], [0.25, 0.75]])
        obs = [0, 1, 0, 1]
        ref_ll, ref_gamma, ref_xi, _, _ = brute_force(model, obs)
        r = model.forward_backward(obs)
        self.assertTrue(close(r.loglik, ref_ll))
        for t in range(len(obs)):
            self.assertEqual(r.gamma[t][1], 0.0)  # unreachable state
            self.assertTrue(close(r.gamma[t][0], 1.0))
            self.assertTrue(close(math.fsum(r.gamma[t]), 1.0))

    def test_gamma_xi_consistency(self):
        rng = __import__("random").Random(7)
        model = random_model(rng, 3, 3)
        obs = [rng.randrange(3) for _ in range(6)]
        r = model.forward_backward(obs)
        for t in range(len(obs) - 1):
            for i in range(3):
                self.assertTrue(
                    close(math.fsum(r.xi[t][i]), r.gamma[t][i], 1e-9)
                )

    def test_long_low_probability_sequence(self):
        model = HMM(
            [0.5, 0.3, 0.2],
            [[0.999, 0.0005, 0.0005],
             [0.0005, 0.999, 0.0005],
             [0.0005, 0.0005, 0.999]],
            [[0.7, 0.1, 0.1, 0.1],
             [0.1, 0.7, 0.1, 0.1],
             [0.1, 0.1, 0.7, 0.1]],
        )
        obs = [i % 4 for i in range(MAX_SEQUENCE_LENGTH)]
        r = model.forward_backward(obs)
        self.assertTrue(math.isfinite(r.loglik))
        self.assertLess(r.loglik, 0.0)
        for row in r.gamma:
            self.assertTrue(close(math.fsum(row), 1.0))
            for g in row:
                self.assertFalse(math.isnan(g))
        v = model.viterbi(obs)
        self.assertIsNotNone(v.path)
        self.assertEqual(len(v.path), MAX_SEQUENCE_LENGTH)
        self.assertTrue(math.isfinite(v.log_score))


class TestViterbi(unittest.TestCase):
    def test_exhaustive_random_small(self):
        rng = __import__("random").Random(112233)
        for _ in range(40):
            n = rng.randint(1, 3)
            m = rng.randint(1, 3)
            T = rng.randint(1, 5)
            model = random_model(rng, n, m)
            obs = [rng.randrange(m) for _ in range(T)]
            _, _, _, ref_path, ref_score = brute_force(model, obs)
            v = model.viterbi(obs)
            if ref_score == NEG_INF:
                self.assertIsNone(v.path)
                self.assertEqual(v.log_score, NEG_INF)
            else:
                self.assertEqual(tuple(v.path), ref_path)
                self.assertEqual(v.log_score, ref_score)

    def test_exact_tie_lexicographic_full_sequence(self):
        # Paths (0, 1) and (1, 0) have bit-identical DP scores (same
        # float operations); the lexicographically smallest full
        # sequence (0, 1) must win -- a final-state-only tie-break
        # would wrongly pick (1, 0).
        model = HMM([0.5, 0.5],
                    [[0.25, 0.75], [0.75, 0.25]],
                    [[0.5, 0.5], [0.5, 0.5]])
        v = model.viterbi([0, 0])
        self.assertEqual(v.path, [0, 1])
        _, _, _, ref_path, ref_score = brute_force(model, [0, 0])
        self.assertEqual(tuple(v.path), ref_path)
        self.assertEqual(v.log_score, ref_score)

    def test_exact_tie_longer(self):
        # All paths tie -> lexicographically smallest full sequence.
        model = HMM([0.5, 0.5],
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[0.5, 0.5], [0.5, 0.5]])
        v = model.viterbi([1, 0, 1])
        self.assertEqual(v.path, [0, 0, 0])

    def test_empty_sequence(self):
        model = HMM([1.0], [[1.0]], [[1.0]])
        v = model.viterbi([])
        self.assertEqual(v.path, [])
        self.assertEqual(v.log_score, 0.0)

    def test_impossible_sequence(self):
        model = HMM([1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]],
                    [[1.0, 0.0], [0.5, 0.5]])
        v = model.viterbi([1])
        self.assertIsNone(v.path)
        self.assertEqual(v.log_score, NEG_INF)


class TestBaumWelch(unittest.TestCase):
    def test_single_step_does_not_decrease_likelihood(self):
        rng = __import__("random").Random(424242)
        for _ in range(25):
            n = rng.randint(1, 3)
            m = rng.randint(1, 3)
            # all-positive model: every random sequence is possible
            model = random_model(rng, n, m, zero_prob=0.0)
            k = rng.randint(1, 4)
            seqs = [[rng.randrange(m) for _ in range(rng.randint(1, 8))]
                    for _ in range(k)]
            old_ll = total_loglik(model, seqs)
            new_model = model.baum_welch_step(seqs)
            new_ll = total_loglik(new_model, seqs)
            self.assertGreaterEqual(new_ll, old_ll - 1e-9 * (1.0 + abs(old_ll)))

    def test_all_empty_training_set_returned_unchanged(self):
        model = HMM([0.5, 0.5], [[0.6, 0.4], [0.3, 0.7]],
                    [[0.9, 0.1], [0.2, 0.8]])
        for seqs in ([], [[], [], []]):
            new_model = model.baum_welch_step(seqs)
            self.assertEqual(new_model.pi, model.pi)
            self.assertEqual(new_model.A, model.A)
            self.assertEqual(new_model.B, model.B)

    def test_empty_sequences_contribute_nothing(self):
        model = HMM([0.5, 0.5], [[0.6, 0.4], [0.3, 0.7]],
                    [[0.9, 0.1], [0.2, 0.8]])
        seqs = [[0, 1, 0], [1, 1]]
        with_empty = model.baum_welch_step([[], seqs[0], [], seqs[1], []])
        without = model.baum_welch_step(seqs)
        self.assertEqual(with_empty.pi, without.pi)
        self.assertEqual(with_empty.A, without.A)
        self.assertEqual(with_empty.B, without.B)

    def test_zero_occupancy_rows_kept(self):
        # state 1 unreachable -> its A and B rows must be preserved
        model = HMM([1.0, 0.0], [[1.0, 0.0], [0.1, 0.9]],
                    [[0.5, 0.5], [0.25, 0.75]])
        new_model = model.baum_welch_step([[0, 1, 0], [1, 0]])
        self.assertEqual(new_model.A[1], model.A[1])
        self.assertEqual(new_model.B[1], model.B[1])
        self.assertEqual(new_model.pi[1], 0.0)

    def test_impossible_sequence_rejects_update_and_model_unchanged(self):
        model = HMM([1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]],
                    [[1.0, 0.0], [0.5, 0.5]])
        before = (model.pi, model.A, model.B)
        with self.assertRaises(ValueError):
            model.baum_welch_step([[0, 0], [1]])
        self.assertEqual((model.pi, model.A, model.B), before)

    def test_training_set_boundaries(self):
        model = HMM([1.0], [[1.0]], [[0.5, 0.5]])
        model.baum_welch_step([[0]] * MAX_TRAIN_SEQUENCES)  # 16 ok
        with self.assertRaises(ValueError):
            model.baum_welch_step([[0]] * (MAX_TRAIN_SEQUENCES + 1))
        model.baum_welch_step([[0] * MAX_TRAIN_LENGTH])  # 64 ok
        with self.assertRaises(ValueError):
            model.baum_welch_step([[0] * (MAX_TRAIN_LENGTH + 1)])

    def test_input_model_not_modified(self):
        model = HMM([0.5, 0.5], [[0.6, 0.4], [0.3, 0.7]],
                    [[0.9, 0.1], [0.2, 0.8]])
        before = (model.pi, model.A, model.B)
        new_model = model.baum_welch_step([[0, 1, 0], [1, 1, 0]])
        self.assertEqual((model.pi, model.A, model.B), before)
        self.assertIsNot(new_model, model)

    def test_updated_rows_are_distributions(self):
        rng = __import__("random").Random(99)
        model = random_model(rng, 3, 3)
        seqs = [[rng.randrange(3) for _ in range(5)] for _ in range(3)]
        new_model = model.baum_welch_step(seqs)
        for row in [new_model.pi] + new_model.A + new_model.B:
            self.assertTrue(close(math.fsum(row), 1.0, 1e-12))


if __name__ == "__main__":
    unittest.main()
