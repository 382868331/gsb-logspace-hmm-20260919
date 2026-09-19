"""hmm 库的单元测试。

覆盖 TASK.md 要求的全部情形：零概率、不可达状态、精确平局路径、
空序列/全空训练、未使用行、多序列边界、长小概率序列、输入模型不变，
以及状态<=3、长度<=5 小例的全路径穷举交叉验证。随机用例固定种子。
"""

import itertools
import math
import random
import unittest

from hmm import (
    NEG_INF,
    HMM,
    ImpossibleSequenceError,
    InvalidModelError,
    InvalidSequenceError,
    logsumexp,
)

TOL = 1e-9


# ---------------------------------------------------------------------------
# 穷举参考实现
# ---------------------------------------------------------------------------
def brute_force(model: HMM, obs):
    """枚举全部状态路径，返回 (loglik, gamma, xi, best_path, best_score)。"""
    n = model.n_states
    T = len(obs)
    if T == 0:
        return 0.0, [], [], [], 0.0
    lp = [math.log(p) if p > 0 else NEG_INF for p in model.start]
    la = [
        [math.log(p) if p > 0 else NEG_INF for p in row]
        for row in model.trans
    ]
    le = [
        [math.log(p) if p > 0 else NEG_INF for p in row]
        for row in model.emit
    ]

    weights = []
    paths = list(itertools.product(range(n), repeat=T))
    for path in paths:
        w = lp[path[0]] + le[path[0]][obs[0]]
        for t in range(1, T):
            w += la[path[t - 1]][path[t]] + le[path[t]][obs[t]]
        weights.append(w)
    ll = logsumexp(weights)
    if ll == NEG_INF:
        return NEG_INF, None, None, None, NEG_INF

    gamma = [[0.0] * n for _ in range(T)]
    xi = [[[0.0] * n for _ in range(n)] for _ in range(T - 1)]
    best_score = NEG_INF
    best_path = None
    for path, w in zip(paths, weights):
        p = math.exp(w - ll)
        for t, s in enumerate(path):
            gamma[t][s] += p
            if t < T - 1:
                xi[t][path[t]][path[t + 1]] += p
        if w > best_score or (w == best_score and (best_path is None or path < best_path)):
            best_score = w
            best_path = path
    return ll, gamma, xi, list(best_path), best_score


def random_distribution(rng: random.Random, k: int, allow_zero: bool = True):
    raw = [0.0 if allow_zero and rng.random() < 0.25 else rng.random() + 0.05
           for _ in range(k)]
    if not any(raw):
        raw[0] = 1.0
    s = sum(raw)
    return [x / s for x in raw]


class ExhaustiveReferenceTests(unittest.TestCase):
    """状态<=3、长度<=5：对随机小例枚举全部路径逐一比对。"""

    def test_exhaustive_small_cases(self):
        rng = random.Random(20260920)
        cases = 0
        for _ in range(60):
            n = rng.randint(1, 3)
            m = rng.randint(1, 3)
            T = rng.randint(0, 5)
            start = random_distribution(rng, n)
            trans = [random_distribution(rng, n) for _ in range(n)]
            emit = [random_distribution(rng, m) for _ in range(n)]
            model = HMM(start, trans, emit)
            obs = [rng.randrange(m) for _ in range(T)]

            bll, bg, bx, bpath, bscore = brute_force(model, obs)
            cases += 1

            alpha, fll = model.forward(obs)
            beta, bkwll = model.backward(obs)
            ll, gamma, xi = model.posteriors(obs)
            vpath, vscore = model.viterbi(obs)

            self.assertFalse(any(math.isnan(v) for row in alpha for v in row))
            if T == 0:
                self.assertEqual((fll, bkwll, ll, bll), (0.0, 0.0, 0.0, 0.0))
                self.assertEqual(gamma, [])
                self.assertEqual(xi, [])
                self.assertEqual(vpath, [])
                self.assertEqual(vscore, 0.0)
                continue

            self.assertAlmostEqual(fll, bll, delta=1e-9)
            self.assertAlmostEqual(bkwll, bll, delta=1e-9)
            self.assertAlmostEqual(ll, bll, delta=1e-9)

            if bll == NEG_INF:
                self.assertIsNone(gamma)
                self.assertIsNone(xi)
                self.assertIsNone(vpath)
                self.assertEqual(vscore, NEG_INF)
                continue

            for t in range(T):
                self.assertAlmostEqual(sum(gamma[t]), 1.0, delta=1e-9)
                for i in range(n):
                    self.assertAlmostEqual(gamma[t][i], bg[t][i], delta=1e-9)
            for t in range(T - 1):
                total = sum(xi[t][i][j] for i in range(n) for j in range(n))
                self.assertAlmostEqual(total, 1.0, delta=1e-9)
                for i in range(n):
                    for j in range(n):
                        self.assertAlmostEqual(
                            xi[t][i][j], bx[t][i][j], delta=1e-9
                        )
                # gamma-xi 一致性
                for i in range(n):
                    self.assertAlmostEqual(
                        sum(xi[t][i][j] for j in range(n)),
                        gamma[t][i],
                        delta=1e-9,
                    )
                    self.assertAlmostEqual(
                        sum(xi[t][k][i] for k in range(n)),
                        gamma[t + 1][i],
                        delta=1e-9,
                    )

            # 数值比较使用明确容差；路径本身的穷举权重也必须在容差内最优。
            self.assertAlmostEqual(vscore, bscore, delta=1e-9)
            w = lp_path(model, obs, vpath)
            self.assertAlmostEqual(w, bscore, delta=1e-9)
            # 穷举中不存在严格更优的路径
            for path in itertools.product(range(n), repeat=T):
                ww = lp_path(model, obs, path)
                self.assertLessEqual(ww, w + 1e-9)

            # BW 单序列交叉验证：用穷举后验在线性域直接数计数。
            new = model.baum_welch([obs])
            for i in range(n):
                self.assertAlmostEqual(new.start[i], bg[0][i], delta=1e-9)
                g_occ = sum(bg[t][i] for t in range(T))
                if g_occ <= 1e-12:
                    self.assertEqual(new.emit[i], model.emit[i])
                else:
                    for k in range(m):
                        num = sum(bg[t][i] for t in range(T) if obs[t] == k)
                        self.assertAlmostEqual(
                            new.emit[i][k], num / g_occ, delta=1e-9
                        )
                g_tr = sum(bg[t][i] for t in range(T - 1))
                if g_tr <= 1e-12:
                    self.assertEqual(new.trans[i], model.trans[i])
                else:
                    for j in range(n):
                        num = sum(bx[t][i][j] for t in range(T - 1))
                        self.assertAlmostEqual(
                            new.trans[i][j], num / g_tr, delta=1e-9
                        )
        self.assertGreaterEqual(cases, 50)


def lp_path(model: HMM, obs, path):
    lp = [math.log(p) if p > 0 else NEG_INF for p in model.start]
    w = lp[path[0]] + math.log(model.emit[path[0]][obs[0]]) \
        if model.emit[path[0]][obs[0]] > 0 else NEG_INF
    if w == NEG_INF:
        return NEG_INF
    for t in range(1, len(obs)):
        a = model.trans[path[t - 1]][path[t]]
        b = model.emit[path[t]][obs[t]]
        if a <= 0 or b <= 0:
            return NEG_INF
        w += math.log(a) + math.log(b)
    return w


# ---------------------------------------------------------------------------
# 精确平局
# ---------------------------------------------------------------------------
class ExactTieTests(unittest.TestCase):
    def test_tie_full_sequence_not_just_last_state(self):
        # 路径 [0,1] 与 [1,0] 分数精确相等；字典序最小的是 [0,1]，
        # 但它结束在编号更大的末状态——只按末状态打破平局会错误选 [1,0]。
        model = HMM(
            [0.5, 0.5],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        path, score = model.viterbi([0, 0])
        self.assertEqual(path, [0, 1])
        self.assertEqual(score, math.log(0.5))
        # 另一条路径分数必须精确相等，证明平局是真实触发的
        self.assertEqual(lp_path(model, [0, 0], [1, 0]), score)

    def test_tie_self_loops(self):
        # 两条自环路径 [0,0] 与 [1,1] 精确平局。
        model = HMM(
            [0.5, 0.5],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        for T in (1, 2, 4):
            path, score = model.viterbi([0] * T)
            self.assertEqual(path, [0] * T)
            self.assertEqual(score, math.log(0.5))

    def test_tie_differs_in_middle(self):
        # 长度 3 时 [0,0,0] 与 [0,1,0] 等四条路径精确平局，
        # 字典序最小完整序列为 [0,0,0]。
        model = HMM(
            [1.0, 0.0, 0.0],
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        )
        path, score = model.viterbi([0, 0, 0])
        self.assertEqual(path, [0, 0, 0])
        self.assertEqual(score, math.log(0.25))
        self.assertEqual(lp_path(model, [0, 0, 0], [0, 1, 0]), score)


# ---------------------------------------------------------------------------
# 零概率 / 不可达 / 不可能序列 / 空序列
# ---------------------------------------------------------------------------
class ZeroProbabilityTests(unittest.TestCase):
    def setUp(self):
        # 状态 2 不可达（start=0），且观测符号 1 在可达状态下发射概率为 0。
        self.model = HMM(
            [0.7, 0.3, 0.0],
            [[0.6, 0.4, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 1.0]],
            [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]],
        )

    def test_impossible_symbol(self):
        model = HMM(
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        obs = [0, 1]  # 状态 0 发不出符号 1，状态 1 又不可达
        self.assertEqual(model.log_likelihood(obs), NEG_INF)
        ll, gamma, xi = model.posteriors(obs)
        self.assertEqual(ll, NEG_INF)
        self.assertIsNone(gamma)
        self.assertIsNone(xi)
        path, score = model.viterbi(obs)
        self.assertIsNone(path)
        self.assertEqual(score, NEG_INF)
        # 不产生 NaN
        alpha, _ = model.forward(obs)
        self.assertFalse(
            any(math.isnan(v) for row in alpha for v in row)
        )

    def test_unreachable_state_gamma_zero(self):
        ll, gamma, xi = self.model.posteriors([0, 1, 0])
        self.assertTrue(math.isfinite(ll))
        for row in gamma:
            self.assertEqual(row[2], 0.0)
        for mat in xi:
            self.assertTrue(all(v == 0.0 for v in mat[2]))
            self.assertTrue(all(mat[i][2] == 0.0 for i in range(3)))

    def test_empty_sequence(self):
        self.assertEqual(self.model.log_likelihood([]), 0.0)
        ll, gamma, xi = self.model.posteriors([])
        self.assertEqual(ll, 0.0)
        self.assertEqual(gamma, [])
        self.assertEqual(xi, [])
        path, score = self.model.viterbi([])
        self.assertEqual(path, [])
        self.assertEqual(score, 0.0)
        alpha, ll2 = self.model.forward([])
        self.assertEqual(alpha, [])
        self.assertEqual(ll2, 0.0)

    def test_long_sequence_becomes_impossible_via_transitions(self):
        # 合法行和下，t0 只能在状态 0 且下一步必须切到状态 1，
        # 而状态 1 发不出符号 0：长度 1 可行，长度 2 不可能。
        model = HMM(
            [1.0, 0.0],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(model.log_likelihood([0]), 0.0)
        self.assertEqual(model.log_likelihood([0, 0]), NEG_INF)
        self.assertEqual(model.viterbi([0, 0]), (None, NEG_INF))


# ---------------------------------------------------------------------------
# Baum-Welch
# ---------------------------------------------------------------------------
class BaumWelchTests(unittest.TestCase):
    def _snapshot(self, model):
        return (
            tuple(model.start),
            tuple(tuple(r) for r in model.trans),
            tuple(tuple(r) for r in model.emit),
        )

    def test_all_empty_training_returns_unchanged(self):
        model = HMM(
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.8, 0.2], [0.3, 0.7]],
        )
        before = self._snapshot(model)
        result = model.baum_welch([])
        self.assertIs(result, model)
        result2 = model.baum_welch([[], []])
        self.assertIs(result2, model)
        self.assertEqual(self._snapshot(model), before)

    def test_impossible_sequence_rejected_model_unchanged(self):
        model = HMM(
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        before = self._snapshot(model)
        with self.assertRaises(ImpossibleSequenceError):
            model.baum_welch([[0, 1]])
        self.assertEqual(self._snapshot(model), before)
        # 混合可拒绝序列时同样整体拒绝
        with self.assertRaises(ImpossibleSequenceError):
            model.baum_welch([[0, 0], [0, 1], []])
        self.assertEqual(self._snapshot(model), before)

    def test_zero_occupation_rows_preserved(self):
        # 状态 2 不可达且从不被占用：其 trans/emit 行必须原样保留。
        preserved_a = (0.1, 0.2, 0.7)
        preserved_b = (0.35, 0.25, 0.4)
        model = HMM(
            [0.6, 0.4, 0.0],
            [[0.7, 0.3, 0.0], [0.4, 0.6, 0.0], preserved_a],
            [[0.8, 0.15, 0.05], [0.2, 0.3, 0.5], preserved_b],
        )
        seqs = [[0, 1, 2, 0], [2, 0, 1], [0, 0, 2, 1, 0]]
        new = model.baum_welch(seqs)
        self.assertEqual(new.trans[2], preserved_a)
        self.assertEqual(new.emit[2], preserved_b)
        for row in new.trans:
            self.assertAlmostEqual(sum(row), 1.0, delta=1e-12)
        for row in new.emit:
            self.assertAlmostEqual(sum(row), 1.0, delta=1e-12)

    def test_no_implicit_smoothing_unseen_symbol_becomes_zero(self):
        # 未见过的符号在 MLE 下应变为 0，而不是被平滑成正值。
        model = HMM(
            [1.0],
            [[1.0]],
            [[0.5, 0.5]],
        )
        new = model.baum_welch([[0, 0, 0]])
        self.assertEqual(new.emit[0], (1.0, 0.0))
        self.assertEqual(new.start, (1.0,))

    def test_empty_sequences_contribute_nothing(self):
        model = HMM(
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.8, 0.2], [0.3, 0.7]],
        )
        a = model.baum_welch([[0, 1, 0], [1, 0]])
        b = model.baum_welch([[], [0, 1, 0], [], [1, 0], []])
        for i in range(2):
            self.assertAlmostEqual(a.start[i], b.start[i], delta=1e-12)
            for j in range(2):
                self.assertAlmostEqual(a.trans[i][j], b.trans[i][j], delta=1e-12)
                self.assertAlmostEqual(a.emit[i][j], b.emit[i][j], delta=1e-12)

    def test_multi_sequence_brute_force_counts(self):
        # 多条非空短序列：用穷举后验在线性域独立累加计数，
        # 与 baum_welch 的对数域聚合逐一比对。
        rng = random.Random(31337)
        n, m = 3, 2
        model = HMM(
            random_distribution(rng, n, allow_zero=False),
            [random_distribution(rng, n, allow_zero=False) for _ in range(n)],
            [random_distribution(rng, m, allow_zero=False) for _ in range(n)],
        )
        sequences = [[rng.randrange(m) for _ in range(rng.randint(1, 4))]
                     for _ in range(4)]
        new = model.baum_welch(sequences)

        pi = [0.0] * n
        a_num = [[0.0] * n for _ in range(n)]
        a_den = [0.0] * n
        b_num = [[0.0] * m for _ in range(n)]
        b_den = [0.0] * n
        for obs in sequences:
            _, bg, bx, _, _ = brute_force(model, obs)
            T = len(obs)
            for i in range(n):
                pi[i] += bg[0][i]
                for t in range(T):
                    b_den[i] += bg[t][i]
                    b_num[i][obs[t]] += bg[t][i]
                    if t < T - 1:
                        a_den[i] += bg[t][i]
                for t in range(T - 1):
                    for j in range(n):
                        a_num[i][j] += bx[t][i][j]
        z = sum(pi)
        for i in range(n):
            self.assertAlmostEqual(new.start[i], pi[i] / z, delta=1e-9)
            if a_den[i] <= 1e-12:
                self.assertEqual(new.trans[i], model.trans[i])
            else:
                for j in range(n):
                    self.assertAlmostEqual(
                        new.trans[i][j], a_num[i][j] / a_den[i], delta=1e-9
                    )
            if b_den[i] <= 1e-12:
                self.assertEqual(new.emit[i], model.emit[i])
            else:
                for k in range(m):
                    self.assertAlmostEqual(
                        new.emit[i][k], b_num[i][k] / b_den[i], delta=1e-9
                    )

    def test_multi_sequence_boundaries(self):
        model = HMM(
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.8, 0.2], [0.3, 0.7]],
        )
        sixteen = [[0, 1] * 32 for _ in range(16)]  # 16 条 x 64 项
        new = model.baum_welch(sixteen)
        self.assertEqual(new.n_states, 2)
        with self.assertRaises(InvalidSequenceError):
            model.baum_welch(sixteen + [[0]])  # 17 条
        with self.assertRaises(InvalidSequenceError):
            model.baum_welch([[0] * 65])  # 65 项
        # 推断允许到 2000
        self.assertTrue(math.isfinite(model.log_likelihood([0] * 2000)))
        with self.assertRaises(InvalidSequenceError):
            model.log_likelihood([0] * 2001)

    def test_em_monotonic_random(self):
        # 固定种子小样本：一次 EM 后总对数似然不得下降超过容差。
        rng = random.Random(777)
        for _ in range(8):
            n = rng.randint(1, 4)
            m = rng.randint(1, 4)
            model = HMM(
                random_distribution(rng, n, allow_zero=False),
                [random_distribution(rng, n, allow_zero=False) for _ in range(n)],
                [random_distribution(rng, m, allow_zero=False) for _ in range(n)],
            )
            k = rng.randint(1, 5)
            seqs = [[rng.randrange(m) for _ in range(rng.randint(1, 20))]
                    for _ in range(k)]
            old = sum(model.log_likelihood(s) for s in seqs)
            new = model.baum_welch(seqs)
            new_ll = sum(new.log_likelihood(s) for s in seqs)
            self.assertGreaterEqual(
                new_ll, old - TOL * (1 + abs(old))
            )

    def test_long_low_probability_sequence_stable(self):
        # 每步概率约 1e-3 的 64 长序列：线性域必然下溢，对数域必须稳定。
        model = HMM(
            [1.0],
            [[1.0]],
            [[1e-3, 1 - 1e-3]],
        )
        seq = [0] * 64
        ll, gamma, xi = model.posteriors(seq)
        self.assertAlmostEqual(ll, 64 * math.log(1e-3), delta=1e-9)
        self.assertTrue(all(math.isfinite(g[0]) for g in gamma))
        for g in gamma:
            self.assertAlmostEqual(g[0], 1.0, delta=1e-12)
        new = model.baum_welch([seq])
        self.assertEqual(new.emit[0][0], 1.0)
        self.assertEqual(new.emit[0][1], 0.0)

    def test_original_model_unchanged_after_update(self):
        model = HMM(
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.8, 0.2], [0.3, 0.7]],
        )
        before = self._snapshot(model)
        model.baum_welch([[0, 1, 0, 1], [1, 1, 0]])
        self.assertEqual(self._snapshot(model), before)

    def test_iterated_bw_converges_to_identifiable_structure(self):
        # 块状数据对应“只发 0 / 只发 1”的两个状态。注意完全相同的
        # 初始行会造成数学上的对称驻点，因此用非对称初参打破对称。
        model = HMM(
            [0.6, 0.4],
            [[0.7, 0.3], [0.3, 0.7]],
            [[0.6, 0.4], [0.4, 0.6]],
        )
        seqs = [[0, 0, 1, 1], [0, 1, 1], [0, 0, 0, 1, 1]]
        prev = sum(model.log_likelihood(s) for s in seqs)
        last = prev
        for _ in range(100):
            model = model.baum_welch(seqs)
            last = sum(model.log_likelihood(s) for s in seqs)
            self.assertGreaterEqual(last, prev - TOL * (1 + abs(prev)))
            prev = last
        self.assertTrue(math.isfinite(last))
        emit = model.emit
        hi0 = max(emit[0][0], emit[1][0])
        hi1 = max(emit[0][1], emit[1][1])
        self.assertGreater(hi0, 0.99)
        self.assertGreater(hi1, 0.99)
        self.assertAlmostEqual(sum(model.start), 1.0, delta=1e-12)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
class ValidationTests(unittest.TestCase):
    def test_row_sum_tolerance(self):
        HMM([0.5, 0.5 + 5e-13], [[1.0, 0.0], [0.0, 1.0]],
            [[1.0], [1.0]])  # 偏差 <= 1e-12，接受并归一化
        with self.assertRaises(InvalidModelError):
            HMM([0.5, 0.5 + 5e-12], [[1.0, 0.0], [0.0, 1.0]],
                [[1.0], [1.0]])

    def test_bad_probabilities(self):
        good_a = [[1.0, 0.0], [0.0, 1.0]]
        good_b = [[1.0], [1.0]]
        with self.assertRaises(InvalidModelError):
            HMM([0.5, -0.5], good_a, good_b)
        with self.assertRaises(InvalidModelError):
            HMM([0.5, float("inf")], good_a, good_b)
        with self.assertRaises(InvalidModelError):
            HMM([0.5, float("nan")], good_a, good_b)
        with self.assertRaises(InvalidModelError):
            HMM([0.5, 0.5], [[1.0, 0.0], [0.5, 0.7]], good_b)

    def test_bad_sizes(self):
        with self.assertRaises(InvalidModelError):
            HMM([], [], [])
        with self.assertRaises(InvalidModelError):
            HMM([1.0 / 13] * 13,
                [[1.0 / 13] * 13] * 13,
                [[1.0] for _ in range(13)])
        with self.assertRaises(InvalidModelError):
            HMM([1.0], [[1.0]], [[1.0 / 17] * 17])

    def test_bad_observations(self):
        model = HMM([1.0], [[1.0]], [[0.5, 0.5]])
        with self.assertRaises(InvalidSequenceError):
            model.log_likelihood([2])
        with self.assertRaises(InvalidSequenceError):
            model.log_likelihood([-1])
        with self.assertRaises(InvalidSequenceError):
            model.log_likelihood([0.5])
        with self.assertRaises(InvalidSequenceError):
            model.log_likelihood([True])
        with self.assertRaises(InvalidSequenceError):
            model.log_likelihood("0")

    def test_normalization_applied(self):
        # 行和与 1 的偏差在 1e-12 以内：接受并按实际行和归一化。
        eps = 1e-13
        model = HMM(
            [0.5, 0.5 + eps],
            [[0.25, 0.75 + eps], [0.75, 0.25 + eps]],
            [[0.5, 0.5 + eps], [0.25, 0.75 + eps]],
        )
        s = 1.0 + eps
        self.assertAlmostEqual(sum(model.start), 1.0, delta=1e-15)
        self.assertAlmostEqual(model.start[0], 0.5 / s, delta=1e-15)
        self.assertAlmostEqual(model.trans[0][1], (0.75 + eps) / s, delta=1e-15)
        self.assertAlmostEqual(model.emit[1][0], 0.25 / s, delta=1e-15)
        # 偏差超范围的行（行和 2）不享受归一化，直接拒绝。
        with self.assertRaises(InvalidModelError):
            HMM([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], [[1.0], [1.0]])

    def test_logsumexp(self):
        self.assertEqual(logsumexp([NEG_INF, NEG_INF]), NEG_INF)
        self.assertAlmostEqual(logsumexp([0.0, 0.0]), math.log(2))
        self.assertAlmostEqual(
            logsumexp([-1000.0, -1001.0]), -1000 + math.log1p(math.exp(-1))
        )


if __name__ == "__main__":
    unittest.main()
