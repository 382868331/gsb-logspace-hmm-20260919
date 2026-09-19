"""对数域离散隐马尔可夫模型（HMM），仅使用 Python 3 标准库。

提供：
  * log-sum-exp 与对数域 forward / backward；
  * 对数似然、后验 gamma / xi（不可能序列返回 -inf / None，不产生 NaN）；
  * Viterbi 最可能路径，浮点分数精确相等时取字典序最小的完整状态序列；
  * 一次 Baum-Welch 参数更新（多序列独立累加，零占用行保留原值）。

所有输入概率先校验（有限、非负、行和误差 <= 1e-12），再按实际行和归一化。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

__all__ = [
    "NEG_INF",
    "HMM",
    "HMMError",
    "InvalidModelError",
    "InvalidSequenceError",
    "ImpossibleSequenceError",
    "logsumexp",
]

NEG_INF = float("-inf")

MAX_STATES = 12
MAX_SYMBOLS = 16
MAX_SEQ_LEN = 2000
MAX_BW_SEQUENCES = 16
MAX_BW_SEQ_LEN = 64
DEFAULT_TOL = 1e-12

Obs = Sequence[int]


class HMMError(ValueError):
    """本库所有取值错误的基类。"""


class InvalidModelError(HMMError):
    """模型参数非法（维度、取值、行和不满足约定）。"""


class InvalidSequenceError(HMMError):
    """观测序列非法（符号越界、过长或类型不对）。"""


class ImpossibleSequenceError(HMMError):
    """Baum-Welch 训练集中出现当前模型下概率为 0 的序列，整次更新被拒绝。"""


def logsumexp(values):  # noqa: ANN001 - 接受任意可迭代对象
    """数值稳定的 log(sum(exp(values)))，全为 -inf 时返回 -inf。"""
    finite = [v for v in values if v != NEG_INF]
    if not finite:
        return NEG_INF
    m = max(finite)
    # math.fsum 保证行和型归一化在构造新模型时仍满足 1e-12 校验。
    s = math.fsum(math.exp(v - m) for v in finite)
    return m + math.log(s)


def _is_real_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _checked_row(row, length: int, where: str, tol: float) -> Tuple[float, ...]:
    if not isinstance(row, (list, tuple)) or len(row) != length:
        raise InvalidModelError(
            f"{where} 必须是长度为 {length} 的 list/tuple，实际为 {row!r}"
        )
    values: List[float] = []
    total = 0.0
    for x in row:
        if not _is_real_number(x):
            raise InvalidModelError(f"{where} 中的概率必须为有限非负实数：{x!r}")
        fx = float(x)
        if not math.isfinite(fx) or fx < 0.0:
            raise InvalidModelError(f"{where} 中的概率必须有限且非负：{x!r}")
        values.append(fx)
        total += fx
    if not math.isfinite(total) or abs(total - 1.0) > tol:
        raise InvalidModelError(
            f"{where} 的行和为 {total!r}，与 1 的偏差超过 {tol:g}"
        )
    inv = 1.0 / total
    return tuple(v * inv for v in values)


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else NEG_INF


def _safe_exp(x: float) -> float:
    return 0.0 if x == NEG_INF else math.exp(x)


class HMM:
    """离散 HMM。

    Parameters
    ----------
    start: 长度 N 的初始状态分布；
    trans: N x N 转移矩阵，``trans[i][j] = P(q_{t+1}=j | q_t=i)``；
    emit:  N x M 发射矩阵，``emit[i][k] = P(观测=k | 状态=i)``。
    1 <= N <= 12，1 <= M <= 16。概率允许为 0；构造时按实际行和重新归一化。
    """

    def __init__(self, start, trans, emit, *, tol: float = DEFAULT_TOL):
        if not isinstance(start, (list, tuple)) or not (
            1 <= len(start) <= MAX_STATES
        ):
            raise InvalidModelError(
                f"状态数必须在 1..{MAX_STATES} 之间，start={start!r}"
            )
        n = len(start)

        if not isinstance(trans, (list, tuple)) or len(trans) != n:
            raise InvalidModelError(f"trans 必须是 {n}x{n} 矩阵：{trans!r}")
        for i, row in enumerate(trans):
            if not isinstance(row, (list, tuple)) or len(row) != n:
                raise InvalidModelError(f"trans 第 {i} 行长度必须为 {n}")

        if not isinstance(emit, (list, tuple)) or len(emit) != n:
            raise InvalidModelError(f"emit 必须是 {n} 行矩阵：{emit!r}")
        m = len(emit[0]) if emit else 0
        if not (1 <= m <= MAX_SYMBOLS):
            raise InvalidModelError(f"观测符号数必须在 1..{MAX_SYMBOLS} 之间")
        for i, row in enumerate(emit):
            if not isinstance(row, (list, tuple)) or len(row) != m:
                raise InvalidModelError(f"emit 第 {i} 行长度必须为 {m}")

        self._tol = tol
        self._n = n
        self._m = m
        self._start = _checked_row(start, n, "start", tol)
        self._trans = tuple(
            _checked_row(row, n, f"trans[{i}]", tol) for i, row in enumerate(trans)
        )
        self._emit = tuple(
            _checked_row(row, m, f"emit[{i}]", tol) for i, row in enumerate(emit)
        )
        self._lp = tuple(_safe_log(p) for p in self._start)
        self._la = tuple(
            tuple(_safe_log(p) for p in row) for row in self._trans
        )
        self._le = tuple(
            tuple(_safe_log(p) for p in row) for row in self._emit
        )

    @classmethod
    def _from_normalized(cls, start, trans, emit, tol: float) -> "HMM":
        """内部构造：参数已是概率域行向量，直接使用而不再归一化。

        Baum-Welch 重估出的行本身满足归一化（误差为浮点舍入量级），
        零占用行通过此通道逐位保留原模型的取值。
        """
        self = object.__new__(cls)
        self._tol = tol
        self._n = len(start)
        self._m = len(emit[0])
        self._start = tuple(float(x) for x in start)
        self._trans = tuple(tuple(float(x) for x in row) for row in trans)
        self._emit = tuple(tuple(float(x) for x in row) for row in emit)
        self._lp = tuple(_safe_log(p) for p in self._start)
        self._la = tuple(
            tuple(_safe_log(p) for p in row) for row in self._trans
        )
        self._le = tuple(
            tuple(_safe_log(p) for p in row) for row in self._emit
        )
        return self

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self._n

    @property
    def n_symbols(self) -> int:
        return self._m

    @property
    def start(self) -> Tuple[float, ...]:
        return self._start

    @property
    def trans(self) -> Tuple[Tuple[float, ...], ...]:
        return self._trans

    @property
    def emit(self) -> Tuple[Tuple[float, ...], ...]:
        return self._emit

    def _validate_obs(self, obs, max_len: int) -> Tuple[int, ...]:
        if not isinstance(obs, (list, tuple)):
            raise InvalidSequenceError(
                f"观测序列必须是 list/tuple，实际类型 {type(obs).__name__}"
            )
        if len(obs) > max_len:
            raise InvalidSequenceError(
                f"序列长度 {len(obs)} 超过上限 {max_len}"
            )
        checked: List[int] = []
        for x in obs:
            if isinstance(x, bool) or not isinstance(x, int):
                raise InvalidSequenceError(f"观测符号必须为 int 索引：{x!r}")
            if not 0 <= x < self._m:
                raise InvalidSequenceError(
                    f"观测符号 {x} 超出合法范围 0..{self._m - 1}"
                )
            checked.append(x)
        return tuple(checked)

    # ------------------------------------------------------------------
    # forward / backward
    # ------------------------------------------------------------------
    def forward(self, obs) -> Tuple[List[List[float]], float]:
        """对数域前向算法。

        返回 ``(log_alpha, loglik)``；``log_alpha[t][i]`` 为
        ``log P(q_t=i, o_0..o_t)``。空序列返回 ``([], 0.0)``。
        """
        obs = self._validate_obs(obs, MAX_SEQ_LEN)
        n = self._n
        if not obs:
            return [], 0.0
        la, le = self._la, self._le
        alpha: List[List[float]] = [[NEG_INF] * n for _ in obs]
        first = alpha[0]
        o0 = obs[0]
        for i in range(n):
            first[i] = self._lp[i] + le[i][o0]
        for t in range(1, len(obs)):
            ot = obs[t]
            prev = alpha[t - 1]
            cur = alpha[t]
            for j in range(n):
                if le[j][ot] == NEG_INF:
                    continue  # cur[j] 保持 -inf
                cur[j] = (
                    logsumexp(prev[i] + la[i][j] for i in range(n)) + le[j][ot]
                )
        return alpha, logsumexp(alpha[-1])

    def backward(self, obs) -> Tuple[List[List[float]], float]:
        """对数域后向算法。

        返回 ``(log_beta, loglik)``；``log_beta[t][i]`` 为
        ``log P(o_{t+1}..o_{T-1} | q_t=i)``。空序列返回 ``([], 0.0)``。
        """
        obs = self._validate_obs(obs, MAX_SEQ_LEN)
        n = self._n
        if not obs:
            return [], 0.0
        la, le = self._la, self._le
        beta: List[List[float]] = [[0.0] * n for _ in obs]
        for t in range(len(obs) - 2, -1, -1):
            ot1 = obs[t + 1]
            nxt = beta[t + 1]
            cur = beta[t]
            for i in range(n):
                cur[i] = logsumexp(
                    la[i][j] + le[j][ot1] + nxt[j] for j in range(n)
                )
        ll = logsumexp(
            self._lp[i] + le[i][obs[0]] + beta[0][i] for i in range(n)
        )
        return beta, ll

    def log_likelihood(self, obs) -> float:
        """序列的对数似然；空序列为 0.0，不可能序列为 -inf。"""
        return self.forward(obs)[1]

    def posteriors(self, obs):
        """返回 ``(loglik, gamma, xi)``（普通概率域的后验）。

        gamma[t][i] = P(q_t=i | obs)；
        xi[t][i][j] = P(q_t=i, q_{t+1}=j | obs)，``xi`` 长度为 T-1。
        不可能序列返回 ``(-inf, None, None)``；空序列返回 ``(0.0, [], [])``。
        """
        obs = self._validate_obs(obs, MAX_SEQ_LEN)
        if not obs:
            return 0.0, [], []
        alpha, ll = self.forward(obs)
        if ll == NEG_INF:
            return NEG_INF, None, None
        beta, _ = self.backward(obs)
        n = self._n
        gamma = [
            [_safe_exp(alpha[t][i] + beta[t][i] - ll) for i in range(n)]
            for t in range(len(obs))
        ]
        la, le = self._la, self._le
        xi = []
        for t in range(len(obs) - 1):
            ot1 = obs[t + 1]
            mat = [[0.0] * n for _ in range(n)]
            at, bn = alpha[t], beta[t + 1]
            for i in range(n):
                ai = at[i]
                if ai == NEG_INF:
                    continue
                lai = la[i]
                row = mat[i]
                for j in range(n):
                    v = ai + lai[j] + le[j][ot1] + bn[j] - ll
                    if v != NEG_INF:
                        row[j] = math.exp(v)
            xi.append(mat)
        return ll, gamma, xi

    # ------------------------------------------------------------------
    # Viterbi
    # ------------------------------------------------------------------
    def viterbi(self, obs) -> Tuple[Optional[List[int]], float]:
        """最可能状态序列及其对数分数。

        分数在浮点计算下**精确相等**（``==``）时，返回字典序最小的完整
        状态序列（通过逐层维护前缀字典序秩实现，而不是只比较末状态）。
        空序列返回 ``([], 0.0)``；不可能序列返回 ``(None, -inf)``。
        """
        obs = self._validate_obs(obs, MAX_SEQ_LEN)
        n = self._n
        if not obs:
            return [], 0.0
        le, la = self._le, self._la

        delta = [self._lp[i] + le[i][obs[0]] for i in range(n)]
        psi: List[List[int]] = [[-1] * n for _ in obs]
        # rank[i]：当前层到达状态 i 的最优前缀，在全部 N 个前缀中的字典序
        # 秩（0 最小）。第 0 层前缀就是 [i]，字典序与状态编号一致。
        rank = list(range(n))

        for t in range(1, len(obs)):
            ot = obs[t]
            new_delta = [NEG_INF] * n
            new_psi = [-1] * n
            for j in range(n):
                if le[j][ot] == NEG_INF:
                    continue
                best = NEG_INF
                best_i = -1
                best_rank = -1
                for i in range(n):
                    if delta[i] == NEG_INF or la[i][j] == NEG_INF:
                        continue
                    cand = delta[i] + la[i][j]
                    if cand > best or (
                        cand == best
                        and best_i >= 0
                        and rank[i] < best_rank
                    ):
                        best = cand
                        best_i = i
                        best_rank = rank[i]
                if best_i >= 0:
                    new_delta[j] = best + le[j][ot]
                    new_psi[j] = best_i
            # 新前缀 R_j = Q_{psi[j]} + [j]：
            # psi 不同则字典序由旧前缀秩决定，psi 相同则由末状态 j 决定。
            # 不可达单元不参与真实比较，赋予排在最后的形式键即可。
            keys = [
                (rank[new_psi[j]], j) if new_psi[j] >= 0 else (n, j)
                for j in range(n)
            ]
            order = sorted(range(n), key=keys.__getitem__)
            new_rank = [0] * n
            for r, j in enumerate(order):
                new_rank[j] = r
            delta = new_delta
            psi[t] = new_psi
            rank = new_rank

        terminal = -1
        terminal_rank = n + 1
        best_score = NEG_INF
        for j in range(n):
            if delta[j] == NEG_INF:
                continue
            if delta[j] > best_score or (
                delta[j] == best_score and rank[j] < terminal_rank
            ):
                best_score = delta[j]
                terminal = j
                terminal_rank = rank[j]
        if terminal < 0:
            return None, NEG_INF

        path = [0] * len(obs)
        cur = terminal
        for t in range(len(obs) - 1, -1, -1):
            path[t] = cur
            if t > 0:
                cur = psi[t][cur]
        return path, best_score

    # ------------------------------------------------------------------
    # 一次 Baum-Welch
    # ------------------------------------------------------------------
    def baum_welch(self, sequences) -> "HMM":
        """对相互独立的多条序列做一次 Baum-Welch 更新，返回新 HMM。

        * 至多 16 条序列、每条至多 64 项；
        * 空序列不贡献任何计数；训练集全空（或只含空序列）时原样返回自身；
        * 状态 i 零占用的 trans/emit 行保留该模型当前行，不做隐式平滑；
        * 任一非空序列在当前模型下不可能（loglik=-inf）时抛出
          ``ImpossibleSequenceError``，本模型保持不变。
        """
        if not isinstance(sequences, (list, tuple)):
            raise InvalidSequenceError(
                "sequences 必须是 list/tuple，每条为一个观测序列"
            )
        if len(sequences) > MAX_BW_SEQUENCES:
            raise InvalidSequenceError(
                f"训练序列数 {len(sequences)} 超过上限 {MAX_BW_SEQUENCES}"
            )
        obs_list = [
            self._validate_obs(seq, MAX_BW_SEQ_LEN) for seq in sequences
        ]
        active = [o for o in obs_list if o]
        if not active:
            return self

        n, m = self._n, self._m
        la, le = self._la, self._le

        # 以“对数项列表 + 最后一次 logsumexp”在对数域累加所有计数，
        # 避免长低概率序列在线性域下溢。
        pi_terms: List[List[float]] = [[] for _ in range(n)]
        a_num: List[List[List[float]]] = [
            [[] for _ in range(n)] for _ in range(n)
        ]
        a_den: List[List[float]] = [[] for _ in range(n)]
        b_num: List[List[List[float]]] = [
            [[] for _ in range(m)] for _ in range(n)
        ]
        b_den: List[List[float]] = [[] for _ in range(n)]

        for obs in active:
            alpha, ll = self.forward(obs)
            if ll == NEG_INF:
                raise ImpossibleSequenceError(
                    "训练集中存在当前模型下概率为 0 的序列，已拒绝整次更新"
                )
            beta, _ = self.backward(obs)
            T = len(obs)

            lg = [
                [alpha[t][i] + beta[t][i] - ll for i in range(n)]
                for t in range(T)
            ]
            for i in range(n):
                g0 = lg[0][i]
                if g0 != NEG_INF:
                    pi_terms[i].append(g0)
                for t in range(T):
                    g = lg[t][i]
                    if g == NEG_INF:
                        continue
                    b_den[i].append(g)
                    b_num[i][obs[t]].append(g)
                    if t < T - 1:
                        a_den[i].append(g)

            for t in range(T - 1):
                ot1 = obs[t + 1]
                at = alpha[t]
                bn = beta[t + 1]
                for i in range(n):
                    if at[i] == NEG_INF:
                        continue
                    row = a_num[i]
                    lai = la[i]
                    for j in range(n):
                        v = at[i] + lai[j] + le[j][ot1] + bn[j] - ll
                        if v != NEG_INF:
                            row[j].append(v)

        pi_log = [logsumexp(terms) for terms in pi_terms]
        pi_norm = logsumexp(pi_log)
        new_start = [_safe_exp(pi_log[i] - pi_norm) for i in range(n)]

        new_trans: List[Union[Tuple[float, ...], List[float]]] = []
        for i in range(n):
            d = logsumexp(a_den[i])
            if d == NEG_INF:
                new_trans.append(self._trans[i])  # 零占用行保留原值
            else:
                new_trans.append(
                    [_safe_exp(logsumexp(a_num[i][j]) - d) for j in range(n)]
                )

        new_emit: List[Union[Tuple[float, ...], List[float]]] = []
        for i in range(n):
            d = logsumexp(b_den[i])
            if d == NEG_INF:
                new_emit.append(self._emit[i])
            else:
                new_emit.append(
                    [_safe_exp(logsumexp(b_num[i][k]) - d) for k in range(m)]
                )

        return HMM._from_normalized(new_start, new_trans, new_emit, self._tol)
