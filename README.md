# 对数域隐马尔可夫推断库

离线传感序列分析的离散 HMM 库：对数域概率推断、最可能路径解码和一次 Baum-Welch 参数更新。完整规格见 `TASK.md`。

## 环境与命令

Windows 原生 Python 3.14.7，仅标准库，无第三方依赖、无外部服务、无 Docker。

- 演示：`python demo.py`（约 8 秒内，展示一次正常推断和一次实际触发的不可能序列失败）
- 测试：`python -m unittest discover -s tests -v`

## 接口（`hmm.py`）

```python
from hmm import HMM

model = HMM(pi, A, B)   # pi: 初始分布(N,), A: 转移矩阵(N,N), B: 发射矩阵(N,M)
```

约束：1..12 状态、1..16 观测符号。概率必须有限非负，每个分布行和与 1 相差 ≤1e-12；校验通过后按行归一化。观测值为整数符号索引 `0..M-1`，单序列长度 ≤2000。模型构造后不可变，访问器 `pi`/`A`/`B` 返回新列表。

### 概率推断：`model.forward_backward(obs)`

对数域 forward/backward + log-sum-exp，返回 `ForwardBackwardResult(loglik, gamma, xi)`：

- `loglik`：观测序列的对数似然；
- `gamma`：T×N 后验状态概率（每行和为 1）；
- `xi`：(T-1)×N×N 后验转移概率；
- 零概率映射为 -inf；不可能序列返回 `loglik=-inf` 且 `gamma/xi=None`，不产生 NaN；空序列返回 `loglik=0.0`、`gamma=[]`、`xi=[]`。

### 最可能路径：`model.viterbi(obs)`

返回 `ViterbiResult(path, log_score)`。浮点分数精确相等时选字典序最小的完整状态序列（不是只按末状态打破平局，也不用近似相等）。空序列返回 `([], 0.0)`；不可能序列返回 `(None, -inf)`。

### 一次参数更新：`model.baum_welch_step(sequences)`

对最多 16 条、每条 ≤64 项的相互独立训练序列做一次 Baum-Welch（EM）更新，返回新 `HMM`，不修改原模型：

- 空序列不贡献计数；全空训练集原样返回；
- 零占用的参数行（不可达状态等）保留原值，不做隐式平滑；
- 任一序列在模型下概率为零则整体拒绝（抛 `ValueError`），输入模型不变；
- 单次更新不会使训练集总对数似然下降（容差 `1e-9*(1+|old|)`）。

## 验证方式

`tests/test_hmm.py` 对状态 ≤3、长度 ≤5 的随机小例（固定种子）穷举全部路径，对照验证对数似然、gamma、xi 和 Viterbi（数值比较使用显式容差；平局用精确相等的浮点分数构造）；另覆盖零概率、不可达状态、精确平局路径、空序列/全空训练、未使用行、多序列边界（16 条/64 项）、长低概率序列（T=2000）稳定性以及输入模型不变性。
