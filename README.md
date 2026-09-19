# 对数域隐马尔可夫推断库

离线传感序列分析用的离散 HMM：对数域 forward/backward、Viterbi 最可能路径、一次 Baum-Welch 参数更新。纯 Python 标准库实现，无需 numpy/scipy。

## 环境与运行

- Windows 原生 Python 3.14.7（任何 Python 3.9+ 亦可），仅标准库，无第三方依赖、无外部服务。

```bash
python demo.py                                  # 约 8 秒内完成的演示
python -m unittest discover -s tests -v         # 全部单元测试
```

## 接口（`hmm.py`）

### 构造模型

```python
from hmm import HMM

model = HMM(start, trans, emit)
# start: 长度 N 的初始状态分布
# trans: N×N 转移矩阵，trans[i][j] = P(q_{t+1}=j | q_t=i)
# emit:  N×M 发射矩阵，emit[i][k] = P(观测=k | 状态=i)
# 约束：1<=N<=12，1<=M<=16；概率有限非负，允许为 0；
# 每行行和与 1 的偏差必须 <= 1e-12，校验通过后按实际行和重新归一化。
```

参数非法时抛 `InvalidModelError`；观测符号越界、类型错误或超长抛 `InvalidSequenceError`。

### 推断

| 方法 | 返回 |
| --- | --- |
| `forward(obs)` | `(log_alpha, loglik)`，`log_alpha[t][i] = log P(q_t=i, o_0..o_t)` |
| `backward(obs)` | `(log_beta, loglik)` |
| `log_likelihood(obs)` | 对数似然（float） |
| `posteriors(obs)` | `(loglik, gamma, xi)`：`gamma[t][i]` 为状态后验；`xi[t][i][j]` 为相邻状态对后验（长度 T-1） |
| `viterbi(obs)` | `(path, log_score)`：最可能状态序列（0 基状态编号）及其对数分数 |

约定：

- 单条推断序列长度 <= 2000；观测值必须是 `0..M-1` 的 `int`。
- 零概率在对数域表示为负无穷；**不可能序列**：`loglik = -inf`，`gamma = xi = None`，Viterbi 返回 `(None, -inf)`，任何路径上都不产生 NaN。
- **空序列**：`loglik = 0.0`，`gamma = xi = []`，Viterbi 路径为 `[]`、分数 `0.0`。
- Viterbi 平局规则：浮点计算得到的路径分数**精确相等（`==`）**时，选择字典序最小的完整状态序列（全程比较，而非只比较末状态）；不使用近似相等改写目标。

### 一次 Baum-Welch 更新

```python
new_model = model.baum_welch(sequences)
```

- 至多 16 条序列、每条至多 64 项，序列相互独立、计数相加；
- 空序列不贡献任何计数；训练集为空或只含空序列时原样返回同一模型对象；
- 状态零占用的 `trans`/`emit` 行保留当前模型的原值，不做隐式平滑（未观测符号的 MLE 为 0）；
- 任一非空序列在当前模型下不可能时抛 `ImpossibleSequenceError`，整次更新拒绝，输入模型不变；
- 返回新的 `HMM`，原模型不被修改。

## 实现要点

- 所有递推在对数域进行，log-sum-exp 用最大值平移 + `math.fsum`，长低概率序列（如每步 1e-3、64 步）不发生下溢。
- BW 的软计数以对数项收集、再统一 log-sum-exp 归一化。
- 随机测试使用固定种子（`random.Random(20260920)` / `777`）、小样本；另有状态<=3、长度<=5 的全路径穷举参考实现交叉验证似然、gamma/xi 与 Viterbi，穷举数值比较使用 1e-9 明确容差。
- 测试覆盖：零概率、不可达状态、精确平局路径、空序列/全空训练、未使用（零占用）行、多序列边界（16×64）、长小概率序列、更新后输入模型不变。
