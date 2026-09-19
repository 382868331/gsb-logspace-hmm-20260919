"""Demo: normal inference plus an actually triggered failure.

Run with:  python demo.py
Everything below is computed by the library at runtime (hmm.py).
"""

import math
import time

from hmm import HMM, NEG_INF


def main():
    started = time.perf_counter()

    print("=== 1. Normal inference (3 states, 3 symbols) ===")
    model = HMM(
        pi=[0.5, 0.3, 0.2],
        A=[[0.7, 0.2, 0.1],
           [0.3, 0.5, 0.2],
           [0.2, 0.3, 0.5]],
        B=[[0.6, 0.3, 0.1],
           [0.1, 0.6, 0.3],
           [0.3, 0.3, 0.4]],
    )
    obs = [0, 1, 2, 1, 0, 2, 2, 1]
    fb = model.forward_backward(obs)
    print(f"observations      : {obs}")
    print(f"log-likelihood    : {fb.loglik:.6f}")
    print("gamma (posterior state probs, first 3 steps):")
    for t in range(3):
        print(f"  t={t}: " + "  ".join(f"{g:.4f}" for g in fb.gamma[t]))
    print("xi[0] (posterior transition probs):")
    for row in fb.xi[0]:
        print("  " + "  ".join(f"{v:.4f}" for v in row))
    vit = model.viterbi(obs)
    print(f"viterbi path      : {vit.path}")
    print(f"viterbi log score : {vit.log_score:.6f}")

    print()
    print("=== 2. One Baum-Welch update (likelihood must not decrease) ===")
    train = [[0, 1, 2, 1, 0], [2, 2, 1, 0], [0, 0, 1, 2, 2, 1], []]
    nonempty = [s for s in train if s]
    old_ll = sum(model.forward_backward(s).loglik for s in nonempty)
    new_model = model.baum_welch_step(train)
    new_ll = sum(new_model.forward_backward(s).loglik for s in nonempty)
    print(f"total train loglik before: {old_ll:.6f}")
    print(f"total train loglik after : {new_ll:.6f}")
    print(f"monotone (up to 1e-9 tol): "
          f"{new_ll >= old_ll - 1e-9 * (1.0 + abs(old_ll))}")

    print()
    print("=== 3. Long low-probability sequence stays stable ===")
    long_obs = [i % 3 for i in range(2000)]
    fb_long = model.forward_backward(long_obs)
    print(f"T=2000 loglik: {fb_long.loglik:.2f} "
          f"(finite: {math.isfinite(fb_long.loglik)}, "
          f"gamma row sum: {sum(fb_long.gamma[1999]):.12f})")

    print()
    print("=== 4. Actually triggered failure: impossible sequence ===")
    # State 1 is the only state that can emit symbol 1, but it is
    # unreachable (pi[1] = 0 and no transitions lead into it).
    rigid = HMM(
        pi=[1.0, 0.0],
        A=[[1.0, 0.0], [0.5, 0.5]],
        B=[[1.0, 0.0], [0.5, 0.5]],
    )
    bad = rigid.forward_backward([0, 1])
    print(f"forward_backward([0, 1]): loglik = {bad.loglik} "
          f"(-inf expected: {bad.loglik == NEG_INF}), "
          f"gamma = {bad.gamma}, xi = {bad.xi}")
    bad_v = rigid.viterbi([0, 1])
    print(f"viterbi([0, 1])          : path = {bad_v.path}, "
          f"log_score = {bad_v.log_score}")
    try:
        rigid.baum_welch_step([[0, 0], [1]])
        print("baum_welch_step: unexpectedly accepted")
    except ValueError as exc:
        print(f"baum_welch_step rejected the update: {exc}")
    print(f"model unchanged after rejection: "
          f"{(rigid.pi, rigid.A, rigid.B) == ([1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]])}")

    print()
    print(f"demo finished in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
