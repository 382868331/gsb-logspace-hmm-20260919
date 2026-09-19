"""Log-space HMM library demo: one normal result plus a triggered failure.

Run: python demo.py (finishes well within ~8 seconds, standard library only).
All output is computed by hmm.py.
"""

import math
import time

from hmm import HMM, ImpossibleSequenceError, NEG_INF


def main() -> None:
    t0 = time.perf_counter()

    # Toy HMM: 2 hidden states (Sunny/Rainy), 3 observation symbols
    # (Dry/Damp/Storm).
    model = HMM(
        start=[0.6, 0.4],
        trans=[[0.7, 0.3], [0.4, 0.6]],
        emit=[[0.8, 0.15, 0.05], [0.1, 0.4, 0.5]],
    )
    obs = [0, 2, 1, 0]
    state_names = ["Sunny", "Rainy"]
    symbol_names = ["Dry", "Damp", "Storm"]

    print("=" * 64)
    print("Normal result: log-space inference and Viterbi")
    print("=" * 64)
    print("Observations:", " -> ".join(symbol_names[o] for o in obs))

    ll, gamma, xi = model.posteriors(obs)
    print(f"log-likelihood log P(obs) = {ll:.10f}   P = {math.exp(ll):.6e}")
    print("Posterior P(state=Rainy | observations) at each time step:")
    print("  " + "  ".join(f"t{t}:{g[1]:.4f}" for t, g in enumerate(gamma)))

    path, score = model.viterbi(obs)
    print(
        "Viterbi best path: ",
        " -> ".join(state_names[s] for s in path),
        f" (path log-score {score:.10f})",
    )

    print()
    print("One Baum-Welch update (3 independent sequences):")
    seqs = [[0, 0, 2, 1], [1, 2, 0], [0, 1, 2, 0, 1]]
    old_ll = sum(model.log_likelihood(s) for s in seqs)
    new_model = model.baum_welch(seqs)
    new_ll = sum(new_model.log_likelihood(s) for s in seqs)
    print(f"  total log-likelihood before: {old_ll:.10f}")
    print(f"  total log-likelihood after:  {new_ll:.10f} (EM, no decrease)")
    print(f"  updated emit[Rainy] = {tuple(round(v, 4) for v in new_model.emit[1])}")

    # ------------------------------------------------------------------
    # Actually triggered failure (1): impossible observation sequence
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("Triggered failure: impossible sequence and rejected training")
    print("=" * 64)

    rigid = HMM(
        start=[1.0, 0.0],
        trans=[[1.0, 0.0], [0.0, 1.0]],
        emit=[[1.0, 0.0], [0.0, 1.0]],
    )
    bad_obs = [0, 1]  # only reachable state 0 cannot emit symbol 1
    bad_ll = rigid.log_likelihood(bad_obs)
    print(f"sequence {bad_obs} under the rigid model: loglik = {bad_ll} (-inf)")
    bad_ll2, bad_gamma, bad_xi = rigid.posteriors(bad_obs)
    print(f"posteriors -> gamma={bad_gamma}, xi={bad_xi}; no NaN produced")
    bad_path, bad_score = rigid.viterbi(bad_obs)
    print(f"viterbi    -> path={bad_path}, score={bad_score}")

    # Actually triggered failure (2): BW rejects the whole batch; input
    # model is untouched.
    before = (rigid.start, rigid.trans, rigid.emit)
    try:
        rigid.baum_welch([[0, 0], bad_obs])
    except ImpossibleSequenceError:
        print(
            "baum_welch raised ImpossibleSequenceError: the training set "
            "contains a zero-probability sequence; the whole update is rejected"
        )
    unchanged = (rigid.start, rigid.trans, rigid.emit) == before
    print(f"input model parameters unchanged after the error: {unchanged}")

    print()
    print(f"Demo finished in {time.perf_counter() - t0:.2f} s.")


if __name__ == "__main__":
    main()
