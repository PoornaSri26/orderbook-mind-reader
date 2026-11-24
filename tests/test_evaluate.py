"""Checks on the metrics and on the cost line, since that is what decides the verdict."""

from __future__ import annotations

import numpy as np

from micro import evaluate


def test_log_loss_of_a_perfect_predictor_is_near_zero():
    y = np.array([1, 1, 0, 0])
    # ``score`` clips probabilities to [1e-6, 1-1e-6] so a confident predictor
    # cannot produce an infinite loss; the floor is therefore -ln(1-1e-6) ~ 1e-6.
    p = np.array([1 - 1e-9, 1 - 1e-9, 1e-9, 1e-9])
    assert evaluate.score("perfect", y, p).log_loss < 1e-5


def test_log_loss_of_a_coin_flip_is_ln2():
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.5)
    assert abs(evaluate.score("coin", y, p).log_loss - np.log(2)) < 1e-9


def test_auc_of_noise_is_about_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    p = rng.random(20_000)
    assert 0.47 < evaluate.score("noise", y, p).auc < 0.53


def test_block_bootstrap_detects_a_clear_improvement():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 20_000)
    good = np.where(y == 1, 0.75, 0.25)
    bad = np.full(20_000, 0.5)
    d = evaluate.block_bootstrap_diff(y, bad, good, block=500, n_boot=50)
    assert d["observed_diff"] < 0 and d["excludes_zero"]


def test_edge_bps_is_zero_when_the_model_is_indifferent():
    y = np.array([1, 0, 1, 0])
    p = np.full(4, 0.5)
    moves = np.array([1.0, -1.0, 1.0, -1.0])
    assert evaluate.edge_in_bps(y, p, moves)["n_trades"] == 0


def test_edge_bps_rewards_a_correct_side():
    y = np.array([1, 1])
    p = np.array([0.9, 0.9])
    moves = np.array([2.0, 2.0])
    out = evaluate.edge_in_bps(y, p, moves)
    assert out["edge_bps"] == 2.0 and out["hit_rate"] == 1.0


def test_verdict_calls_out_predictable_but_not_tradeable():
    assert "not tradeable" in evaluate.verdict(0.05, 0.3)
