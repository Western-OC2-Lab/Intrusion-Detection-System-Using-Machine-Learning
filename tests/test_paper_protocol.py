"""Testes de conformidade do preset ``paper`` com Yang et al. (2022)."""

from __future__ import annotations

import argparse

from mth_ids_pipeline.config import (
    DEFAULT_KMEANS_FRAC,
    PAPER_CV_FOLDS,
    PAPER_HPO_ON_VALIDATION,
    PAPER_KMEANS_FRAC,
    PAPER_META_LEARNER,
    PAPER_TEST_SIZE,
)
from mth_ids_pipeline.orchestration.experiment_runner import ExperimentConfig
from mth_ids_pipeline.protocol import get_protocol_settings


def test_kmeans_frac_default_is_0008():
    assert DEFAULT_KMEANS_FRAC == 0.008
    assert PAPER_KMEANS_FRAC == 0.008


def test_phase02_frac_cli_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=DEFAULT_KMEANS_FRAC)
    defaults = parser.parse_args([]).__dict__
    assert defaults["frac"] == 0.008


def test_paper_protocol_uses_70_30_split():
    ps = get_protocol_settings("paper")
    assert ps.test_size == PAPER_TEST_SIZE == 0.3


def test_paper_protocol_uses_10_fold_cv():
    ps = get_protocol_settings("paper")
    assert ps.cv_folds == PAPER_CV_FOLDS == 10


def test_paper_protocol_uses_best_base_meta_learner():
    ps = get_protocol_settings("paper")
    assert ps.meta_learner == PAPER_META_LEARNER == "best-base"


def test_paper_protocol_does_not_use_xgb_meta_learner():
    ps = get_protocol_settings("paper")
    assert ps.meta_learner != "xgb"


def test_paper_protocol_hpo_on_validation_with_cv():
    ps = get_protocol_settings("paper")
    assert ps.hpo_on_validation is PAPER_HPO_ON_VALIDATION is True
    assert ps.cv_folds == 10


def test_paper_protocol_normalization_post_split():
    ps = get_protocol_settings("paper")
    assert ps.phase1_zscore is False
    assert ps.supervised_scale_mode == "split"
    assert ps.post_sample_zscore is False


def test_experiment_config_from_paper_protocol():
    cfg = ExperimentConfig.from_protocol("paper")
    assert cfg.test_size == 0.3
    assert cfg.cv_folds == 10
    assert cfg.hpo_on_validation is True
    assert cfg.meta_learner == "best-base"
    assert cfg.kmeans_frac == 0.008
    assert cfg.phase1_zscore is False
    assert cfg.supervised_scale_mode == "split"


def test_notebook_protocol_keeps_legacy_kmeans_frac():
    ps = get_protocol_settings("notebook")
    assert ps.kmeans_frac == 0.008
    assert ps.test_size == 0.2
    assert ps.meta_learner == "xgb"
