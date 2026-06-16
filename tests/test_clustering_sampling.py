from __future__ import annotations

import pandas as pd

from mth_ids_pipeline.core.clustering import sample_kmeans


def test_sample_kmeans_encodes_can_text_labels():
    df = pd.DataFrame(
        {
            "CAN_ID": [1, 2, 3, 4, 5, 6],
            "DATA_0": [0, 1, 0, 1, 0, 1],
            "Label": ["BENIGN", "DoS", "Fuzzy", "Gear", "RPM", "BENIGN"],
        }
    )

    sampled = sample_kmeans(
        df,
        n_clusters=2,
        frac=1.0,
        random_state=0,
        minority_labels=(),
    )

    assert set(sampled["Label"]) == {0, 1, 2, 3, 4}
    assert sampled["Label"].dtype == "int64"


def test_sample_kmeans_keeps_numeric_label_sorting():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "Label": ["0", "10", "2", "0"],
        }
    )

    sampled = sample_kmeans(
        df,
        n_clusters=2,
        frac=1.0,
        random_state=0,
        minority_labels=(),
    )

    assert set(sampled["Label"]) == {0, 1, 2}
