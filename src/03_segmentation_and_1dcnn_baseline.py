"""Evaluate normal-versus-PVC beat classification with record-level folds.

Each MIT-BIH record is used as a test record exactly once.  A separate record
group selects the PVC probability threshold in each fold, so test records
never choose training epochs or alarm thresholds.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


WINDOW_SIZE = 300
HALF_WINDOW = WINDOW_SIZE // 2
PRE_RR_FEATURE_NAMES = ("pre_rr_seconds", "pre_rr_ratio")
PREPOST_RR_FEATURE_NAMES = PRE_RR_FEATURE_NAMES + ("post_rr_seconds", "post_to_pre_rr_ratio")
ROLLING_RR_HISTORY_SIZE = 8
TARGET_PVC_RECALL = 0.90
CALIBRATED_TARGET_PVC_RECALL = 0.85
CALIBRATED_TRAINING_EPOCHS = 6
BEAT_SYMBOLS = {"N", "V"}
# These are the beat annotations defined by WFDB.  They are used only to find
# the preceding R peak; the classifier itself remains restricted to N and V.
R_PEAK_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "/", "f", "Q", "?", "P", "U",
}
BALANCE_FEATURES = (
    "valid_beats",
    "normal_N",
    "pvc_V",
)
BALANCE_WEIGHTS = np.asarray((1.0, 1.5, 2.5), dtype=float)

# This fixed allocation keeps the six independent commands reproducible.  It
# gives each test group 1,132-1,268 V beats while keeping records 201 and 202,
# which come from the same subject, in the same fold.
OPTIMIZED_FOLD_GROUPS = [
    ["103", "108", "109", "116", "203", "217", "222", "228"],
    ["101", "123", "200", "207", "212", "213", "220", "234"],
    ["105", "107", "124", "201", "202", "205", "231", "233"],
    ["104", "106", "111", "113", "117", "119", "209", "215"],
    ["100", "102", "114", "118", "210", "219", "221", "223"],
    ["112", "115", "121", "122", "208", "214", "230", "232"],
]


def normalise_beat(beat: np.ndarray) -> np.ndarray:
    std = beat.std()
    return (beat - beat.mean()) / (std if std > 1e-8 else 1.0)


def load_record(
    record_base: Path,
    include_post_rr: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    record = wfdb.rdrecord(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")
    signal = record.p_signal[:, 0]
    beats, rhythm_features, labels, symbols = [], [], [], Counter()
    cardiac_beats = [
        (peak, symbol)
        for peak, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in R_PEAK_SYMBOLS
    ]
    previous_rr_intervals = []
    # The post-RR experiment delays classification until the next R peak is
    # known. Earlier feature sets keep their original causal pre-RR samples.
    final_index = len(cardiac_beats) - 1 if include_post_rr else len(cardiac_beats)
    for index in range(1, final_index):
        previous_peak, _ = cardiac_beats[index - 1]
        peak, symbol = cardiac_beats[index]
        pre_rr_samples = peak - previous_peak
        post_rr_samples = (
            cardiac_beats[index + 1][0] - peak
            if index + 1 < len(cardiac_beats)
            else np.nan
        )
        if pre_rr_samples <= 0:
            continue
        if previous_rr_intervals:
            reference_rr = np.median(previous_rr_intervals[-ROLLING_RR_HISTORY_SIZE:])
            pre_rr_ratio = pre_rr_samples / reference_rr if reference_rr > 0 else 1.0
        else:
            pre_rr_ratio = 1.0
        previous_rr_intervals.append(pre_rr_samples)
        if symbol not in BEAT_SYMBOLS:
            continue
        if peak < HALF_WINDOW or peak + HALF_WINDOW >= len(signal):
            continue
        beat = signal[peak - HALF_WINDOW : peak + HALF_WINDOW]
        beats.append(normalise_beat(beat).astype(np.float32))
        rhythm_features.append(
            (
                pre_rr_samples / record.fs,
                pre_rr_ratio,
                post_rr_samples / record.fs if np.isfinite(post_rr_samples) else np.nan,
                post_rr_samples / pre_rr_samples if np.isfinite(post_rr_samples) else np.nan,
            )
        )
        labels.append(0 if symbol == "N" else 1)
        symbols[symbol] += 1
    return (
        np.asarray(beats, dtype=np.float32).reshape((-1, WINDOW_SIZE, 1)),
        np.asarray(rhythm_features, dtype=np.float32).reshape((-1, len(PREPOST_RR_FEATURE_NAMES))),
        np.asarray(labels, dtype=np.int32),
        dict(symbols),
    )


def load_all_records(
    data_dir: Path,
    feature_set: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]]:
    record_names = sorted(path.stem for path in data_dir.glob("*.hea"))
    if not record_names:
        raise FileNotFoundError(f"Expected MIT-BIH .hea files in {data_dir}")
    include_post_rr = feature_set == "waveform-prepost-rr"
    return {name: load_record(data_dir / name, include_post_rr) for name in record_names}


def balance_vector(labels: np.ndarray) -> np.ndarray:
    """Summarise one record using the exact valid-beat rule used for training."""
    return np.asarray(
        (
            len(labels),
            int((labels == 0).sum()),
            int((labels == 1).sum()),
        ),
        dtype=float,
    )


def balance_score(group_totals: np.ndarray, target: np.ndarray) -> float:
    """Weighted squared relative deviation from the ideal per-group totals."""
    relative_deviation = (group_totals - target) / np.maximum(target, 1.0)
    return float(np.sum(BALANCE_WEIGHTS * np.square(relative_deviation)))


def build_balanced_groups(record_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]], folds: int) -> list[list[str]]:
    """Create deterministic equal-record folds by optimizing N/V balance."""
    names = sorted(record_data)
    if len(names) % folds:
        raise ValueError(f"{len(names)} records cannot be split equally into {folds} folds")
    records_per_fold = len(names) // folds
    if folds == len(OPTIMIZED_FOLD_GROUPS) and set(names) == {
        name for group in OPTIMIZED_FOLD_GROUPS for name in group
    }:
        return [group.copy() for group in OPTIMIZED_FOLD_GROUPS]

    # The fallback supports another complete dataset while retaining the same
    # optimization criteria. The local MIT-BIH setup uses the checked schedule above.
    feature_matrix = np.asarray(
        [balance_vector(record_data[name][2]) for name in names], dtype=float
    )
    target = feature_matrix.sum(axis=0) / folds
    group_slots = np.repeat(np.arange(folds), records_per_fold)
    rng = np.random.default_rng(20260724)
    best_assignment, best_score = None, float("inf")

    # Random restarts plus record swaps avoid the irreversible early choices of
    # the former greedy allocator while keeping each group at exactly 8 records.
    for _ in range(120):
        assignment = group_slots[rng.permutation(len(names))].copy()
        totals = np.vstack([feature_matrix[assignment == group].sum(axis=0) for group in range(folds)])
        current_score = balance_score(totals, target)
        for step in range(5000):
            left, right = rng.choice(len(names), size=2, replace=False)
            left_group, right_group = assignment[left], assignment[right]
            if left_group == right_group:
                continue
            candidate_totals = totals.copy()
            candidate_totals[left_group] += feature_matrix[right] - feature_matrix[left]
            candidate_totals[right_group] += feature_matrix[left] - feature_matrix[right]
            candidate_score = balance_score(candidate_totals, target)
            temperature = 0.05 * (1.0 - step / 5000) + 0.001
            if candidate_score < current_score or rng.random() < np.exp((current_score - candidate_score) / temperature):
                assignment[left], assignment[right] = right_group, left_group
                totals, current_score = candidate_totals, candidate_score
        if current_score < best_score:
            best_assignment, best_score = assignment.copy(), current_score

    return [sorted(name for name, group in zip(names, best_assignment) if group == index) for index in range(folds)]


def describe_groups(record_data, groups) -> dict:
    """Return the evidence that the six groups use the same data composition."""
    rows = []
    for index, names in enumerate(groups, start=1):
        totals = np.sum(
            [balance_vector(record_data[name][2]) for name in names], axis=0
        )
        row = {feature: int(value) for feature, value in zip(BALANCE_FEATURES, totals)}
        row.update({"group": index, "records": names, "record_count": len(names)})
        rows.append(row)
    return {"features": list(BALANCE_FEATURES), "groups": rows}


def combine_records(
    record_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]],
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([record_data[name][0] for name in names], axis=0),
        np.concatenate([record_data[name][1] for name in names], axis=0),
        np.concatenate([record_data[name][2] for name in names], axis=0),
    )


def uses_rhythm_features(feature_set: str) -> bool:
    return feature_set != "waveform"


def experiment_prefix(feature_set: str) -> str:
    return {
        "waveform": "pvc",
        "waveform-rr": "pvc_rr",
        "waveform-prepost-rr": "pvc_prepost_rr",
    }[feature_set]


def rhythm_feature_names(feature_set: str) -> tuple[str, ...]:
    if feature_set == "waveform-rr":
        return PRE_RR_FEATURE_NAMES
    if feature_set == "waveform-prepost-rr":
        return PREPOST_RR_FEATURE_NAMES
    return ()


def selected_rhythm_features(features: np.ndarray, feature_set: str) -> np.ndarray:
    return features[:, : len(rhythm_feature_names(feature_set))]


def input_feature_description(feature_set: str) -> str:
    if feature_set == "waveform-rr":
        return "waveform plus pre-RR interval features"
    if feature_set == "waveform-prepost-rr":
        return "waveform plus pre-RR and post-RR interval features"
    return "waveform only"


def fit_rhythm_feature_scaler(features: np.ndarray, feature_set: str) -> dict[str, np.ndarray]:
    """Fit feature scaling on a training split only."""
    selected_features = selected_rhythm_features(features, feature_set)
    mean = selected_features.mean(axis=0)
    scale = selected_features.std(axis=0)
    return {"mean": mean, "scale": np.where(scale > 1e-8, scale, 1.0)}


def scale_rhythm_features(features: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((features - scaler["mean"]) / scaler["scale"]).astype(np.float32)


def model_inputs(
    waveforms: np.ndarray,
    rhythm_features: np.ndarray,
    feature_set: str,
    scaler: dict[str, np.ndarray] | None = None,
):
    if not uses_rhythm_features(feature_set):
        return waveforms
    if scaler is None:
        raise ValueError("Rhythm-feature scaling must be fitted on the training split")
    return {
        "ecg_beat": waveforms,
        "rr_features": scale_rhythm_features(selected_rhythm_features(rhythm_features, feature_set), scaler),
    }


def scaler_metadata(scaler: dict[str, np.ndarray] | None, feature_set: str) -> dict | None:
    if scaler is None:
        return None
    return {
        "feature_names": list(rhythm_feature_names(feature_set)),
        "mean": scaler["mean"].astype(float).tolist(),
        "standard_deviation": scaler["scale"].astype(float).tolist(),
        "normalization": "z-score fitted on the corresponding training records only",
    }


def build_model(feature_set: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1), name="ecg_beat")
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    model_inputs_list = inputs
    if uses_rhythm_features(feature_set):
        rhythm_inputs = tf.keras.Input(shape=(len(rhythm_feature_names(feature_set)),), name="rr_features")
        rhythm_x = tf.keras.layers.Dense(8, activation="relu", name="rr_feature_encoder")(rhythm_inputs)
        x = tf.keras.layers.Concatenate(name="fuse_waveform_and_rr")([x, rhythm_x])
        model_inputs_list = {"ecg_beat": inputs, "rr_features": rhythm_inputs}
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class")(x)
    model_name = f"tiny_1dcnn_{experiment_prefix(feature_set)}"
    model = tf.keras.Model(model_inputs_list, outputs, name=model_name)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(class_id=1, name="pvc_recall")],
    )
    return model


def class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    include_probability_metrics: bool = False,
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int32)
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["normal_N", "PVC_V"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    pvc_recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(report["accuracy"]),
        "specificity": float(specificity),
        "balanced_accuracy": float((specificity + pvc_recall) / 2),
        "pvc_precision": float(report["PVC_V"]["precision"]),
        "pvc_recall": float(report["PVC_V"]["recall"]),
        "pvc_f1": float(report["PVC_V"]["f1-score"]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    if include_probability_metrics:
        metrics.update(
            {
                "auroc": float(roc_auc_score(labels, probabilities)),
                "auprc": float(average_precision_score(labels, probabilities)),
            }
        )
    return metrics


def select_threshold(validation_labels: np.ndarray, validation_probabilities: np.ndarray) -> tuple[dict, bool]:
    candidates = [
        threshold_metrics(validation_labels, validation_probabilities, threshold)
        for threshold in np.arange(0.01, 1.00, 0.01)
    ]
    feasible = [item for item in candidates if item["pvc_recall"] >= TARGET_PVC_RECALL]
    if feasible:
        return max(
            feasible,
            key=lambda item: (item["pvc_precision"], item["pvc_f1"], item["threshold"]),
        ), True
    return max(
        candidates,
        key=lambda item: (item["pvc_recall"], item["pvc_f1"], item["pvc_precision"]),
    ), False


def select_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
) -> tuple[dict, bool]:
    """Select one predeclared operating point from out-of-fold predictions."""
    candidates = [
        threshold_metrics(labels, probabilities, threshold)
        for threshold in np.arange(0.01, 1.00, 0.01)
    ]
    feasible = [item for item in candidates if item["pvc_recall"] >= target_recall]
    if feasible:
        return max(
            feasible,
            key=lambda item: (item["pvc_f1"], item["pvc_precision"], item["threshold"]),
        ), True
    return max(
        candidates,
        key=lambda item: (item["pvc_recall"], item["pvc_f1"], item["pvc_precision"]),
    ), False


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_platt_scaler(labels: np.ndarray, probabilities: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling only on cross-fitted, never-in-sample probabilities."""
    scaler = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    scaler.fit(probabilities_to_logits(probabilities), labels)
    return scaler


def apply_platt_scaler(scaler: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return scaler.predict_proba(probabilities_to_logits(probabilities))[:, 1]


def metric_summary(folds: list[dict]) -> dict:
    metric_names = (
        "accuracy",
        "specificity",
        "balanced_accuracy",
        "pvc_precision",
        "pvc_recall",
        "pvc_f1",
        "auroc",
        "auprc",
    )
    summary = {}
    for name in metric_names:
        values = np.asarray([fold["test_metrics"][name] for fold in folds], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    combined_confusion = np.sum(
        [np.asarray(fold["test_metrics"]["confusion_matrix"], dtype=int) for fold in folds], axis=0
    )
    tn, fp, fn, tp = combined_confusion.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    summary["pooled_test_metrics"] = {
        "sample_count": int(combined_confusion.sum()),
        "accuracy": float((tn + tp) / combined_confusion.sum()),
        "specificity": float(specificity),
        "balanced_accuracy": float((specificity + recall) / 2),
        "pvc_precision": float(precision),
        "pvc_recall": float(recall),
        "pvc_f1": float(2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "confusion_matrix": combined_confusion.tolist(),
    }
    return summary


def run_fold(record_data, groups, group_description, fold_index, epochs, batch_size, seed, feature_set) -> dict:
    all_names = sorted(record_data)
    test_records = groups[fold_index]
    validation_records = groups[(fold_index + 1) % len(groups)]
    train_records = sorted(set(all_names) - set(test_records) - set(validation_records))
    x_train, rr_train, y_train = combine_records(record_data, train_records)
    x_validation, rr_validation, y_validation = combine_records(record_data, validation_records)
    x_test, rr_test, y_test = combine_records(record_data, test_records)
    rhythm_scaler = fit_rhythm_feature_scaler(rr_train, feature_set) if uses_rhythm_features(feature_set) else None
    train_inputs = model_inputs(x_train, rr_train, feature_set, rhythm_scaler)
    validation_inputs = model_inputs(x_validation, rr_validation, feature_set, rhythm_scaler)
    test_inputs = model_inputs(x_test, rr_test, feature_set, rhythm_scaler)
    print(f"Fold {fold_index + 1}/{len(groups)}")
    print(f"Train {x_train.shape}, validation {x_validation.shape}, test {x_test.shape}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed + fold_index)
    model = build_model(feature_set)
    history = model.fit(
        train_inputs, y_train,
        validation_data=(validation_inputs, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0,
    )
    validation_probabilities = model.predict(validation_inputs, verbose=0)[:, 1]
    selected_threshold, target_met = select_threshold(y_validation, validation_probabilities)
    validation_metrics = threshold_metrics(
        y_validation,
        validation_probabilities,
        selected_threshold["threshold"],
        include_probability_metrics=True,
    )
    test_probabilities = model.predict(test_inputs, verbose=0)[:, 1]
    test_metrics = threshold_metrics(
        y_test,
        test_probabilities,
        selected_threshold["threshold"],
        include_probability_metrics=True,
    )
    metric_keys = (
        "accuracy",
        "specificity",
        "balanced_accuracy",
        "pvc_precision",
        "pvc_recall",
        "pvc_f1",
        "auroc",
        "auprc",
        "confusion_matrix",
    )
    return {
        "task": "MIT-BIH normal beat (N) versus premature ventricular contraction (V)",
        "label_definition": {"0": "normal_N", "1": "PVC_V", "excluded": "all annotation symbols except N and V"},
        "input_features": input_feature_description(feature_set),
        "rhythm_feature_scaler": scaler_metadata(rhythm_scaler, feature_set),
        "fold": fold_index + 1,
        "epochs_ran": len(history.history["loss"]),
        "fold_group_balance": group_description,
        "splits": {
            "train": {"records": train_records, "sample_count": int(len(y_train))},
            "validation": {"records": validation_records, "sample_count": int(len(y_validation))},
            "test": {"records": test_records, "sample_count": int(len(y_test))},
        },
        "threshold_selection": {
            "target_pvc_recall": TARGET_PVC_RECALL,
            "target_met_on_validation": target_met,
            "selected_threshold": selected_threshold["threshold"],
            "validation_metrics": {
                key: validation_metrics[key]
                for key in metric_keys
            },
        },
        "test_metrics": {
            key: test_metrics[key]
            for key in metric_keys
        },
    }


def train_fixed_epochs(
    train_inputs,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    feature_set: str,
) -> tf.keras.Model:
    """Train a cross-fitting model without using its held-out prediction group."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(feature_set)
    model.fit(
        train_inputs,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        verbose=0,
    )
    return model


def run_calibrated_fold(
    record_data,
    groups,
    group_description,
    fold_index,
    epochs,
    batch_size,
    seed,
    feature_set,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate one outer test group with inner group-level cross-fitted Platt scaling."""
    all_names = sorted(record_data)
    test_records = groups[fold_index]
    development_groups = [group for index, group in enumerate(groups) if index != fold_index]
    development_records = sorted(name for group in development_groups for name in group)
    x_test, rr_test, y_test = combine_records(record_data, test_records)

    oof_labels, oof_raw_probabilities, inner_splits = [], [], []
    for inner_index, calibration_records in enumerate(development_groups):
        inner_train_records = sorted(
            name for group_index, group in enumerate(development_groups)
            if group_index != inner_index for name in group
        )
        x_inner_train, rr_inner_train, y_inner_train = combine_records(record_data, inner_train_records)
        x_inner_calibration, rr_inner_calibration, y_inner_calibration = combine_records(record_data, calibration_records)
        inner_scaler = fit_rhythm_feature_scaler(rr_inner_train, feature_set) if uses_rhythm_features(feature_set) else None
        inner_model = train_fixed_epochs(
            model_inputs(x_inner_train, rr_inner_train, feature_set, inner_scaler),
            y_inner_train,
            epochs,
            batch_size,
            seed + fold_index * 10 + inner_index,
            feature_set,
        )
        inner_probabilities = inner_model.predict(
            model_inputs(x_inner_calibration, rr_inner_calibration, feature_set, inner_scaler),
            verbose=0,
        )[:, 1]
        oof_labels.append(y_inner_calibration)
        oof_raw_probabilities.append(inner_probabilities)
        inner_splits.append(
            {
                "train_records": inner_train_records,
                "calibration_records": calibration_records,
                "train_sample_count": int(len(y_inner_train)),
                "calibration_sample_count": int(len(y_inner_calibration)),
                "rhythm_feature_scaler": scaler_metadata(inner_scaler, feature_set),
            }
        )

    y_oof = np.concatenate(oof_labels)
    raw_oof_probabilities = np.concatenate(oof_raw_probabilities)
    platt_scaler = fit_platt_scaler(y_oof, raw_oof_probabilities)
    calibrated_oof_probabilities = apply_platt_scaler(platt_scaler, raw_oof_probabilities)
    selected_threshold, target_met = select_f1_threshold(
        y_oof,
        calibrated_oof_probabilities,
        CALIBRATED_TARGET_PVC_RECALL,
    )
    oof_metrics = threshold_metrics(
        y_oof,
        calibrated_oof_probabilities,
        selected_threshold["threshold"],
        include_probability_metrics=True,
    )

    x_development, rr_development, y_development = combine_records(record_data, development_records)
    outer_scaler = fit_rhythm_feature_scaler(rr_development, feature_set) if uses_rhythm_features(feature_set) else None
    outer_model = train_fixed_epochs(
        model_inputs(x_development, rr_development, feature_set, outer_scaler),
        y_development,
        epochs,
        batch_size,
        seed + fold_index * 10 + len(development_groups),
        feature_set,
    )
    raw_test_probabilities = outer_model.predict(
        model_inputs(x_test, rr_test, feature_set, outer_scaler),
        verbose=0,
    )[:, 1]
    calibrated_test_probabilities = apply_platt_scaler(platt_scaler, raw_test_probabilities)
    test_metrics = threshold_metrics(
        y_test,
        calibrated_test_probabilities,
        selected_threshold["threshold"],
        include_probability_metrics=True,
    )
    metric_keys = (
        "accuracy",
        "specificity",
        "balanced_accuracy",
        "pvc_precision",
        "pvc_recall",
        "pvc_f1",
        "auroc",
        "auprc",
        "confusion_matrix",
    )
    result = {
        "task": "MIT-BIH normal beat (N) versus premature ventricular contraction (V)",
        "protocol": "Nested record-group cross-fitting with Platt scaling",
        "label_definition": {"0": "normal_N", "1": "PVC_V", "excluded": "all annotation symbols except N and V"},
        "input_features": input_feature_description(feature_set),
        "rhythm_feature_scaler": scaler_metadata(outer_scaler, feature_set),
        "fold": fold_index + 1,
        "outer_test_records": test_records,
        "inner_cross_fitting": inner_splits,
        "calibration": {
            "method": "Platt scaling on logits of inner out-of-fold probabilities",
            "oof_sample_count": int(len(y_oof)),
            "raw_oof_brier_score": float(brier_score_loss(y_oof, raw_oof_probabilities)),
            "calibrated_oof_brier_score": float(brier_score_loss(y_oof, calibrated_oof_probabilities)),
            "coefficient": float(platt_scaler.coef_[0, 0]),
            "intercept": float(platt_scaler.intercept_[0]),
            "fixed_training_epochs": int(epochs),
        },
        "splits": {
            "development": {"records": development_records, "sample_count": int(len(y_development))},
            "test": {"records": test_records, "sample_count": int(len(y_test))},
        },
        "threshold_selection": {
            "rule": "Maximize PVC F1 among calibrated OOF thresholds with PVC recall at least 0.85",
            "target_pvc_recall": CALIBRATED_TARGET_PVC_RECALL,
            "target_met_on_oof": target_met,
            "selected_threshold": selected_threshold["threshold"],
            "oof_metrics": {key: oof_metrics[key] for key in metric_keys},
        },
        "test_metrics": {key: test_metrics[key] for key in metric_keys},
    }
    arrays = {
        "oof_labels": y_oof,
        "oof_raw_probabilities": raw_oof_probabilities,
        "oof_calibrated_probabilities": calibrated_oof_probabilities,
        "test_labels": y_test,
        "test_raw_probabilities": raw_test_probabilities,
        "test_calibrated_probabilities": calibrated_test_probabilities,
    }
    return result, arrays


def finalise_experiment(record_data, groups, group_description, fold_results, args):
    all_names = sorted(record_data)
    summary = metric_summary(fold_results)
    final_threshold = float(np.median([fold["threshold_selection"]["selected_threshold"] for fold in fold_results]))
    final_epochs = int(round(np.median([fold["epochs_ran"] for fold in fold_results])))
    x_all, rr_all, y_all = combine_records(record_data, all_names)
    rhythm_scaler = fit_rhythm_feature_scaler(rr_all, args.feature_set) if uses_rhythm_features(args.feature_set) else None
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed + args.folds)
    final_model = build_model(args.feature_set)
    final_model.fit(
        model_inputs(x_all, rr_all, args.feature_set, rhythm_scaler), y_all,
        epochs=final_epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(y_all),
        verbose=0,
    )
    result = {
        "task": "MIT-BIH normal beat (N) versus premature ventricular contraction (V)",
        "label_definition": {"0": "normal_N", "1": "PVC_V", "excluded": "all annotation symbols except N and V"},
        "evaluation_protocol": "Six-fold record-level cross-validation with cyclic validation groups",
        "window_size": WINDOW_SIZE,
        "parameter_count": int(final_model.count_params()),
        "input_features": input_feature_description(args.feature_set),
        "rhythm_feature_scaler": scaler_metadata(rhythm_scaler, args.feature_set),
        "threshold_selection_rule": "Highest PVC precision among thresholds meeting validation PVC recall target",
        "candidate_thresholds": "0.01 to 0.99 in increments of 0.01",
        "fold_groups": groups,
        "fold_group_balance": group_description,
        "folds": fold_results,
        "summary": summary,
        "final_deployment_model": {
            "training_records": all_names,
            "sample_count": int(len(y_all)),
            "normal_N_count": int((y_all == 0).sum()),
            "pvc_V_count": int((y_all == 1).sum()),
            "epochs": final_epochs,
            "pvc_threshold": final_threshold,
            "threshold_source": "Median of six validation-selected thresholds",
        },
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    prefix = experiment_prefix(args.feature_set)
    final_model.save(args.models_dir / f"tiny_1dcnn_{prefix}.keras")
    rendered = json.dumps(result, indent=2)
    (args.models_dir / f"{prefix}_vs_normal_metrics.json").write_text(rendered, encoding="utf-8")
    print(f"Saved final deployment model and reports to {args.models_dir}")


def finalise_calibrated_experiment(record_data, groups, group_description, fold_results, args):
    """Train one deployment model and derive its calibration from outer OOF predictions."""
    all_names = sorted(record_data)
    summary = metric_summary(fold_results)
    oof_labels, oof_raw_probabilities = [], []
    prefix = experiment_prefix(args.feature_set)
    for fold_index in range(1, args.folds + 1):
        artifact_path = args.models_dir / f"{prefix}_calibrated_fold_{fold_index}_probabilities.npz"
        artifact = np.load(artifact_path)
        oof_labels.append(artifact["test_labels"])
        oof_raw_probabilities.append(artifact["test_raw_probabilities"])
    y_outer_oof = np.concatenate(oof_labels)
    raw_outer_oof_probabilities = np.concatenate(oof_raw_probabilities)
    deployment_scaler = fit_platt_scaler(y_outer_oof, raw_outer_oof_probabilities)
    calibrated_outer_oof_probabilities = apply_platt_scaler(deployment_scaler, raw_outer_oof_probabilities)
    deployment_threshold, deployment_target_met = select_f1_threshold(
        y_outer_oof,
        calibrated_outer_oof_probabilities,
        CALIBRATED_TARGET_PVC_RECALL,
    )

    x_all, rr_all, y_all = combine_records(record_data, all_names)
    rhythm_scaler = fit_rhythm_feature_scaler(rr_all, args.feature_set) if uses_rhythm_features(args.feature_set) else None
    final_model = train_fixed_epochs(
        model_inputs(x_all, rr_all, args.feature_set, rhythm_scaler),
        y_all,
        args.calibrated_epochs,
        args.batch_size,
        args.seed + args.folds * 100,
        args.feature_set,
    )
    result = {
        "task": "MIT-BIH normal beat (N) versus premature ventricular contraction (V)",
        "protocol": "Nested record-group cross-fitting with Platt scaling",
        "label_definition": {"0": "normal_N", "1": "PVC_V", "excluded": "all annotation symbols except N and V"},
        "evaluation_protocol": "Six-fold outer record-level evaluation; inner five-group cross-fitting for calibration",
        "window_size": WINDOW_SIZE,
        "parameter_count": int(final_model.count_params()),
        "input_features": input_feature_description(args.feature_set),
        "rhythm_feature_scaler": scaler_metadata(rhythm_scaler, args.feature_set),
        "fold_groups": groups,
        "fold_group_balance": group_description,
        "folds": fold_results,
        "summary": summary,
        "final_deployment_model": {
            "training_records": all_names,
            "sample_count": int(len(y_all)),
            "normal_N_count": int((y_all == 0).sum()),
            "pvc_V_count": int((y_all == 1).sum()),
            "fixed_training_epochs": int(args.calibrated_epochs),
            "calibration_method": "Platt scaling fitted on six outer out-of-fold prediction sets",
            "calibration_coefficient": float(deployment_scaler.coef_[0, 0]),
            "calibration_intercept": float(deployment_scaler.intercept_[0]),
            "raw_oof_brier_score": float(brier_score_loss(y_outer_oof, raw_outer_oof_probabilities)),
            "calibrated_oof_brier_score": float(brier_score_loss(y_outer_oof, calibrated_outer_oof_probabilities)),
            "pvc_threshold": float(deployment_threshold["threshold"]),
            "threshold_rule": "Maximize PVC F1 among calibrated OOF thresholds with PVC recall at least 0.85",
            "target_recall_met_on_outer_oof": deployment_target_met,
        },
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(args.models_dir / f"tiny_1dcnn_{prefix}_calibrated.keras")
    rendered = json.dumps(result, indent=2)
    (args.models_dir / f"{prefix}_calibrated_metrics.json").write_text(rendered, encoding="utf-8")
    print(f"Saved calibrated deployment model and reports to {args.models_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--calibrated-epochs",
        type=int,
        default=CALIBRATED_TRAINING_EPOCHS,
        help="Fixed epochs for nested Platt cross-fitting; avoids using a calibration group for early stopping",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol",
        choices=("baseline", "nested-platt"),
        default="baseline",
        help="Use baseline evaluation or nested record-group Platt calibration",
    )
    parser.add_argument(
        "--feature-set",
        choices=("waveform", "waveform-rr", "waveform-prepost-rr"),
        default="waveform",
        help="Use waveform only, pre-RR fusion, or pre/post-RR fusion",
    )
    parser.add_argument("--fold-index", type=int, help="Run one fold, using 1-based numbering")
    parser.add_argument("--finalize", action="store_true", help="Summarise six saved folds and train final model")
    parser.add_argument("--show-groups", action="store_true", help="Print the six-group balance check without training")
    args = parser.parse_args()
    chosen_modes = int(args.fold_index is not None) + int(args.finalize) + int(args.show_groups)
    if chosen_modes != 1:
        parser.error("Choose exactly one of --fold-index N, --finalize, or --show-groups")

    record_data = load_all_records(args.data_dir, args.feature_set)
    groups = build_balanced_groups(record_data, args.folds)
    group_description = describe_groups(record_data, groups)
    if args.show_groups:
        print(json.dumps(group_description, indent=2))
        return
    args.models_dir.mkdir(parents=True, exist_ok=True)
    if args.fold_index is not None:
        if not 1 <= args.fold_index <= args.folds:
            parser.error(f"--fold-index must be between 1 and {args.folds}")
        if args.protocol == "baseline":
            result = run_fold(
                record_data,
                groups,
                group_description,
                args.fold_index - 1,
                args.epochs,
                args.batch_size,
                args.seed,
                args.feature_set,
            )
            output_path = args.models_dir / f"{experiment_prefix(args.feature_set)}_vs_normal_fold_{args.fold_index}.json"
        else:
            result, arrays = run_calibrated_fold(
                record_data,
                groups,
                group_description,
                args.fold_index - 1,
                args.calibrated_epochs,
                args.batch_size,
                args.seed,
                args.feature_set,
            )
            output_path = args.models_dir / f"{experiment_prefix(args.feature_set)}_calibrated_fold_{args.fold_index}.json"
            np.savez_compressed(
                args.models_dir / f"{experiment_prefix(args.feature_set)}_calibrated_fold_{args.fold_index}_probabilities.npz",
                **arrays,
            )
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved fold {args.fold_index} result to {output_path}")
        return

    fold_results = []
    prefix = experiment_prefix(args.feature_set)
    for fold_index in range(1, args.folds + 1):
        if args.protocol == "baseline":
            path = args.models_dir / f"{prefix}_vs_normal_fold_{fold_index}.json"
        else:
            path = args.models_dir / f"{prefix}_calibrated_fold_{fold_index}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run --fold-index {fold_index} first")
        fold_results.append(json.loads(path.read_text(encoding="utf-8")))
    if args.protocol == "baseline":
        finalise_experiment(record_data, groups, group_description, fold_results, args)
    else:
        finalise_calibrated_experiment(record_data, groups, group_description, fold_results, args)


if __name__ == "__main__":
    main()
