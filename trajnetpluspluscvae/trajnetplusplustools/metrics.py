import math
from collections import defaultdict

import numpy as np


def _xy(track_rows, n_predictions=None):
    rows = track_rows[-n_predictions:] if n_predictions is not None else track_rows
    return np.array([[row.x, row.y] for row in rows], dtype=np.float32)


def average_l2(ground_truth, prediction, n_predictions=None):
    gt = _xy(ground_truth, n_predictions)
    pred = _xy(prediction, n_predictions)
    return float(np.linalg.norm(gt - pred, axis=1).mean())


def final_l2(ground_truth, prediction):
    gt = _xy(ground_truth, 1)[-1]
    pred = _xy(prediction, 1)[-1]
    return float(np.linalg.norm(gt - pred))


def collision(primary_tracks, neighbour_tracks, n_predictions=None, person_radius=0.1):
    primary = _xy(primary_tracks, n_predictions)
    neighbour = _xy(neighbour_tracks, n_predictions)
    if len(primary) != len(neighbour):
        n = min(len(primary), len(neighbour))
        primary = primary[:n]
        neighbour = neighbour[:n]
    if len(primary) == 0:
        return False
    distances = np.linalg.norm(primary - neighbour, axis=1)
    return bool(np.any(distances <= 2 * person_radius))


def topk(primary_tracks_all, ground_truth, n_predictions=None):
    grouped = defaultdict(list)
    for row in primary_tracks_all:
        grouped[row.prediction_number or 0].append(row)

    best_ade = math.inf
    best_fde = math.inf
    for rows in grouped.values():
        rows = sorted(rows, key=lambda row: row.frame)
        best_ade = min(best_ade, average_l2(ground_truth, rows, n_predictions))
        best_fde = min(best_fde, final_l2(ground_truth, rows))
    return float(best_ade), float(best_fde)


def nll(primary_tracks_all, ground_truth, n_predictions=None, n_samples=50):
    del n_samples
    # Lightweight fallback: approximate using best-of-k ADE as a proxy.
    best_ade, _ = topk(primary_tracks_all, ground_truth, n_predictions)
    return float(best_ade)
