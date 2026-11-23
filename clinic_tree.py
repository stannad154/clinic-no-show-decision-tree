"""Run: python clinic_tree.py [path/to/clinic_appointments.csv]. Standard library only."""

import csv
import random
import sys
from pathlib import Path

FEATURES = [
    "lead_time_days",
    "prior_no_shows",
    "age",
    "distance_miles",
    "sms_reminder",
    "new_patient",
    "appointment_hour",
]
DEPTH = 3
MIN_LEAF = 15
ANNUAL_VOLUME = 10000


def impurity(rows):
    # Gini times node size, so child impurities can be added.
    positives = sum(y for x, y in rows)
    return 2 * positives * (len(rows) - positives) / len(rows)


def fit(rows, depth=0, node=1):
    positives = sum(y for x, y in rows)
    # Choose the cheaper action on training observations in this leaf.
    predict_absence = 60 * (len(rows) - positives) < 120 * positives
    tree = dict(node=node, positives=positives, size=len(rows), absence=predict_absence)
    if depth == DEPTH:
        return tree
    best = impurity(rows)
    split = None
    for feature in FEATURES:
        values = sorted({x[feature] for x, y in rows})
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left = [r for r in rows if r[0][feature] <= threshold]
            right = [r for r in rows if r[0][feature] > threshold]
            if min(len(left), len(right)) < MIN_LEAF:
                continue
            score = impurity(left) + impurity(right)
            if score < best - 1e-10:
                best = score
                split = feature, threshold, left, right
    if split:
        feature, threshold, left, right = split
        print(
            f"Node {node}: {feature} <= {threshold} -> {2 * node}, else {2 * node + 1}"
        )
        tree.update(
            feature=feature,
            threshold=threshold,
            left=fit(left, depth + 1, 2 * node),
            right=fit(right, depth + 1, 2 * node + 1),
        )
    return tree


def leaf(tree, appointment):
    while "feature" in tree:
        branch = (
            "left" if appointment[tree["feature"]] <= tree["threshold"] else "right"
        )
        tree = tree[branch]
    return tree


def decision(tree, appointment):
    return "overbook" if leaf(tree, appointment)["absence"] else "hold"


def evaluate(rows, tree=None):
    counts = [0] * 4  # TN, FP, FN, TP
    for x, y in rows:
        predicted = tree is not None and decision(tree, x) == "overbook"
        counts[2 * y + predicted] += 1
    tn, fp, fn, tp = counts
    loss = 60 * fp + 120 * fn
    print(
        "TN FP FN TP",
        counts,
        "accuracy",
        (tn + tp) / len(rows),
        "precision",
        tp / (tp + fp) if tp + fp else "undefined",
        "recall",
        tp / (tp + fn),
        "value",
        -loss,
        "per visit",
        -loss / len(rows),
    )
    return loss


def main():
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("clinic_appointments.csv")
    )
    with path.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    labels = {"checked-in": 0, "no-show": 1}
    rows = [({k: float(r[k]) for k in FEATURES}, labels[r["no_show"]]) for r in records]
    train, test = [], []
    rng = random.Random(42)
    for label in (0, 1):
        group = [r for r in rows if r[1] == label]
        rng.shuffle(group)
        n = round(0.30 * len(group))
        test.extend(group[:n])
        train.extend(group[n:])
    print(
        "Train / test",
        len(train),
        len(test),
        "no-shows",
        sum(y for x, y in train),
        sum(y for x, y in test),
    )
    tree = fit(train)
    for name, values in [
        ("D", [21, 1, 27, 9.5, 0, 1, 15]),
        ("E", [3, 0, 61, 2, 1, 0, 9]),
    ]:
        appointment = dict(zip(FEATURES, values))
        result = leaf(tree, appointment)
        print(name, decision(tree, appointment), result)
    print("Train v1")
    evaluate(train, tree)
    print("Test baseline")
    baseline = evaluate(test)
    print("Test v1")
    model = evaluate(test, tree)
    savings = baseline - model
    print(
        "Savings: total, per visit, annual at",
        ANNUAL_VOLUME,
        "visits:",
        savings,
        savings / len(test),
        savings / len(test) * ANNUAL_VOLUME,
    )


if __name__ == "__main__":
    main()
