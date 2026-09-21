"""
Train the sector-prediction model and report its performance against an
explicit majority-class baseline.

Usage:
    python scripts/train_model.py [--trees 300] [--depth 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import THREAT_DATA_CSV  # noqa: E402
from src.pattern_model import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--depth", type=int, default=14)
    args = parser.parse_args()

    if not THREAT_DATA_CSV.exists():
        raise SystemExit(
            "No dataset found. Run `python scripts/generate_dataset.py` first."
        )

    _, m = train(n_estimators=args.trees, max_depth=args.depth)

    print("=" * 58)
    print("  SECTOR PREDICTION MODEL")
    print("=" * 58)
    print(f"  Training samples    : {m.n_train}")
    print(f"  Test samples        : {m.n_test}")
    print("-" * 58)
    print(f"  Test accuracy       : {m.accuracy*100:6.2f}%")
    print(f"  Majority baseline   : {m.baseline_accuracy*100:6.2f}%")
    print(f"  Lift over baseline  : {m.lift_over_baseline*100:+6.2f} points")
    print(f"  Macro F1            : {m.macro_f1:6.3f}")
    print(f"  5-fold CV accuracy  : {m.cv_mean*100:6.2f}% +/- {m.cv_std*100:.2f}")
    print("-" * 58)
    print("  Feature importances:")
    for feat, imp in sorted(m.feature_importances.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(imp * 50)
        print(f"    {feat:<20s} {imp:5.3f}  {bar}")
    print("=" * 58)

    if m.lift_over_baseline < 0.05:
        print("\n  WARNING: the model barely beats a constant guess.")
        print("  That is the signature of labels carrying no information about")
        print("  the features. Check the dataset generator before reporting.\n")
    else:
        print(f"\n  Model beats the baseline by {m.lift_over_baseline*100:.1f} points.\n")


if __name__ == "__main__":
    main()
