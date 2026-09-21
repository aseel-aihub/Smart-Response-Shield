"""
Regenerate the synthetic asset register and the labelled threat dataset.

Usage:
    python scripts/generate_dataset.py [--samples 3000] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASET_SIZE, RANDOM_SEED  # noqa: E402
from src.data_generation import build_and_save  # noqa: E402
from src.facilities import write_facilities_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DATASET_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--noise", type=float, default=None,
                        help="Label noise rate, 0.0-1.0 (default from config).")
    args = parser.parse_args()

    facilities = write_facilities_csv()
    print(f"Asset register written: {len(facilities)} facilities")

    kwargs = {"n_samples": args.samples, "seed": args.seed}
    if args.noise is not None:
        kwargs["label_noise"] = args.noise

    df = build_and_save(**kwargs)

    print(f"Dataset written: {len(df)} rows")
    print("\nSector distribution:")
    print(df["target_sector"].value_counts().to_string())

    corrupted = (df["target_sector"] != df["true_sector"]).mean()
    print(f"\nLabel noise actually applied: {corrupted*100:.1f}%")
    print(f"Mean distance to nearest facility: {df['nearest_facility_km'].mean():.2f} km")
    print(f"Range: {df['range_km'].min():.1f}-{df['range_km'].max():.1f} km")


if __name__ == "__main__":
    main()
