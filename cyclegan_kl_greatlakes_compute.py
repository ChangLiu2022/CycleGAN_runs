
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy


def kl_divergence_per_channel_rob(
    signal1: np.ndarray, signal2: np.ndarray, bins: int = 50
) -> np.ndarray:
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)
    if signal1.ndim != 2 or signal2.ndim != 2:
        raise ValueError(
            f"Expected 2D [time, channel]; got {signal1.shape}, {signal2.shape}"
        )
    if signal1.shape[1] != signal2.shape[1]:
        raise ValueError(
            f"Channel mismatch: {signal1.shape[1]} vs {signal2.shape[1]}"
        )

    n_channels = signal1.shape[1]
    kl_divs = np.zeros(n_channels)

    for ch in range(n_channels):
        combined_min = min(signal1[:, ch].min(), signal2[:, ch].min())
        combined_max = max(signal1[:, ch].max(), signal2[:, ch].max())
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)

        hist1, _ = np.histogram(signal1[:, ch], bins=bin_edges, density=False)
        hist2, _ = np.histogram(signal2[:, ch], bins=bin_edges, density=False)

        epsilon = 1e-10
        hist1 = (hist1 + epsilon) / (hist1.sum() + bins * epsilon)
        hist2 = (hist2 + epsilon) / (hist2.sum() + bins * epsilon)

        kl_divs[ch] = entropy(hist1, hist2)

    return kl_divs


def load_transformed_npz(path: str | Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=False)


EXPECTED_KEYS = (
    "dayk_train_aligned",
    "dayk_test_aligned",
    "day0_X_train",
    "dayk_X_train",
    "dayk_X_test",
)


def kl_vs_day0(
    day0_X_train: np.ndarray,
    other: np.ndarray,
    bins: int = 100,
    label: str = "",
) -> dict:
    divs = kl_divergence_per_channel_rob(day0_X_train, other, bins=bins)
    return {
        "label": label,
        "kl_per_channel": divs,
        "mean_kl": float(np.mean(divs)),
        "median_kl": float(np.median(divs)),
    }


def compute_all_kl_for_file(
    path: str | Path,
    results_root: Path,
    bins: int = 100,
) -> dict:
    path = Path(path)
    data = load_transformed_npz(path)
    missing = [k for k in EXPECTED_KEYS if k not in data]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}. Present: {list(data.keys())}")

    day0 = data["day0_X_train"]
    d_train_raw = data["dayk_X_train"]
    d_train_aligned = data["dayk_train_aligned"]
    d_test_raw = data["dayk_X_test"]
    d_test_aligned = data["dayk_test_aligned"]

    try:
        rel = path.resolve().relative_to(results_root.resolve())
        parts = list(rel.with_suffix("").parts)
        base = parts[-1]
        if base.startswith("transformed_"):
            parts[-1] = base.replace("transformed_", "", 1)
        tag = "/".join(parts)
    except ValueError:
        tag = path.stem.replace("transformed_", "", 1)

    results = {
        "file": str(path),
        "tag": tag,
        "train_raw": kl_vs_day0(
            day0, d_train_raw, bins=bins, label="train dayk raw vs day0"
        ),
        "train_aligned": kl_vs_day0(
            day0, d_train_aligned, bins=bins, label="train aligned vs day0"
        ),
        "test_raw": kl_vs_day0(
            day0, d_test_raw, bins=bins, label="test dayk raw vs day0"
        ),
        "test_aligned": kl_vs_day0(
            day0, d_test_aligned, bins=bins, label="test aligned vs day0"
        ),
    }
    data.close()
    return results


def results_to_row(r: dict) -> dict:
    flat = {"file": r["file"], "tag": r["tag"]}
    for split in ("train_raw", "train_aligned", "test_raw", "test_aligned"):
        flat[f"mean_kl_{split}"] = r[split]["mean_kl"]
    return flat


def collect_transformed_npz(results_root: Path) -> list[str]:
    out: list[Path] = []
    root = results_root.resolve()
    if not root.is_dir():
        print(f"WARNING: RESULTS_ROOT is not a directory: {root}", file=sys.stderr)
        return []
    for run_dir in sorted(root.glob("cgan_r*")):
        ckpt = run_dir / "checkpoints"
        if not ckpt.is_dir():
            continue
        out.extend(sorted(ckpt.glob("transformed_*.npz")))
    return [str(p) for p in out]


def all_results_to_jsonable(all_results: list[dict]) -> list[dict]:
    out = []
    for r in all_results:
        d = {
            "file": r["file"],
            "tag": r["tag"],
        }
        for split in ("train_raw", "train_aligned", "test_raw", "test_aligned"):
            x = r[split]
            d[split] = {
                "label": x["label"],
                "mean_kl": x["mean_kl"],
                "median_kl": x["median_kl"],
                "kl_per_channel": x["kl_per_channel"].tolist(),
            }
        out.append(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory that contains cgan_r0, cgan_r1, … (each with checkpoints/transformed_*.npz)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Writes kl_summary.csv, kl_detail.pkl, kl_detail.json, meta.json here",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Histogram bins for KL (default 100, matches LINK_kl_div)",
    )
    args = p.parse_args()

    results_root: Path = args.results_root
    out_dir: Path = args.output_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = collect_transformed_npz(results_root)
    print(
        f"Found {len(npz_paths)} transformed_*.npz file(s) under "
        f"{results_root.resolve()}/cgan_r*/checkpoints/"
    )

    if not npz_paths:
        print(
            "No archives found. Set --results-root or check layout.",
            file=sys.stderr,
        )
        meta = {
            "results_root": str(results_root.resolve()),
            "output_dir": str(out_dir.resolve()),
            "n_archives": 0,
            "bins": args.bins,
            "warning": "no_npz_found",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        sys.exit(0)

    all_results: list[dict] = []
    rows = []
    for path in npz_paths:
        r = compute_all_kl_for_file(path, results_root, bins=args.bins)
        all_results.append(r)
        rows.append(results_to_row(r))

    summary_df = pd.DataFrame(rows)
    summary_df["delta_mean_kl_train_raw_minus_aligned"] = (
        summary_df["mean_kl_train_raw"] - summary_df["mean_kl_train_aligned"]
    )
    summary_df["delta_mean_kl_test_raw_minus_aligned"] = (
        summary_df["mean_kl_test_raw"] - summary_df["mean_kl_test_aligned"]
    )

    csv_path = out_dir / "kl_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    pkl_path = out_dir / "kl_detail.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {pkl_path}")

    json_path = out_dir / "kl_detail.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results_to_jsonable(all_results), f, indent=2)
    print(f"Wrote {json_path}")

    meta = {
        "results_root": str(results_root.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_archives": len(all_results),
        "bins": args.bins,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'meta.json'}")

    print(f"Computed KL for {len(all_results)} archive(s). Done.")


if __name__ == "__main__":
    main()
