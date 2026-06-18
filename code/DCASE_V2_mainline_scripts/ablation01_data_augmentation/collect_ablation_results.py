from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MAINLINE_ROOT = SCRIPT_DIR.parent
RUN_ROOT = MAINLINE_ROOT / "runs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_json_rows(run_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(run_dir.glob("*/final_metrics.json")):
        metrics = read_json(metrics_path)
        view_name = metrics_path.parent.name
        d2 = metrics.get("D2_wav", {})
        d3 = metrics.get("D3_wav", {})
        d2_macro_recall = d2.get("official_domain_acc", "")
        d3_macro_recall = d3.get("official_domain_acc", "")
        d2_sample_acc = d2.get("sample_acc", "")
        d3_sample_acc = d3.get("sample_acc", "")
        avg_macro_recall = metrics.get("avg_D2_D3_wav_official_acc", "")
        avg_sample_acc = ""
        if d2_sample_acc != "" and d3_sample_acc != "":
            avg_sample_acc = (float(d2_sample_acc) + float(d3_sample_acc)) / 2.0
        rows.append(
            {
                "view_name": view_name,
                "tag": "full_wav",
                "best_epoch": metrics.get("best_epoch", ""),
                "best_metric_value": metrics.get("best_metric_value", ""),
                "D2_wav_macro_recall": d2_macro_recall,
                "D3_wav_macro_recall": d3_macro_recall,
                "avg_D2_D3_wav_macro_recall": avg_macro_recall,
                "D2_wav_sample_acc": d2_sample_acc,
                "D3_wav_sample_acc": d3_sample_acc,
                "avg_D2_D3_wav_sample_acc": avg_sample_acc,
                "D2_wav_official_acc": d2_macro_recall,
                "D3_wav_official_acc": d3_macro_recall,
                "avg_D2_D3_wav_official_acc": avg_macro_recall,
                "init_checkpoint": metrics.get("init_checkpoint", ""),
                "task_id": metrics.get("task_id", ""),
                "epochs": metrics.get("epochs", ""),
                "batch_size": metrics.get("batch_size", ""),
                "lr": metrics.get("lr", ""),
                "class_weight": metrics.get("class_weight", ""),
                "run_dir": str(metrics_path.parent),
            }
        )
    return rows


def add_multicenter_rows(run_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(run_dir.glob("*/summary.csv")):
        view_name = summary_path.parent.name
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            for source_row in csv.DictReader(f):
                d2_macro_recall = float(source_row["D2_wav_official_acc"]) / 100.0
                d3_macro_recall = float(source_row["D3_wav_official_acc"]) / 100.0
                avg_macro_recall = float(source_row["avg_D2_D3_wav_official_acc"]) / 100.0
                d2_sample_acc = float(source_row["D2_wav_sample_acc"]) / 100.0
                d3_sample_acc = float(source_row["D3_wav_sample_acc"]) / 100.0
                avg_sample_acc = (d2_sample_acc + d3_sample_acc) / 2.0
                rows.append(
                    {
                        "view_name": view_name,
                        "tag": source_row.get("tag", ""),
                        "best_epoch": "",
                        "best_metric_value": "",
                        "D2_wav_macro_recall": d2_macro_recall,
                        "D3_wav_macro_recall": d3_macro_recall,
                        "avg_D2_D3_wav_macro_recall": avg_macro_recall,
                        "D2_wav_sample_acc": d2_sample_acc,
                        "D3_wav_sample_acc": d3_sample_acc,
                        "avg_D2_D3_wav_sample_acc": avg_sample_acc,
                        "D2_wav_official_acc": d2_macro_recall,
                        "D3_wav_official_acc": d3_macro_recall,
                        "avg_D2_D3_wav_official_acc": avg_macro_recall,
                        "init_checkpoint": "",
                        "task_id": "",
                        "epochs": "",
                        "batch_size": "",
                        "lr": "",
                        "class_weight": "",
                        "run_dir": str(summary_path.parent),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect D2+D3 augmentation ablation metrics into one CSV.")
    parser.add_argument("--run-dir", default=str(RUN_ROOT / "ablation_d23" / "aug_ablation_40ep"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = add_json_rows(run_dir)
    if not rows:
        rows = add_multicenter_rows(run_dir)

    out_path = Path(args.output) if args.output else run_dir / "ablation_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "view_name",
        "tag",
        "best_epoch",
        "best_metric_value",
        "D2_wav_macro_recall",
        "D3_wav_macro_recall",
        "avg_D2_D3_wav_macro_recall",
        "D2_wav_sample_acc",
        "D3_wav_sample_acc",
        "avg_D2_D3_wav_sample_acc",
        "D2_wav_official_acc",
        "D3_wav_official_acc",
        "avg_D2_D3_wav_official_acc",
        "init_checkpoint",
        "task_id",
        "epochs",
        "batch_size",
        "lr",
        "class_weight",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
