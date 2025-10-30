"""批量处理音频文件并生成鸟类事件统计结果。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from tqdm import tqdm

from train.test_CNN_LSTM import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_ROOT,
    load_label_maps,
    load_model,
    predict_audio,
)


def iter_audio_paths(sources: Sequence[Path]) -> Iterable[Path]:
    """Yield audio paths from input sources.

    Each source can be a file or directory. Directories will be searched
    recursively for wav/mp3/flac files. Text/CSV files are expected to contain
    a column named ``audio_path`` or plain newline separated paths.
    """

    exts = {".wav", ".mp3", ".flac"}
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"输入路径不存在: {src}")

        if src.is_dir():
            for path in src.rglob("*"):
                if path.suffix.lower() in exts:
                    yield path
        elif src.suffix.lower() in {".csv", ".tsv", ".txt"}:
            delimiter = "," if src.suffix.lower() == ".csv" else "\t"
            with src.open("r", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh, delimiter=delimiter)
                if "audio_path" in reader.fieldnames or reader.fieldnames is None:
                    fh.seek(0)
                    for line in fh:
                        line = line.strip()
                        if not line or line.lower().startswith("audio_path"):
                            continue
                        yield Path(line)
                else:
                    for row in reader:
                        yield Path(row["audio_path"])
        else:
            if src.suffix.lower() not in exts:
                raise ValueError(
                    f"不支持的文件类型: {src.suffix}，请提供音频文件、目录或包含路径的文本/CSV。"
                )
            yield src


def run_batch(
    inputs: Sequence[str],
    model_path: str,
    processed_root: str,
    confidence_threshold: float,
    output_dir: str,
) -> None:
    """Run batch inference over all provided inputs and write CSV outputs."""

    if not inputs:
        raise ValueError("请至少提供一个输入文件或目录。")

    sources = [Path(p).expanduser().resolve() for p in inputs]
    audio_paths = sorted(set(iter_audio_paths(sources)))
    if not audio_paths:
        raise FileNotFoundError("未在输入路径中找到任何音频文件。")

    label_names, idx2label = load_label_maps(processed_root)
    model = load_model(model_path, num_classes=len(label_names))

    events_records: List[dict] = []
    summary_counter = defaultdict(int)

    for audio_path in tqdm(audio_paths, desc="识别音频", unit="文件"):
        events = predict_audio(str(audio_path), model=model, idx2label=idx2label)
        for event in events:
            if event["confidence"] < confidence_threshold:
                continue
            summary_counter[event["label"]] += 1
            events_records.append(
                {
                    "audio_file": audio_path.name,
                    "audio_path": str(audio_path),
                    "start_sec": event["start_sec"],
                    "end_sec": event["end_sec"],
                    "label": event["label"],
                    "confidence": event["confidence"],
                }
            )

    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if events_records:
        events_df = pd.DataFrame(events_records)
        events_df = events_df.sort_values(["audio_file", "start_sec"]).reset_index(drop=True)
        events_path = output_dir_path / "bird_events.csv"
        events_df.to_csv(events_path, index=False, encoding="utf-8-sig")
    else:
        events_df = pd.DataFrame(columns=[
            "audio_file",
            "audio_path",
            "start_sec",
            "end_sec",
            "label",
            "confidence",
        ])
        events_path = output_dir_path / "bird_events.csv"
        events_df.to_csv(events_path, index=False, encoding="utf-8-sig")

    summary_rows = sorted(summary_counter.items(), key=lambda kv: kv[0])
    summary_df = pd.DataFrame(summary_rows, columns=["label", "count"])
    summary_df.loc[len(summary_df)] = [
        "总计",
        int(summary_df["count"].sum()) if not summary_df.empty else 0,
    ]
    summary_path = output_dir_path / "bird_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n处理完成。生成的文件：")
    print(f"- 事件明细: {events_path}")
    print(f"- 物种统计: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量处理音频文件，输出鸟类叫声明细和统计结果。"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="音频文件、文件夹或包含 audio_path 列表的 CSV/TXT",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="模型权重 .pth 文件路径",
    )
    parser.add_argument(
        "--processed-root",
        default=DEFAULT_PROCESSED_ROOT,
        help="包含标签 .npz 文件的目录",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="最小置信度阈值，低于该值的事件将被过滤",
    )
    parser.add_argument(
        "--output-dir",
        default="batch_results",
        help="输出 CSV 文件保存目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(
        inputs=args.inputs,
        model_path=args.model_path,
        processed_root=args.processed_root,
        confidence_threshold=args.confidence_threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
