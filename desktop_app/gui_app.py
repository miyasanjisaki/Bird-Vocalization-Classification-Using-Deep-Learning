"""基于 PySimpleGUI 的桌面端鸟类叫声识别应用。"""

from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import PySimpleGUI as sg

from train.test_CNN_LSTM import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_ROOT,
    load_label_maps,
    load_model,
    predict_audio,
)

CUSTOM_THEME = {
    "BACKGROUND": "#0b1120",
    "TEXT": "#f8fafc",
    "INPUT": "#111827",
    "TEXT_INPUT": "#e2e8f0",
    "SCROLL": "#1f2937",
    "BUTTON": ("#0b1120", "#38bdf8"),
    "PROGRESS": ("#38bdf8", "#94a3b8"),
    "BORDER": 0,
    "SLIDER_DEPTH": 0,
    "PROGRESS_DEPTH": 0,
}

sg.theme_add_new("BirdVision", CUSTOM_THEME)
sg.theme("BirdVision")
sg.set_options(font=("Microsoft YaHei", 11))

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "PingFang SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


@lru_cache(maxsize=2)
def _load_detector(model_path: str, processed_root: str):
    label_names, idx2label = load_label_maps(processed_root)
    model = load_model(model_path, num_classes=len(label_names))
    return model, idx2label


def build_layout(default_model: str, default_processed_root: str) -> list:
    control_frame = sg.Frame(
        "模型与数据",
        [
            [
                sg.Text("模型权重", size=(10, 1)),
                sg.Input(default_model, key="-MODEL-", expand_x=True),
                sg.FileBrowse("浏览", file_types=(("PyTorch 模型", "*.pth"),)),
            ],
            [
                sg.Text("标签目录", size=(10, 1)),
                sg.Input(default_processed_root, key="-LABELS-", expand_x=True),
                sg.FolderBrowse("浏览"),
            ],
            [
                sg.Text("音频文件", size=(10, 1)),
                sg.Input(key="-AUDIO-", expand_x=True),
                sg.FileBrowse("浏览", file_types=(("音频文件", "*.wav;*.mp3;*.flac"),)),
            ],
            [
                sg.Text("置信度阈值", size=(10, 1)),
                sg.Slider(
                    range=(0.0, 1.0),
                    orientation="h",
                    resolution=0.05,
                    default_value=0.0,
                    expand_x=True,
                    key="-THRESH-",
                ),
            ],
            [
                sg.Button("识别", key="-RUN-", size=(12, 1)),
                sg.Button("保存统计", key="-SAVE-SUM-", disabled=True),
                sg.Button("保存明细", key="-SAVE-DETAIL-", disabled=True),
                sg.Button("保存时间轴", key="-SAVE-FIG-", disabled=True),
                sg.Button("退出", key="-EXIT-"),
            ],
        ],
        expand_x=True,
    )

    summary_table = sg.Table(
        values=[],
        headings=["鸟种", "叫声次数", "平均置信度"],
        key="-SUMMARY-",
        auto_size_columns=False,
        col_widths=[22, 12, 14],
        justification="left",
        num_rows=10,
        enable_events=False,
        expand_x=True,
        expand_y=True,
        alternating_row_color="#1f2937",
    )

    detail_table = sg.Table(
        values=[],
        headings=["开始(秒)", "结束(秒)", "鸟种", "置信度"],
        key="-DETAILS-",
        auto_size_columns=False,
        col_widths=[14, 14, 24, 12],
        justification="left",
        num_rows=16,
        enable_events=False,
        expand_x=True,
        expand_y=True,
        alternating_row_color="#1f2937",
    )

    timeline_image = sg.Image(
        key="-TIMELINE-",
        background_color=CUSTOM_THEME["BACKGROUND"],
        expand_x=True,
        expand_y=True,
        pad=(0, 0),
    )

    timeline_column = sg.Column(
        [[timeline_image]],
        key="-TIMELINE-COL-",
        scrollable=True,
        vertical_scroll_only=False,
        size=(1200, 340),
        expand_x=True,
        expand_y=True,
        pad=(0, 0),
        background_color=CUSTOM_THEME["BACKGROUND"],
    )

    timeline_frame = sg.Frame(
        "叫声时间轴",
        [[timeline_column]],
        expand_x=True,
        expand_y=True,
        pad=((0, 0), (10, 0)),
    )

    return [
        [control_frame],
        [
            sg.Frame(
                "统计信息",
                [[summary_table]],
                expand_x=True,
                expand_y=True,
                size=(640, 300),
            ),
            sg.Frame(
                "事件明细",
                [[detail_table]],
                expand_x=True,
                expand_y=True,
                size=(640, 300),
            ),
        ],
        [timeline_frame],
        [sg.StatusBar("准备就绪", key="-STATUS-")],
    ]


def make_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        events_df.groupby("label")
        .agg(叫声次数=("label", "size"), 平均置信度=("confidence", "mean"))
        .reset_index()
        .sort_values("叫声次数", ascending=False)
    )
    return summary_df


def format_table_data(
    df: pd.DataFrame, columns: Dict[str, Tuple[str, str]]
) -> Tuple[list, list]:
    df = df.copy()
    ordered_cols = list(columns.keys())
    df = df[ordered_cols]
    for col, (fmt, _) in columns.items():
        if fmt == "int":
            df[col] = df[col].astype(int)
        elif fmt == "float":
            df[col] = df[col].map(lambda x: f"{x:.2f}")
        else:
            df[col] = df[col].astype(str)
    data = df.values.tolist()
    headings = [columns[col][1] for col in ordered_cols]
    return data, headings


def build_timeline_image(events_df: pd.DataFrame) -> bytes:
    """Create a timeline visualization for detected events."""

    timeline_df = events_df.sort_values(["start_sec", "label"]).reset_index(drop=True)
    unique_labels = list(dict.fromkeys(timeline_df["label"].tolist()))
    label_to_y = {label: idx for idx, label in enumerate(unique_labels)}

    timeline_df = timeline_df.assign(
        y=timeline_df["label"].map(label_to_y),
        mid_sec=(timeline_df["start_sec"] + timeline_df["end_sec"]) / 2,
    )

    min_sec = float(timeline_df["start_sec"].min())
    max_sec = float(timeline_df["end_sec"].max())
    duration = max(max_sec - min_sec, 1.0)
    fig_width = max(12.0, duration / 2.5)

    fig, ax = plt.subplots(figsize=(fig_width, 4.6), facecolor=CUSTOM_THEME["BACKGROUND"])
    ax.set_facecolor("#111827")

    for _, row in timeline_df.iterrows():
        ax.hlines(
            y=row["y"],
            xmin=row["start_sec"],
            xmax=row["end_sec"],
            colors="#38bdf8",
            linewidth=6,
            alpha=0.6,
        )

    scatter = ax.scatter(
        timeline_df["mid_sec"],
        timeline_df["y"],
        c=timeline_df["confidence"],
        cmap="viridis",
        s=120,
        edgecolors="#0f172a",
        linewidths=0.6,
    )

    if max_sec == min_sec:
        ax.set_xlim(min_sec - 0.5, max_sec + 0.5)
    else:
        ax.set_xlim(min_sec, max_sec)

    ax.set_ylim(-0.6, max(len(unique_labels) - 0.4, 0.6))
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels, color=CUSTOM_THEME["TEXT"])
    ax.set_xlabel("时间 (秒)", color=CUSTOM_THEME["TEXT"], labelpad=8)
    ax.set_ylabel("鸟种", color=CUSTOM_THEME["TEXT"], labelpad=8)
    ax.tick_params(axis="x", colors=CUSTOM_THEME["TEXT"])
    ax.tick_params(axis="y", colors=CUSTOM_THEME["TEXT"])
    ax.grid(True, axis="x", linestyle="--", alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("置信度", color=CUSTOM_THEME["TEXT"])
    cbar.ax.yaxis.set_tick_params(color=CUSTOM_THEME["TEXT"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=CUSTOM_THEME["TEXT"])

    ax.set_title("鸟类叫声时间轴", color=CUSTOM_THEME["TEXT"], fontsize=14, pad=12)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor=CUSTOM_THEME["BACKGROUND"],
    )
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def main() -> None:
    default_model = Path(DEFAULT_MODEL_PATH).as_posix()
    default_processed_root = Path(DEFAULT_PROCESSED_ROOT).as_posix()

    window = sg.Window(
        "鸟类叫声识别桌面版",
        build_layout(default_model, default_processed_root),
        resizable=True,
        size=(1380, 920),
        margins=(24, 20),
        element_padding=(10, 10),
        finalize=True,
    )

    window["-SUMMARY-"].expand(True, True, True)
    window["-DETAILS-"].expand(True, True, True)
    window["-TIMELINE-"].expand(True, True)
    window["-TIMELINE-COL-"].expand(True, True, True)

    events_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None
    timeline_image: bytes | None = None

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "-EXIT-"):
            break

        if event == "-RUN-":
            audio_path = values.get("-AUDIO-")
            if not audio_path:
                window["-STATUS-"].update("请选择音频文件。")
                continue

            model_path = values.get("-MODEL-", default_model)
            processed_root = values.get("-LABELS-", default_processed_root)
            threshold = float(values.get("-THRESH-", 0.0))

            try:
                model, idx2label = _load_detector(model_path, processed_root)
                events = predict_audio(audio_path, model=model, idx2label=idx2label)
            except FileNotFoundError as exc:
                window["-STATUS-"].update(str(exc))
                continue
            except Exception as exc:  # pylint: disable=broad-except
                window["-STATUS-"].update(f"识别失败: {exc}")
                continue

            events_df = pd.DataFrame(events)
            if events_df.empty:
                window["-STATUS-"].update("未检测到鸟类事件。")
                window["-DETAILS-"].update(values=[], visible=True)
                window["-SUMMARY-"].update(values=[])
                window["-SAVE-SUM-"].update(disabled=True)
                window["-SAVE-DETAIL-"].update(disabled=True)
                window["-SAVE-FIG-"].update(disabled=True)
                window["-TIMELINE-"].update(data=None)
                timeline_image = None
                continue

            events_df.insert(0, "audio_file", Path(audio_path).name)
            if threshold > 0:
                events_df = events_df[events_df["confidence"] >= threshold]

            if events_df.empty:
                window["-STATUS-"].update("过滤后没有满足阈值的事件。")
                window["-DETAILS-"].update(values=[])
                window["-SUMMARY-"].update(values=[])
                window["-SAVE-SUM-"].update(disabled=True)
                window["-SAVE-DETAIL-"].update(disabled=True)
                window["-SAVE-FIG-"].update(disabled=True)
                window["-TIMELINE-"].update(data=None)
                timeline_image = None
                continue

            summary_df = make_summary(events_df)
            detail_columns = {
                "start_sec": ("float", "开始(秒)"),
                "end_sec": ("float", "结束(秒)"),
                "label": ("str", "鸟种"),
                "confidence": ("float", "置信度"),
            }
            summary_columns = {
                "label": ("str", "鸟种"),
                "叫声次数": ("int", "叫声次数"),
                "平均置信度": ("float", "平均置信度"),
            }

            detail_data, _ = format_table_data(events_df, detail_columns)
            summary_data, _ = format_table_data(summary_df, summary_columns)

            window["-DETAILS-"].update(values=detail_data)
            window["-SUMMARY-"].update(values=summary_data)
            window["-SAVE-SUM-"].update(disabled=False)
            window["-SAVE-DETAIL-"].update(disabled=False)
            try:
                timeline_image = build_timeline_image(events_df)
                window["-TIMELINE-"].update(data=timeline_image)
                window["-SAVE-FIG-"].update(disabled=False)
            except Exception as exc:  # pylint: disable=broad-except
                timeline_image = None
                window["-TIMELINE-"].update(data=None)
                window["-SAVE-FIG-"].update(disabled=True)
                window["-STATUS-"].update(f"识别完成，但时间轴生成失败: {exc}")
            else:
                window["-STATUS-"].update("识别完成。")

        if event == "-SAVE-SUM-" and summary_df is not None and not summary_df.empty:
            file_path = sg.popup_get_file(
                "保存统计为...",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV 文件", "*.csv"),),
            )
            if file_path:
                summary_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                window["-STATUS-"].update(f"统计信息已保存到 {file_path}")

        if event == "-SAVE-DETAIL-" and events_df is not None and not events_df.empty:
            file_path = sg.popup_get_file(
                "保存事件明细为...",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV 文件", "*.csv"),),
            )
            if file_path:
                events_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                window["-STATUS-"].update(f"事件明细已保存到 {file_path}")

        if event == "-SAVE-FIG-" and timeline_image:
            file_path = sg.popup_get_file(
                "保存时间轴为...",
                save_as=True,
                default_extension=".png",
                file_types=(("PNG 图片", "*.png"),),
            )
            if file_path:
                with open(file_path, "wb") as file:
                    file.write(timeline_image)
                window["-STATUS-"].update(f"时间轴图片已保存到 {file_path}")

    window.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        sg.popup_error(f"程序发生错误: {exc}")
        sys.exit(1)
