"""基于 PySimpleGUI 的鸟类叫声识别应用（含 DBSCAN 时序聚合）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import PySimpleGUI as sg
from sklearn.cluster import DBSCAN

from desktop_app.gui_app import (
    CUSTOM_THEME,
    _load_detector,
    build_timeline_image,
    format_table_data,
    update_timeline_canvas,
)
from train.test_CNN_LSTM import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_ROOT,
    predict_audio,
)


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
        ],
        expand_x=True,
    )

    aggregation_frame = sg.Frame(
        "时序聚合",
        [
            [
                sg.Text("聚合窗口(秒)", size=(12, 1)),
                sg.Slider(
                    range=(0.1, 5.0),
                    orientation="h",
                    resolution=0.1,
                    default_value=1.5,
                    expand_x=True,
                    key="-EPS-",
                ),
            ],
            [
                sg.Text("最小事件数", size=(12, 1)),
                sg.Spin(
                    [1, 2, 3, 4, 5],
                    initial_value=1,
                    key="-MIN-SAMPLES-",
                    size=(6, 1),
                ),
            ],
            [
                sg.Button("识别", key="-RUN-", size=(12, 1)),
                sg.Button("应用聚合", key="-APPLY-AGG-", size=(12, 1), disabled=True),
                sg.Button("保存统计", key="-SAVE-SUM-", disabled=True),
                sg.Button("保存聚合", key="-SAVE-DETAIL-", disabled=True),
                sg.Button("保存原始", key="-SAVE-RAW-DETAIL-", disabled=True),
                sg.Button("保存时间轴", key="-SAVE-FIG-", disabled=True),
                sg.Button("退出", key="-EXIT-"),
            ],
        ],
        expand_x=True,
    )

    summary_table = sg.Table(
        values=[],
        headings=["鸟种", "聚合段数", "总事件数", "平均置信度", "覆盖时长(秒)"],
        key="-SUMMARY-",
        auto_size_columns=False,
        col_widths=[20, 12, 12, 16, 16],
        justification="left",
        num_rows=10,
        enable_events=False,
        expand_x=True,
        expand_y=True,
        alternating_row_color="#1f2937",
    )

    aggregated_table = sg.Table(
        values=[],
        headings=["开始(秒)", "结束(秒)", "鸟种", "平均置信度", "事件数", "覆盖时长(秒)"],
        key="-AGG-DETAIL-",
        auto_size_columns=False,
        col_widths=[12, 12, 20, 14, 10, 16],
        justification="left",
        num_rows=12,
        enable_events=False,
        expand_x=True,
        expand_y=True,
        alternating_row_color="#1f2937",
    )

    raw_table = sg.Table(
        values=[],
        headings=["开始(秒)", "结束(秒)", "鸟种", "置信度"],
        key="-RAW-DETAIL-",
        auto_size_columns=False,
        col_widths=[14, 14, 24, 12],
        justification="left",
        num_rows=14,
        enable_events=False,
        expand_x=True,
        expand_y=True,
        alternating_row_color="#1f2937",
    )

    timeline_canvas = sg.Canvas(
        key="-TIMELINE-CANVAS-",
        background_color=CUSTOM_THEME["BACKGROUND"],
        size=(1200, 400),
        expand_x=True,
        expand_y=True,
    )

    timeline_frame = sg.Frame(
        "聚合后叫声时间轴（可滚动查看）",
        [[timeline_canvas]],
        expand_x=True,
        expand_y=True,
        pad=((0, 0), (10, 0)),
    )

    return [
        [control_frame, aggregation_frame],
        [
            sg.Frame(
                "聚合统计",
                [[summary_table]],
                expand_x=True,
                expand_y=True,
                size=(600, 280),
            ),
            sg.Frame(
                "聚合事件明细",
                [[aggregated_table]],
                expand_x=True,
                expand_y=True,
                size=(680, 280),
            ),
        ],
        [
            sg.Frame(
                "原始事件明细",
                [[raw_table]],
                expand_x=True,
                expand_y=True,
                size=(1280, 260),
            )
        ],
        [timeline_frame],
        [sg.StatusBar("准备就绪", key="-STATUS-")],
    ]


def aggregate_events_dbscan(
    events_df: pd.DataFrame, eps: float, min_samples: int
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "audio_file",
                "label",
                "start_sec",
                "end_sec",
                "confidence",
                "event_count",
                "duration",
            ]
        )

    aggregated_rows = []

    for label, group in events_df.groupby("label"):
        group = group.sort_values("start_sec").reset_index(drop=True)
        midpoints = ((group["start_sec"] + group["end_sec"]) / 2).to_numpy().reshape(-1, 1)

        try:
            clustering = DBSCAN(eps=max(eps, 0.05), min_samples=max(min_samples, 1)).fit(midpoints)
            cluster_labels = clustering.labels_
        except Exception:  # pylint: disable=broad-except
            cluster_labels = [-1] * len(group)

        group = group.assign(_cluster=cluster_labels)

        for cluster_id, cluster_df in group.groupby("_cluster"):
            if cluster_id == -1:
                for _, row in cluster_df.iterrows():
                    aggregated_rows.append(
                        {
                            "audio_file": row.get("audio_file", ""),
                            "label": row["label"],
                            "start_sec": float(row["start_sec"]),
                            "end_sec": float(row["end_sec"]),
                            "confidence": float(row["confidence"]),
                            "event_count": 1,
                            "duration": float(row["end_sec"] - row["start_sec"]),
                        }
                    )
                continue

            start_sec = float(cluster_df["start_sec"].min())
            end_sec = float(cluster_df["end_sec"].max())
            confidence = float(cluster_df["confidence"].mean())
            duration = max(end_sec - start_sec, 0.0)

            aggregated_rows.append(
                {
                    "audio_file": cluster_df["audio_file"].iloc[0] if "audio_file" in cluster_df else "",
                    "label": cluster_df["label"].iloc[0],
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "confidence": confidence,
                    "event_count": int(len(cluster_df)),
                    "duration": duration,
                }
            )

    aggregated_df = pd.DataFrame(aggregated_rows)
    if aggregated_df.empty:
        return aggregated_df

    aggregated_df = aggregated_df.sort_values(["start_sec", "label"]).reset_index(drop=True)
    return aggregated_df


def make_aggregated_summary(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    if aggregated_df.empty:
        return pd.DataFrame(columns=["label", "聚合段数", "总事件数", "平均置信度", "覆盖时长"])

    summary_df = (
        aggregated_df.groupby("label")
        .agg(
            聚合段数=("label", "size"),
            总事件数=("event_count", "sum"),
            平均置信度=("confidence", "mean"),
            覆盖时长=("duration", "sum"),
        )
        .reset_index()
        .sort_values("总事件数", ascending=False)
    )
    return summary_df


def update_tables_and_timeline(
    window: sg.Window,
    aggregated_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, bytes | None]:
    summary_df = make_aggregated_summary(aggregated_df)

    summary_columns: Dict[str, Tuple[str, str]] = {
        "label": ("str", "鸟种"),
        "聚合段数": ("int", "聚合段数"),
        "总事件数": ("int", "总事件数"),
        "平均置信度": ("float", "平均置信度"),
        "覆盖时长": ("float", "覆盖时长(秒)"),
    }
    aggregated_columns: Dict[str, Tuple[str, str]] = {
        "start_sec": ("float", "开始(秒)"),
        "end_sec": ("float", "结束(秒)"),
        "label": ("str", "鸟种"),
        "confidence": ("float", "平均置信度"),
        "event_count": ("int", "事件数"),
        "duration": ("float", "覆盖时长(秒)"),
    }
    raw_columns: Dict[str, Tuple[str, str]] = {
        "start_sec": ("float", "开始(秒)"),
        "end_sec": ("float", "结束(秒)"),
        "label": ("str", "鸟种"),
        "confidence": ("float", "置信度"),
    }

    summary_data, _ = format_table_data(summary_df, summary_columns)
    aggregated_data, _ = format_table_data(aggregated_df, aggregated_columns)
    raw_data, _ = format_table_data(raw_df, raw_columns)

    window["-SUMMARY-"].update(values=summary_data)
    window["-AGG-DETAIL-"].update(values=aggregated_data)
    window["-RAW-DETAIL-"].update(values=raw_data)

    timeline_image = None
    if not aggregated_df.empty:
        timeline_image = build_timeline_image(aggregated_df)
        update_timeline_canvas(window, timeline_image)
    else:
        window["-TIMELINE-CANVAS-"].update(data=None)

    return summary_df, timeline_image


def main() -> None:
    default_model = Path(DEFAULT_MODEL_PATH).as_posix()
    default_processed_root = Path(DEFAULT_PROCESSED_ROOT).as_posix()

    window = sg.Window(
        "鸟类叫声识别（含时序聚合）",
        build_layout(default_model, default_processed_root),
        resizable=True,
        size=(1700, 1000),
        margins=(24, 20),
        element_padding=(10, 10),
        finalize=True,
    )

    window["-SUMMARY-"].expand(True, True, True)
    window["-AGG-DETAIL-"].expand(True, True, True)
    window["-RAW-DETAIL-"].expand(True, True, True)
    window["-TIMELINE-CANVAS-"].expand(True, True)

    raw_events_df: pd.DataFrame | None = None
    aggregated_df: pd.DataFrame | None = None
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

            model_path = values.get("-MODEL-", "")
            processed_root = values.get("-LABELS-", "")
            threshold = float(values.get("-THRESH-", 0.0))
            eps = float(values.get("-EPS-", 1.5))
            min_samples = int(values.get("-MIN-SAMPLES-", 1))

            try:
                model, idx2label = _load_detector(model_path, processed_root)
                events = predict_audio(audio_path, model=model, idx2label=idx2label)
            except FileNotFoundError as exc:
                window["-STATUS-"].update(str(exc))
                continue
            except Exception as exc:  # pylint: disable=broad-except
                window["-STATUS-"].update(f"识别失败: {exc}")
                continue

            raw_events_df = pd.DataFrame(events)
            if raw_events_df.empty:
                window["-STATUS-"].update("未检测到鸟类事件。")
                window["-SUMMARY-"].update(values=[])
                window["-AGG-DETAIL-"].update(values=[])
                window["-RAW-DETAIL-"].update(values=[])
                window["-SAVE-SUM-"].update(disabled=True)
                window["-SAVE-DETAIL-"].update(disabled=True)
                window["-SAVE-RAW-DETAIL-"].update(disabled=True)
                window["-SAVE-FIG-"].update(disabled=True)
                window["-APPLY-AGG-"].update(disabled=True)
                window["-TIMELINE-CANVAS-"].update(data=None)
                timeline_image = None
                continue

            raw_events_df.insert(0, "audio_file", Path(audio_path).name)
            if threshold > 0:
                raw_events_df = raw_events_df[raw_events_df["confidence"] >= threshold]

            if raw_events_df.empty:
                window["-STATUS-"].update("过滤后没有满足阈值的事件。")
                window["-SUMMARY-"].update(values=[])
                window["-AGG-DETAIL-"].update(values=[])
                window["-RAW-DETAIL-"].update(values=[])
                window["-SAVE-SUM-"].update(disabled=True)
                window["-SAVE-DETAIL-"].update(disabled=True)
                window["-SAVE-RAW-DETAIL-"].update(disabled=True)
                window["-SAVE-FIG-"].update(disabled=True)
                window["-APPLY-AGG-"].update(disabled=True)
                window["-TIMELINE-CANVAS-"].update(data=None)
                timeline_image = None
                continue

            aggregated_df = aggregate_events_dbscan(raw_events_df, eps, min_samples)
            summary_df, timeline_image = update_tables_and_timeline(window, aggregated_df, raw_events_df)

            window["-SAVE-SUM-"].update(disabled=summary_df.empty)
            window["-SAVE-DETAIL-"].update(disabled=aggregated_df.empty)
            window["-SAVE-RAW-DETAIL-"].update(disabled=raw_events_df.empty)
            window["-SAVE-FIG-"].update(disabled=timeline_image is None)
            window["-APPLY-AGG-"].update(disabled=False)
            window["-STATUS-"].update("识别并聚合完成。")

        if event == "-APPLY-AGG-" and raw_events_df is not None and not raw_events_df.empty:
            eps = float(values.get("-EPS-", 1.5))
            min_samples = int(values.get("-MIN-SAMPLES-", 1))
            aggregated_df = aggregate_events_dbscan(raw_events_df, eps, min_samples)
            summary_df, timeline_image = update_tables_and_timeline(window, aggregated_df, raw_events_df)

            window["-SAVE-SUM-"].update(disabled=summary_df.empty)
            window["-SAVE-DETAIL-"].update(disabled=aggregated_df.empty)
            window["-SAVE-FIG-"].update(disabled=timeline_image is None)
            window["-STATUS-"].update("聚合参数已更新。")

        if event == "-SAVE-SUM-" and summary_df is not None and not summary_df.empty:
            file_path = sg.popup_get_file(
                "保存统计为...",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV 文件", "*.csv"),),
            )
            if file_path:
                summary_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                window["-STATUS-"].update(f"聚合统计已保存到 {file_path}")

        if event == "-SAVE-DETAIL-" and aggregated_df is not None and not aggregated_df.empty:
            file_path = sg.popup_get_file(
                "保存聚合明细为...",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV 文件", "*.csv"),),
            )
            if file_path:
                aggregated_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                window["-STATUS-"].update(f"聚合事件已保存到 {file_path}")

        if event == "-SAVE-RAW-DETAIL-" and raw_events_df is not None and not raw_events_df.empty:
            file_path = sg.popup_get_file(
                "保存原始明细为...",
                save_as=True,
                default_extension=".csv",
                file_types=(("CSV 文件", "*.csv"),),
            )
            if file_path:
                raw_events_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                window["-STATUS-"].update(f"原始事件已保存到 {file_path}")

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
