"""基于 PySimpleGUI 的桌面端鸟类叫声识别应用。"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import PySimpleGUI as sg

from train.test_CNN_LSTM import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_ROOT,
    load_label_maps,
    load_model,
    predict_audio,
)

sg.theme("SystemDefaultForReal")


@lru_cache(maxsize=2)
def _load_detector(model_path: str, processed_root: str):
    label_names, idx2label = load_label_maps(processed_root)
    model = load_model(model_path, num_classes=len(label_names))
    return model, idx2label


def build_layout(default_model: str, default_processed_root: str) -> list:
    return [
        [sg.Text("模型权重"), sg.Input(default_model, key="-MODEL-", size=(45, 1)), sg.FileBrowse("浏览", file_types=(("PyTorch 模型", "*.pth"),))],
        [sg.Text("标签目录"), sg.Input(default_processed_root, key="-LABELS-", size=(45, 1)), sg.FolderBrowse("浏览")],
        [sg.Text("音频文件"), sg.Input(key="-AUDIO-", size=(45, 1)), sg.FileBrowse("浏览", file_types=(("音频文件", "*.wav;*.mp3;*.flac"),))],
        [
            sg.Text("置信度阈值"),
            sg.Slider(
                range=(0.0, 1.0),
                orientation="h",
                resolution=0.05,
                default_value=0.0,
                size=(30, 15),
                key="-THRESH-",
            ),
        ],
        [
            sg.Button("识别", key="-RUN-", size=(10, 1)),
            sg.Button("保存统计", key="-SAVE-SUM-", disabled=True),
            sg.Button("保存明细", key="-SAVE-DETAIL-", disabled=True),
            sg.Button("退出", key="-EXIT-")
        ],
        [
            sg.Frame(
                "统计信息",
                [[
                    sg.Table(
                        values=[],
                        headings=["鸟种", "叫声次数", "平均置信度"],
                        key="-SUMMARY-",
                        auto_size_columns=True,
                        justification="left",
                        num_rows=6,
                        enable_events=False,
                    )
                ]],
                expand_x=True,
                expand_y=True,
            )
        ],
        [
            sg.Frame(
                "事件明细",
                [[
                    sg.Table(
                        values=[],
                        headings=["开始(秒)", "结束(秒)", "鸟种", "置信度"],
                        key="-DETAILS-",
                        auto_size_columns=True,
                        justification="left",
                        num_rows=12,
                        enable_events=False,
                    )
                ]],
                expand_x=True,
                expand_y=True,
            )
        ],
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


def main() -> None:
    default_model = Path(DEFAULT_MODEL_PATH).as_posix()
    default_processed_root = Path(DEFAULT_PROCESSED_ROOT).as_posix()

    window = sg.Window(
        "鸟类叫声识别桌面版",
        build_layout(default_model, default_processed_root),
        resizable=True,
    )

    events_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None

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

            detail_data, detail_headings = format_table_data(events_df, detail_columns)
            summary_data, summary_headings = format_table_data(summary_df, summary_columns)

            window["-DETAILS-"].update(values=detail_data, headings=detail_headings)
            window["-SUMMARY-"].update(values=summary_data, headings=summary_headings)
            window["-SAVE-SUM-"].update(disabled=False)
            window["-SAVE-DETAIL-"].update(disabled=False)
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

    window.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        sg.popup_error(f"程序发生错误: {exc}")
        sys.exit(1)
