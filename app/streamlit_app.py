"""Streamlit 应用：可视化鸟类叫声识别结果。"""

import os
import tempfile
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

from train.test_CNN_LSTM import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_ROOT,
    load_label_maps,
    load_model,
    predict_audio,
)


st.set_page_config(page_title="鸟类叫声识别", layout="wide")


@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, processed_root: str):
    """加载模型与标签映射，并缓存结果。"""
    label_names, idx2label = load_label_maps(processed_root)
    model = load_model(model_path, num_classes=len(label_names))
    return model, idx2label


def build_timeline_chart(events_df: pd.DataFrame) -> alt.Chart:
    """构建 Altair 时间轴图表。"""
    base = alt.Chart(events_df).encode(
        x=alt.X("start_sec", title="开始时间 (秒)"),
        x2="end_sec",
        y=alt.Y("label", title="鸟种"),
        color=alt.Color("label", legend=None),
        tooltip=["audio_file", "label", "start_sec", "end_sec", "confidence"],
    )
    return base.mark_bar(size=8, opacity=0.7)


def main():
    st.title("鸟类叫声识别与可视化")
    st.markdown(
        "上传音频后，系统会以滑窗方式识别鸟类叫声，返回时间段、物种与置信度，并统计各物种的叫声次数。"
    )

    with st.expander("参数设置", expanded=False):
        default_model_path = Path(DEFAULT_MODEL_PATH).as_posix()
        default_processed_root = Path(DEFAULT_PROCESSED_ROOT).as_posix()
        model_path = st.text_input("模型权重路径", default_model_path)
        processed_root = st.text_input("标签 npz 文件目录", default_processed_root)
        confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.0, 0.05)

    uploaded_file = st.file_uploader("上传音频文件", type=["wav", "mp3", "flac"])

    if uploaded_file is None:
        st.info("请先上传音频文件。")
        return

    st.audio(uploaded_file, format=uploaded_file.type)

    try:
        model, idx2label = load_detector(model_path, processed_root)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pylint: disable=broad-except
        st.exception(exc)
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        events = predict_audio(tmp_path, model=model, idx2label=idx2label)
    finally:
        os.unlink(tmp_path)

    events_df = pd.DataFrame(events)
    events_df.insert(0, "audio_file", uploaded_file.name)

    if confidence_threshold > 0:
        events_df = events_df[events_df["confidence"] >= confidence_threshold]

    if events_df.empty:
        st.warning("没有满足条件的鸟类事件。")
        return

    summary_df = (
        events_df.groupby("label")
        .agg(叫声次数=("label", "size"), 平均置信度=("confidence", "mean"))
        .reset_index()
        .sort_values("叫声次数", ascending=False)
    )

    col_chart, col_table = st.columns([2, 1], gap="large")

    with col_chart:
        timeline = build_timeline_chart(events_df)
        st.altair_chart(timeline, use_container_width=True)

    with col_table:
        st.subheader("统计信息")
        st.dataframe(summary_df, use_container_width=True)

    st.subheader("事件明细")
    st.dataframe(events_df[["start_sec", "end_sec", "label", "confidence"]], use_container_width=True)

    csv = events_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载事件明细 CSV",
        data=csv,
        file_name=f"{Path(uploaded_file.name).stem}_events.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
