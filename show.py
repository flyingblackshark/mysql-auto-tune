import streamlit as st
import pandas as pd
import numpy as np
from pipeline import tune_pipeline
st.title('MySQL Auto Tuning System')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv("./res_all.csv", nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
data = load_data(200)
history_data = data.loc[::-1].reset_index(drop=True)
st.subheader("历史调优数据")
metric_chart_options = ["latency","read","write"]
metric_chart_select = st.multiselect("选择查看的指标",metric_chart_options)

chart_data = pd.DataFrame(
    history_data[['latency','read','write']],
    columns=metric_chart_select)
st.line_chart(chart_data)



best_knobs = pd.DataFrame(data.iloc[0,1:-1])
st.subheader("上次找到的最优knobs及其指标数值")
chart_data = st.dataframe(
    best_knobs,
    )
st.subheader("选择调优指标")
metric_options = ["Average_Latency", "Read", "Write"]

selected_metric = st.selectbox("选择指标", metric_options)

st.subheader("选择负载类型")
load_options = ["Read_Write", "Read_Only", "Write_Only"]

selected_load = st.selectbox("选择负载", load_options)
if st.button("开始调优"):
    if selected_metric == "Average_Latency":
        target_metric_name="latency"
    elif selected_metric == "Read":
        target_metric_name="read"
    elif selected_metric == "Write":
        target_metric_name="write"

    if selected_load == "Read_Write":
        loadtype="read_write"
        wltype="read_write"
    elif selected_metric == "Read_Only":
        loadtype="read_only"
        wltype="read_only"
    elif selected_metric == "Write_Only":
        loadtype="write_only"
        wltype="write_only"

    st.write("启动调优")
    tune_pipeline()