from controller import load_workload, read_metric, restart_db, run_workload
import streamlit as st
import pandas as pd
import numpy as np
import os
import mysql.connector
from pipeline import tune_pipeline
from settings import mysql_ip, mysql_port, mysql_test_db,mysql_user,mysql_password,wl_metrics
st.title('MySQL Auto Tuning System')


@st.cache_data
def load_data(nrows):
    data = pd.read_csv("./res_all.csv", nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
if os.path.isfile("res_all.csv"):
    data = load_data(200)
    history_data = data#.loc[::-1].reset_index(drop=True)
    st.subheader("历史调优数据")
    metric_chart_options = ["latency","read","write"]
    metric_chart_select = st.multiselect("选择查看的指标",metric_chart_options)

    chart_data = pd.DataFrame(
        history_data[['latency','read','write']],
        columns=metric_chart_select)
    st.line_chart(chart_data)

    best_knobs = pd.DataFrame(data.iloc[-1,1:-1])
    st.subheader("上次找到的最优knobs及其指标数值")
    chart_data = st.dataframe(
        best_knobs,
        )
else:
    st.subheader("暂无历史数据")

st.subheader("选择调优指标")
metric_options = ["Average_Latency", "Read", "Write"]

selected_metric = st.selectbox("选择指标", metric_options)

st.subheader("选择负载类型")
load_options = ["Read_Write", "Read_Only", "Write_Only"]

selected_load = st.selectbox("选择负载", load_options)
def detect_metric_settings():
    target_metric_name="latency"
    if selected_metric == "Read":
        target_metric_name="read"
    elif selected_metric == "Write":
        target_metric_name="write"
    return target_metric_name
def detect_load_settings():
    wltype="read_write"
    if selected_load == "Read_Only":
        wltype="read_only"
    elif selected_load == "Write_Only":
        wltype="write_only"
    return wltype

def base_line_test():
    base_line_metric=[]
    wltype = detect_load_settings()
    rres = run_workload(wltype)
    print(rres)
    if("_ERROR" in rres):
        print("run workload error")
        exit()
    metric_list=wl_metrics[wltype]
    for i,x in enumerate(metric_list):
        base_line_metric.append(read_metric(x, rres))
    return base_line_metric
if st.button("初始化测试数据库"):
    load_workload()
if st.button("清除优化参数并重启数据库"):
    mydb = mysql.connector.connect(
    host=mysql_ip,
    user=mysql_user,
    password=mysql_password
    )
    knob_cursor = mydb.cursor()
    knob_cursor.execute("RESET PERSIST;")
    print("RESET PERSIST")
    restart_db()
if st.button("进行基准性能测试"):
    base_line = base_line_test()
    st.dataframe(
        base_line,
        )

if st.button("开始调优"):
    st.write("启动调优")
    tune_pipeline()


