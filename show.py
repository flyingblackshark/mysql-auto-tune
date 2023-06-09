from controller import load_workload, read_metric, restart_db, run_workload
import streamlit as st
import pandas as pd
import numpy as np
import rfmodel
import os
import mysql.connector
from pipeline import train_pipeline, tune_pipeline
from settings import mysql_ip, mysql_port, mysql_test_db,mysql_user,mysql_password,wl_metrics
@st.cache_data
def load_data(nrows):
    data = pd.read_csv("./res_all.csv", nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
def detect_search_settings(selected_search):
    target_search_name="random"
    if selected_search == "GridSearch":
        target_search_name="grid"
    elif selected_search == "BayesSearch":
        target_search_name="bayes"
    return target_search_name
def detect_metric_settings(selected_metric):
    target_metric_name="latency"
    if selected_metric == "Read":
        target_metric_name="read"
    elif selected_metric == "Write":
        target_metric_name="write"
    return target_metric_name
def detect_load_settings(selected_load):
    wltype="read_write"
    if selected_load == "Read_Only":
        wltype="read_only"
    elif selected_load == "Write_Only":
        wltype="write_only"
    return wltype
def baseline_test(selected_load):
    base_line_metric=[]
    wltype = detect_load_settings(selected_load)
    rres = run_workload(wltype)
    print(rres)
    if("_ERROR" in rres):
        print("run workload error")
        exit()
    metric_list=wl_metrics[wltype]
    for i,x in enumerate(metric_list):
        base_line_metric.append(read_metric(x, rres))
    return base_line_metric

if __name__ == '__main__':
    st.title('MySQL Auto Tuning System')
    history_tab, train_tab,tune_tab, test_tab,settings_tab = st.tabs(["历史数据模块","训练模块", "调优模块", "测试模块","设置模块"])
    with history_tab:
        if os.path.isfile("res_all.csv"):
            data = load_data(200)
            history_data = data[['latency','read','write']]#.loc[::-1].reset_index(drop=True)
            st.subheader("历史调优数据")
            metric_chart_options = ["latency","read","write"]
          
            if os.path.isfile("baseline.csv"):
                baseline_data = pd.read_csv("baseline.csv")
                data['baseline_latency']=[baseline_data.iloc[0,1] for i in range(199)]
                data['baseline_read']=[baseline_data.iloc[1,1] for i in range(199)]
                data['baseline_write']=[baseline_data.iloc[2,1] for i in range(199)]
                history_data = data[['latency','read','write','baseline_latency','baseline_read','baseline_write']]
                metric_chart_options = ['latency','read','write','baseline_latency','baseline_read','baseline_write']
            metric_chart_select = st.multiselect("选择查看的指标",metric_chart_options)
            chart_data = pd.DataFrame(
                history_data,
                columns=metric_chart_select)
            st.line_chart(chart_data)

            best_knobs = pd.DataFrame(data.iloc[-1,1:-1])
            st.subheader("上次找到的最优knobs及其指标数值")
            chart_data = st.dataframe(
                best_knobs,
                )
        else:
            st.subheader("暂无历史数据")
    load_options = ["Read_Write", "Read_Only", "Write_Only"]
    with train_tab:
        if st.button("开始训练模型"):
            st.subheader("选择训练方法")
            search_options = ["RandomSearch","GridSearch","BayesSearch"]
            train_selected_search = st.selectbox("选择训练方法", search_options)
            st.subheader("选择负载类型")
            train_selected_load = st.selectbox("选择训练负载", load_options)
            st.write("启动训练")
            train_pipeline(detect_load_settings(train_selected_load),detect_search_settings(train_selected_search))
    with tune_tab:
        st.subheader("选择调优指标")
        metric_options = ["Average_Latency", "Read", "Write"]
        tune_selected_metric = st.selectbox("选择指标", metric_options)
        st.subheader("选择负载类型")
        tune_selected_load = st.selectbox("选择调优负载", load_options)
       
        if st.button("开始调优"):
            st.write("启动调优")
            tune_pipeline(detect_load_settings(tune_selected_load))
        rfmodel.res_output = st.empty()
    with test_tab:
        test_selected_load = st.selectbox("选择基准测试负载", load_options)
        if st.button("进行基准性能测试"):
            base_line = baseline_test(test_selected_load)
            st.dataframe(base_line,)
            base_line = pd.DataFrame(base_line)
            base_line.to_csv("baseline.csv")
    with settings_tab:
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


  


    