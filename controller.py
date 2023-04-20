import sys
import os
import psutil
import time
import numpy as np
from ruamel import yaml
import mysql.connector
import streamlit as st
from settings import mysql_ip, mysql_port, mysql_test_db,mysql_user,mysql_password
#MEM_MAX = psutil.virtual_memory().total
MEM_MAX = 0.8*32*1024*1024*1024                 # memory size of tikv node, not current PC


#------------------knob controller------------------

knob_set=\
    {
    "innodb_buffer_pool_size":
        {
         
            "set_func": None,
            "minval": 64,                           # if type==int, indicate min possible value
            "maxval": 1024,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                           # default value
        },
    "innodb_log_file_size":
        {
          
            "set_func": None,
            "minval": 64,                          # if type==int, indicate min possible value
            "maxval": 1024,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                            # default value
        },
        "key_buffer_size":
        {
          
            "set_func": None,
            "minval": 64,                          # if type==int, indicate min possible value
            "maxval": 1024,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                            # default value
        },
        "read_buffer_size":
        {
          
            "set_func": None,
            "minval": 32,                          # if type==int, indicate min possible value
            "maxval": 512,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                            # default value
        },
         "sort_buffer_size":
        {
          
            "set_func": None,
            "minval": 64,                          # if type==int, indicate min possible value
            "maxval": 1024,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                            # default value
        },
         "join_buffer_size":
        {
          
            "set_func": None,
            "minval": 64,                          # if type==int, indicate min possible value
            "maxval": 1024,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                            # default value
        },
        "max_connections":
        {
        
            "set_func": None,
            "minval": 50,                          # if type==int, indicate min possible value
            "maxval": 1000,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                            # default value
        },
    # "max_connections":
    #     {
    #         "changebyyml": True,
    #         "set_func": None,
    #         "minval": 0,                            # if type==int, indicate min possible value
    #         "maxval": 0,                            # if type==int, indicate max possible value
    #         "enumval": [32,64,128,256,512],          # if type==enum, list all valid values
    #         "type": "enum",                         # int / enum
    #         "default": 32                            # default value
    #     },
    # "innodb_log_file_size":
    #     {
    #         "changebyyml": True,
    #         "set_func": None,
    #         "minval": 0,                            # if type==int, indicate min possible value
    #         "maxval": 0,                            # if type==int, indicate max possible value
    #         "enumval": [128,256,512,1024],                # if type==enum, list all valid values
    #         "type": "enum",                         # int / enum
    #         "default": 128                            # default value
    #     },
    }


#------------------metric controller------------------

metric_set=\
    {"latency":
         {
         "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #incremental
         },
        "write":
         {
         "lessisbetter": 0,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #incremental
         },
        "read":
        {
         "lessisbetter": 0,                    # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #instant
        },
    # "get_throughput":
    #     {
    #      "read_func": read_get_throughput,
    #      "lessisbetter": 0,                   # whether less value of this metric is better(1: yes)
    #      "calc": "ins",                       #incremental
    #     },
    # "get_latency":
    #     {
    #      "read_func": read_get_latency,
    #      "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
    #      "calc": "ins",                       #instant
    #     },
    # "scan_throughput":
    #     {
    #      "read_func": read_scan_throughput,
    #      "lessisbetter": 0,                   # whether less value of this metric is better(1: yes)
    #      "calc": "ins",                       #incremental
    #     },
    # "scan_latency":
    #     {
    #      "read_func": read_scan_latency,
    #      "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
    #      "calc": "ins",                       #instant
    #     },
    # "store_size":
    #     {
    #      "read_func": read_store_size,
    #      "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
    #      "calc": "ins",                       #instant
    #     },
    # "compaction_cpu":
    #     {
    #      "read_func": read_compaction_cpu,
    #      "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
    #      "calc": "inc",                       #incremental
    #     },
    }


#------------------workload controller------------------

def run_workload(wl_type):
    tables_num_std = 10
    table_size_std = 1000000
    threads=20
    script = "/usr/share/sysbench/oltp_read_write.lua"
    if wl_type == "read_only":
        script = "/usr/share/sysbench/oltp_read_only.lua"
    elif wl_type == "write_only":
        script = "/usr/share/sysbench/oltp_write_only.lua"
    cmd="sysbench --db-driver=mysql --mysql-user="+mysql_user+" --mysql_password="+mysql_password+" --mysql-db="+mysql_test_db+" --mysql-host="+mysql_ip+" --mysql-port="+mysql_port+" --time=5 --tables="+str(tables_num_std)+" --table-size="+str(table_size_std)+" --threads="+str(threads)+" "+script+" run"
    print(cmd)
    res=os.popen(cmd).read()
    return(res)

def load_workload():
    tables_num_std = 10
    table_size_std = 1000000
    threads=20
    mydb = mysql.connector.connect(
    host=mysql_ip,
    user=mysql_user,
    password=mysql_password
    )
    knob_cursor = mydb.cursor()
    knob_cursor.execute("DROP DATABASE sbtest;")
    print("DROP DATABASE sbtest")

    cmd="sysbench --db-driver=mysql --mysql-user="+mysql_user+" --mysql_password="+mysql_password+" --mysql-db="+mysql_test_db+" --mysql-host="+mysql_ip+" --mysql-port="+mysql_port+" --tables="+str(tables_num_std)+" --table-size="+str(table_size_std)+" --threads="+str(threads)+" /usr/share/sysbench/oltp_read_write.lua prepare"
    print(cmd)
    res=os.popen(cmd).read()
    return(res)

#------------------common functions------------------

def set_mysql_knob(knob_sessname, knob_val):
    mydb = mysql.connector.connect(
    host=mysql_ip,
    user=mysql_user,
    password=mysql_password
    )
    knob_sess=knob_sessname.split('.')[0:-1]
    knob_name=knob_sessname.split('.')[-1]

    if(knob_set[knob_sessname]['type']=='enum'):
        idx=knob_val
        knob_val=knob_set[knob_sessname]['enumval'][idx]
    if(knob_set[knob_sessname]['type']=='bool'):
        if(knob_val==0):
            knob_val=False
        else:
            knob_val=True
    if(knob_name=='key_buffer_size' or knob_name=='read_buffer_size' or knob_name=='sort_buffer_size' or knob_name=='join_buffer_size'):
        knob_val=knob_val*1024
    if(knob_name=='innodb_buffer_pool_size' or knob_name=='innodb_log_file_size'):
        knob_val=knob_val*1024*1024
    knob_cursor = mydb.cursor()
    knob_cursor.execute("SET PERSIST_ONLY "+knob_sessname+" = "+str(knob_val)+";")
    print("set_mysql:: ",knob_sessname, knob_sess, knob_name, knob_val)

    time.sleep(0.5)
    return('success')

def set_knob(knob_name, knob_val):
    res=set_mysql_knob(knob_name, knob_val)
    return res

def read_knob(knob_name, knob_cache):
    res=knob_cache[knob_name]
    return res

def read_metric(metric_name, rres=None):
    if(rres!=None):
        rl=rres.split('\n')
        rl.reverse()
        if(metric_name=="latency"):
            i=0
            while((not rl[i].strip().startswith('avg:'))):
                 i+=1
            dat = rl[i].strip()[4:]
            dat=float(dat)
            return(dat)
        elif(metric_name=="write"):
            i=0
            while((not rl[i].strip().startswith('write:'))):
                 i+=1
            dat = rl[i].strip()[6:]
            dat=float(dat)
            return(dat)
        elif(metric_name=="read"):
            i=0
            while((not rl[i].strip().startswith('read:'))):
                 i+=1
            dat = rl[i].strip()[5:]
            dat=float(dat)
            return(dat)
        # elif(metric_name=="write_throughput"):
        #     i=0
        #     while((not rl[i].startswith('UPDATE ')) and (not rl[i].startswith('INSERT '))):
        #         i+=1
        #     dat=rl[i][rl[i].find("OPS:") + 5:].split(",")[0]
        #     dat=float(dat)
        #     return(dat)
        # elif(metric_name=="get_throughput"):
        #     i=0
        #     while(not rl[i].startswith('READ ')):
        #         i+=1
        #     dat=rl[i][rl[i].find("OPS:") + 5:].split(",")[0]
        #     dat=float(dat)
        #     return(dat)
        # elif(metric_name=="scan_throughput"):
        #     i=0
        #     while(not rl[i].startswith('SCAN ')):
        #         i+=1
        #     dat=rl[i][rl[i].find("OPS:") + 5:].split(",")[0]
        #     dat=float(dat)
        #     return(dat)
    return 0

def init_knobs():
    # if there are knobs whose range is related to PC memory size, initialize them here
    pass

def calc_metric(metric_after, metric_before, metric_list):
    num_metrics = len(metric_list)
    new_metric = np.zeros([1, num_metrics])
    for i, x in enumerate(metric_list):
        if(metric_set[x]["calc"]=="inc"):
            new_metric[0][i]=metric_after[0][i]-metric_before[0][i]
        elif(metric_set[x]["calc"]=="ins"):
            new_metric[0][i]=metric_after[0][i]
    return(new_metric)

def restart_db():
    os.popen("sudo systemctl restart mysql")
    while(1):
        time.sleep(10)
        clrres = os.popen("sudo systemctl status mysql").read()
        if("active (running)" in clrres):
            print("unsafe_cleanup_data finished, res == "+clrres.split('\n')[-2])
            break
        else:
            print("unsafe_cleanup_data failed")
            continue


