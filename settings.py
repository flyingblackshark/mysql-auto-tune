
mysql_ip="127.0.0.1"
mysql_port="3306"
mysql_test_db="sbtest"
mysql_user="root"
mysql_password="fbs"
threads=20
tables_num_std = 10
table_size_std = 1000000
benchmark_interval_time=20 #seconds
wl_metrics={
    "read_only": ["latency","read","write"], 
    "read_write": ["latency","read","write"], 
    "write_only": ["latency","read","write"], 
}

# workload to be load
loadtype = "read_write" # not working  , controlled by show.py
# workload to be run
wltype = "read_write" # not working  , controlled by show.py

# only 1 target metric to be optimized
target_metric_name="latency"

# several knobs to be tuned
target_knob_set=['innodb_buffer_pool_size',
                 'innodb_log_file_size',
                  'max_connections',
                  'key_buffer_size',
                  'read_buffer_size',
                  'sort_buffer_size',
                  'join_buffer_size',
                  'innodb_flush_log_at_trx_commit',
                  'innodb_log_files_in_group',
                  'innodb_log_compressed_pages']