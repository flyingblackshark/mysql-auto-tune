from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from controller import knob_set, MEM_MAX
import random
import queue
import streamlit as st

TOP_NUM_CONFIG = 10
NUM_SAMPLES = 30
GPR_EPS = 0.001
def gen_random_data(target_data):
    random_knob_result = {}
    for name in target_data.knob_labels:
        vartype = knob_set[name]['type']
        # if vartype == 'bool':
        #     flag = random.randint(0, 1)
        #     if flag == 0:
        #         random_knob_result[name] = False
        #     else:
        #         random_knob_result[name] = True
        if (vartype == 'enum' or vartype == 'bool'):
            enumvals = knob_set[name]['enumval']
            enumvals_len = len(enumvals)
            rand_idx = random.randint(0, enumvals_len - 1)
            #random_knob_result[name] = knob_set[name]['enumval'][rand_idx]
            random_knob_result[name] = rand_idx
        elif vartype == 'int':
            minval=knob_set[name]['minval']
            maxval=knob_set[name]['maxval']
            random_knob_result[name] = random.randint(int(minval), int(maxval))
        elif vartype == 'real':
            minval=knob_set[name]['minval']
            maxval=knob_set[name]['maxval']
            random_knob_result[name] = random.uniform(float(minval), float(maxval))
        # elif vartype == STRING:
        #     random_knob_result[name] = "None"
        # elif vartype == TIMESTAMP:
        #     random_knob_result[name] = "None"
    return random_knob_result
def configuration_recommendation(target_data, runrec=None):
    from show import res_output
    print("running configuration recommendation...")
    if(target_data.num_previousamples<10 and runrec==None):                               #  give random recommendation on several rounds at first
        res_output.clear()
        res_output.write("正在进行第"+str(target_data.num_previousamples+1)+"轮随机knobs训练")
        return gen_random_data(target_data)

    X_workload = target_data.new_knob_set
    X_columnlabels = target_data.knob_labels
    y_workload = target_data.new_metric_set
    y_columnlabels = target_data.metric_labels
    rowlabels_workload = target_data.new_rowlabels

    X_target = target_data.previous_knob_set
    y_target = target_data.previous_metric_set
    rowlabels_target = target_data.previous_rowlabels

    # Filter ys by current target objective metric
    target_objective = target_data.target_metric
    target_obj_idx = [i for i, cl in enumerate(y_columnlabels) if cl == target_objective]   #idx of target metric in y_columnlabels matrix

    lessisbetter = target_data.target_lessisbetter==1

    y_workload = y_workload[:, target_obj_idx]
    y_target = y_target[:, target_obj_idx]
    y_columnlabels = y_columnlabels[target_obj_idx]

    X_matrix = np.vstack([X_target, X_workload])


    # Scale to N(0, 1)
    X_scaler = StandardScaler()
    X_scaler.fit(X_matrix)
    X_scaled = X_scaler.transform(X_matrix)
    #X_scaled = X_scaler.fit_transform(X_matrix)
    if y_target.shape[0] < 5:  # FIXME
        # FIXME (dva): if there are fewer than 5 target results so far
        # then scale the y values (metrics) using the workload's
        # y_scaler. I'm not sure if 5 is the right cutoff.
        y_target_scaler = None
        y_workload_scaler = StandardScaler()
        y_matrix = np.vstack([y_target, y_workload])
        y_scaled = y_workload_scaler.fit_transform(y_matrix)
    else:
        # FIXME (dva): otherwise try to compute a separate y_scaler for
        # the target and scale them separately.
        try:
            y_target_scaler = StandardScaler()
            y_workload_scaler = StandardScaler()
            y_target_scaled = y_target_scaler.fit_transform(y_target)
            y_workload_scaled = y_workload_scaler.fit_transform(y_workload)
            y_scaled = np.vstack([y_target_scaled, y_workload_scaled])
        except ValueError:
            y_target_scaler = None
            y_workload_scaler = StandardScaler()
            y_scaled = y_workload_scaler.fit_transform(y_target)


    num_samples = NUM_SAMPLES
    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    X_min = np.empty(X_scaled.shape[1])
    X_max = np.empty(X_scaled.shape[1])

    X_mem = np.zeros([1, X_scaled.shape[1]])
    X_default = np.empty(X_scaled.shape[1])

    # Get default knob values
    for i, k_name in enumerate(X_columnlabels):
        X_default[i] = knob_set[k_name]['default']

    X_default_scaled = X_scaler.transform(X_default.reshape(1, X_default.shape[0]))[0]

    # Determine min/max for knob values
    for i in range(X_scaled.shape[1]):
        # if i < total_dummies or i in binary_index_set:
        #     col_min = 0
        #     col_max = 1
        # else:
        col_min = X_scaled[:, i].min()
        col_max = X_scaled[:, i].max()
        # Set min value to the default value
        # FIXME: support multiple methods can be selected by users
        #col_min = X_default_scaled[i]

        X_min[i] = col_min
        X_max[i] = col_max
        X_samples[:, i] = np.random.rand(num_samples) * (col_max - col_min) + col_min

    # Maximize the throughput, moreisbetter
    # Use gradient descent to minimize -throughput
    if not lessisbetter:
        y_scaled = -y_scaled

    q = queue.PriorityQueue()
    for x in range(0, y_scaled.shape[0]):
        q.put((y_scaled[x][0], x))

    i = 0
    while i < TOP_NUM_CONFIG:
        try:
            item = q.get_nowait()
            # Tensorflow get broken if we use the training data points as
            # starting points for GPRGD. We add a small bias for the
            # starting points. GPR_EPS default value is 0.001
            # if the starting point is X_max, we minus a small bias to
            # make sure it is within the range.
            dist = sum(np.square(X_max - X_scaled[item[1]]))
            if dist < 0.001:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] - abs(GPR_EPS)))
            else:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] + abs(GPR_EPS)))
            i = i + 1
        except queue.Empty:
            break

    #X_samples=np.rint(X_samples)
    #X_samples=X_scaler.transform(X_samples)

    model = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 7, random_state = 18)
    model.fit(X_scaled, y_scaled) 

    print("predict:::::::: ", X_samples.shape, X_scaler.inverse_transform(X_samples).astype(np.int16), type(X_samples[0][0]))
    res = model.predict(X_samples)

    best_config_idx = np.argmin(res)
    best_config = X_scaler.inverse_transform(X_samples)[best_config_idx]
    print("rec:::::::", X_scaler.inverse_transform(X_samples))
    print('best_config==', best_config_idx, best_config)
    log_progress('best_config==', best_config_idx, best_config.transpose())
    best_config = np.rint(best_config)
    best_config = best_config.astype(np.int16)

    conf_map = {k: best_config[i] for i, k in enumerate(X_columnlabels)}
    print(conf_map)
    return  conf_map


