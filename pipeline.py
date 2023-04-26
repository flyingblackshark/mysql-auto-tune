from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from controller import read_metric, read_knob,  set_knob, knob_set, init_knobs, run_workload, calc_metric, restart_db
from skopt import BayesSearchCV
from rfmodel import configuration_recommendation,build_dataset
from datamodel import RFDataSet
from settings import mysql_ip, mysql_port, target_knob_set, target_metric_name, wl_metrics, loadtype
import numpy as np
import time
import pandas as pd
#import streamlit as st
from res import clean_unsafe_pkl, pack_pkl
def train_pipeline(wltype,searchtype):
    ds = RFDataSet()
    Round=200
    init_knobs()
    metric_list=wl_metrics[wltype]
    ds.initdataset(metric_list)
    num_knobs = len(target_knob_set)
    num_metrics = len(metric_list)

    KEY = str(time.time())
    # while(Round>0):
    #     print("################## start a new Round ##################")
    #     rec = build_dataset(ds)
    #     knob_cache = {}
    #     for x in rec.keys():
    #         set_knob(x, rec[x])
    #         knob_cache[x] = rec[x]

    #     print("Round: ", Round, rec)
    #     restart_db()


    #     new_knob_set = np.zeros([1, num_knobs])
    #     new_metric_before = np.zeros([1, num_metrics])
    #     new_metric_after = np.zeros([1, num_metrics])

    #     for i,x in enumerate(metric_list):
    #         new_metric_before[0][i] = read_metric(x)

    #     for i,x in enumerate(target_knob_set):
    #         new_knob_set[0][i] = read_knob(x, knob_cache)

    #     rres = run_workload(wltype)
    #     print(rres)
    #     if("_ERROR" in rres):
    #         print("run workload error")
    #         exit()

    #     for i,x in enumerate(metric_list):
    #         new_metric_after[0][i] = read_metric(x, rres)

    #     new_metric = calc_metric(new_metric_after, new_metric_before, metric_list)

    #     ds.add_new_data(new_knob_set, new_metric)

    #     import pickle
    #     fp = "train_"+KEY+"_"+str(Round)+"_.pkl"
    #     with open(fp, "wb") as f:
    #         pickle.dump(ds, f)

    #     ds.printdata()

    #     ds.merge_new_data()

    #     Round-=1
    train_set = pd.read_csv("train_set.csv")
    train_set = train_set.drop(axis=0,index=0)
    X_matrix = train_set.iloc[:,:-2].values
    y_matrix = train_set.iloc[:,-2].values
    # X_matrix = X_matrix.to_numpy()
    # # Scale to N(0, 1)
    # X_scaler = StandardScaler()
    # X_scaler.fit(X_matrix)
    # X_scaled = X_scaler.transform(X_matrix)
    # y_matrix = y_matrix.to_numpy()
    # # Scale to N(0, 1)
    # y_scaler = StandardScaler()
    # y_scaler.fit(y_matrix)
    # y_scaled = y_scaler.transform(y_matrix)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_matrix = sc_X.fit_transform(X_matrix)
    y_matrix = np.squeeze(sc_y.fit_transform(y_matrix.reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, test_size=0.2, random_state=42)

    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        # errors = 100 * mean_squared_error(predictions,test_labels)
        # accuracy = 100 - errors 
        accuracy = explained_variance_score(test_labels,predictions)
        print('Model Performance')
        #print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('evs = {:0.2f}%.'.format(accuracy))
       
        return accuracy

    rf = RandomForestRegressor()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['sqrt', 'log2',None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    if searchtype == 'random':
        cv = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    elif searchtype == 'grid':
        cv = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
    elif searchtype == 'bayes':
        cv = BayesSearchCV(estimator = rf, search_spaces= param_grid, cv = 3, n_jobs = -1, verbose = 2)
    # Fit the random search model
    base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model.fit(X_train, y_train)
    cv.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    best_random = cv.best_estimator_
    random_accuracy = evaluate(best_random, X_test, y_test)

    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

    import pickle
    fp = "bestmodel.pkl"
    with open(fp, "wb") as f:
        pickle.dump(best_random, f)

    #model.fit(X_train, y_train) 

def tune_pipeline(wltype):
    clean_unsafe_pkl() # clean pkl when interrupted
    ds = RFDataSet()
    Round=200
    init_knobs()
    metric_list=wl_metrics[wltype]
    ds.initdataset(metric_list)
    num_knobs = len(target_knob_set)
    num_metrics = len(metric_list)

    KEY = str(time.time())
    while(Round>0):
        print("################## start a new Round ##################")
        rec = configuration_recommendation(ds)
        knob_cache = {}
        for x in rec.keys():
            set_knob(x, rec[x])
            knob_cache[x] = rec[x]

        print("Round: ", Round, rec)
        restart_db()


        new_knob_set = np.zeros([1, num_knobs])
        new_metric_before = np.zeros([1, num_metrics])
        new_metric_after = np.zeros([1, num_metrics])

        for i,x in enumerate(metric_list):
            new_metric_before[0][i] = read_metric(x)

        for i,x in enumerate(target_knob_set):
            new_knob_set[0][i] = read_knob(x, knob_cache)

        rres = run_workload(wltype)
        print(rres)
        if("_ERROR" in rres):
            print("run workload error")
            exit()

        for i,x in enumerate(metric_list):
            new_metric_after[0][i] = read_metric(x, rres)

        new_metric = calc_metric(new_metric_after, new_metric_before, metric_list)

        ds.add_new_data(new_knob_set, new_metric)

        import pickle
        fp = "ds_"+KEY+"_"+str(Round)+"_.pkl"
        with open(fp, "wb") as f:
            pickle.dump(ds, f)

        ds.printdata()

        ds.merge_new_data()

        Round-=1
    pack_pkl()



