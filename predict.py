import pickle
import numpy as np
import pandas as pd
import sqlalchemy
import math
import argparse
import sys
import csv
import time
import train

from datetime import timedelta
from train import offset
from train import postgres_connector
from train import add_to_feature_table
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("database_name")
    parser.add_argument("model_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    db_name = args.database_name.split(':')
    db_host = db_name[0]
    port = db_name[1]
    model_filepath = args.model_path
    output_filepath = args.output_path
    
    # Get connect engine
    engine = postgres_connector(db_host,
                                int(port),
                                "intern_task",
                                "candidate", 
                                "dcard-data-intern-2020"
                                )
    # set begin time
    begin_time = time.time()
    X_test, Y, keys = train.organize_data("posts_test", engine, "test")

    # load model parameter for efficiency
    load_model = pickle.load(open(model_filepath, "rb"))

    X_pred = load_model.predict(X_test)
    
    # transform dataset type
    X_pred = list(X_pred)
    keys = list(keys)

    # zip dataset
    ans = []
    for key, pred in zip(keys, X_pred):
        ans.append([key, pred])

    # write into csv file
    with open(output_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['post_key', 'is_trending'])
        writer.writerows(ans)
    end_time = time.time()

    print("Total cost: {} sec.".format(end_time - begin_time))


    # For testing  
    
    # acc = 0.0
    # err = 0
    # one = 0.0
    # print("Confusion Matrix:")
    # print(confusion_matrix(Y, X_pred, labels=[0, 1]))
    
    # print(f1_score(Y, X_pred, average=None))
    # print("F1 score: label=0")
    # print(f1_score(Y, X_pred, pos_label=0))
    # print("F1 score: label=1")

    # print(f1_score(Y, X_pred, pos_label=1))

    # for idx in range(len(Y)):
    #     if (Y[idx] == 1):
    #         one += 1.0
    #     if Y[idx] == X_pred[idx]:
    #         acc += 1.0
    #     else:
    #         err += 1

    # print("Accuracy: {}".format( acc / len(Y)))
    # print("Error is: {}".format(err))

    # print("Hot article total: {}".format(one))
    # print("hot article is {}".format(one / len(Y)))