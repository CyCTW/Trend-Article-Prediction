import pickle
import numpy as np
import pandas as pd
import sqlalchemy
import math
import argparse
import sys
import csv

from datetime import timedelta
from train import postgres_connector
from train import add_to_feature_table

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("database_name")
    parser.add_argument("model_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    # print(args)
    if len(sys.argv) != 4:
        print("python3 train.py {database_name} {model_path} {output_filepath}")
        exit()
    else:
        print(args.database_name)
        db_name = args.database_name.split(':')
        print(db_name)
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

    # First build label and features
    database_name = "posts_test"
    # begin_time = time.time()

    # query = db_query("post_key, like_count_36_hour", database_name)

    query = "SELECT * FROM " + database_name + " WHERE like_count_36_hour >= 1000 LIMIT 10"

    table1 = pd.read_sql(query, engine, index_col = 'post_key')
    table1['like_count_36_hour'] = table1['like_count_36_hour'].apply(lambda x : 1)

    one_label_count = len(table1.index)
    query = "SELECT * FROM " + database_name + " WHERE like_count_36_hour < 1000 LIMIT 10"

    table2 = pd.read_sql(query, engine, index_col = 'post_key')

    # do random sample by one_label_count times
    # table2 = table2.sample(n=one_label_count, random_state=1)

    table2['like_count_36_hour'] = table2['like_count_36_hour'].apply(lambda x : 0)

    table = pd.concat([table1, table2])

    table['count'] = 0
    print(table)

    feat1 = pd.DataFrame(index = table.index)
    # set offset number
    offset = 5

    threshold = math.ceil(10.0 / offset)
    thresholds = [ (t+1)*offset for t in range(threshold)]
    table_names = ['shared', 'liked', 'comment', 'collected']

    for name in table_names:
        for t in thresholds:
            new_column_name = name + "_in_" + str(t) + "_hours" 
            feat1[new_column_name] = 0

    print(feat1)
    # feat1.loc["0002f1f8-c96b-4332-8d19-9cdfa9900f75", 'share_3_hour'] = 1

    database_name = ["post_shared_test", "post_comment_created_test", "post_liked_test", "post_collected_test"]
    for db_name, name in zip(database_name, table_names):
        feat1 = add_to_feature_table(db_name, feat1, table, name, engine, thresholds)
    # print(feat1)
    

    feat1.sort_index(inplace=True)
    label = table
    label.sort_index(inplace=True)

    X = []
    Y = []
    keys = feat1.index

    for col in feat1.columns:
        X.append( feat1[col].values.tolist() )
    Y = label['like_count_36_hour'].values.tolist()

    # X = X[:, None]
    Xp = np.array(X)
    # print(Xp.shape)
    # print(Xp.T)
    X_test = Xp.T

    load_model = pickle.load(open(model_filepath, "rb"))

    X_pred = load_model.predict(X_test)

    X_pred = list(X_pred)
    keys = list(keys)
    print(X_pred)
    print(type(X_pred))
    print(keys)
    print(type(keys))

    # transform data type
    ans = []
    for key, pred in zip(keys, X_pred):
        ans.append([key, pred])

    # write into csv file
    with open(output_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['post_key', 'is_trending'])
        writer.writerows(ans)

    acc = 0.0
    err = 0
    # print(Y)
    # print(X_pred)
    one = 0.0

    for idx in range(len(Y)):
        if (Y[idx] == 1):
            one += 1.0
        if Y[idx] == X_pred[idx]:
            acc += 1.0
        else:
            err += 1

    print("Accuracy: {}".format( acc / len(Y)))
    print("Error is: {}".format(err))

    print("Hot article total: {}".format(one))
    print("hot article is {}".format(one / len(Y)))