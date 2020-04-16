import pandas as pd
import sqlalchemy

from sklearn.datasets import make_classification

from datetime import timedelta
import time
import math
import pickle
import argparse
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Connecor function
def postgres_connector(host, port, database, user, password=None):
    user_info = user if password is None else user + ':' + password
    # example: postgresql://federe r:grandestslam@localhost:5432/tennis
    url = 'postgres://%s@%s:%d/%s' % (user_info, host, port, database)
    return sqlalchemy.create_engine(url, client_encoding='utf-8')


def add_to_feature_table(database_name, feat1, table, tablename, engine, thresholds):
    # begin_time = time.time()

    query = "SELECT * FROM " + database_name

    share_feature = pd.read_sql(query, engine, index_col = ['post_key'])

    # threshold, max is 10 (10 hours data)
    time_thresholds = []
    for i in thresholds:
        time_thresholds.append(timedelta(hours = i))
    
    post_create_table = table[['created_at_hour', 'count']]

    diff_time = share_feature[['created_at_hour', 'count']].sub(post_create_table)
    diff_time.dropna(inplace=True)

    # print(diff_time)
    for time, t in zip(time_thresholds, thresholds):
        diff_time_hours = diff_time.loc[(diff_time["created_at_hour"] < time)]

        diff_time_hours = diff_time_hours.groupby(level="post_key").sum()
        
        t1 = tablename + "_in_" + str(t) + "_hours" 
        
        # rename column in order to do operation between dataframes 
        diff_time_hours = diff_time_hours.rename(columns = {'count': t1})
        
        feat1 = feat1.add(diff_time_hours, fill_value = 0)
        feat1.dropna(inplace=True)

    return feat1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("database_name")
    parser.add_argument("model_path")
    args = parser.parse_args()
    # print(args)
    if len(sys.argv) != 3:
        print("python3 train.py {database_name} {model_path}")
        exit()
    else:
        print(args.database_name)
        db_name = args.database_name.split(':')
        print(db_name)
        db_host = db_name[0]
        port = db_name[1]
        model_filepath = args.model_path
    
    # Get connect engine
    engine = postgres_connector(db_host,
                                int(port),
                                "intern_task",
                                "candidate", 
                                "dcard-data-intern-2020"
                                )

    # First build label and features
    database_name = "posts_train"
    # begin_time = time.time()

    query = "SELECT * FROM posts_train WHERE like_count_36_hour >= 1000"

    table1 = pd.read_sql(query, engine, index_col = 'post_key')
    table1['like_count_36_hour'] = table1['like_count_36_hour'].apply(lambda x : 1)

    one_label_count = len(table1.index)
    query = "SELECT * FROM posts_train WHERE like_count_36_hour < 1000"

    table2 = pd.read_sql(query, engine, index_col = 'post_key')

    # do random sample by one_label_count times
    # table2 = table2.sample(n=one_label_count, random_state=1)
    table2['like_count_36_hour'] = table2['like_count_36_hour'].apply(lambda x : 0)

    table = pd.concat([table1, table2])


    table['count'] = 0
    print(table)
    # end_time = time.time()

    # print("Cost time: {}".format(end_time - begin_time))

    # build feature dataframe
    feat1 = pd.DataFrame(index = table.index)

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

    database_name = ["post_shared_train", "post_comment_created_train", "post_liked_train", "post_collected_train"]
    for db_name, name in zip(database_name, table_names):
        feat1 = add_to_feature_table(db_name, feat1, table, name, engine, thresholds)
    print(feat1)

    label = table
    feat1.sort_index(inplace=True)
    label.sort_index(inplace=True)

    X = []
    Y = []
    for col in feat1.columns:
        X.append( feat1[col].values.tolist() )
    Y = label['like_count_36_hour'].values.tolist()

    # X = X[:, None]
    Xp = np.array(X)
    print(Xp.shape)
    print(Xp.T)
    X = Xp.T
    # print(len(X), len(X[0]))
    # print(len(Y))
    clf = RandomForestClassifier()
    # clf = GaussianNB()
    # clf = LogisticRegression(max_iter=10000)
    clf.fit(X, Y)

    pickle.dump(clf, open(model_filepath, "wb"))
    # print(feat1['collected_3_hour'].values.tolist())

