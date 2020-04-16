import pandas as pd
import sqlalchemy

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

# set offset number
offset = 3

# Connecor function
def postgres_connector(host, port, database, user, password=None):
    user_info = user if password is None else user + ':' + password
    # example: postgresql://federe r:grandestslam@localhost:5432/tennis
    url = 'postgres://%s@%s:%d/%s' % (user_info, host, port, database)
    return sqlalchemy.create_engine(url, client_encoding='utf-8')


def add_to_feature_table(database_name, tablename, feat1, label, engine, thresholds):
    # begin_time = time.time()

    query = "SELECT * FROM " + database_name

    share_feature = pd.read_sql(query, engine, index_col = ['post_key'])

    # threshold, max is 10 (10 hours data)
    time_thresholds = []
    for i in thresholds:
        time_thresholds.append(timedelta(hours = i))
    
    post_create_table = label[['created_at_hour', 'count']]

    # subtract create time dataframe to get the time difference.
    diff_time = share_feature[['created_at_hour', 'count']].sub(post_create_table)
    diff_time.dropna(inplace=True)

    for time, t in zip(time_thresholds, thresholds):
        # compare with thresholds
        diff_time_hours = diff_time.loc[(diff_time["created_at_hour"] < time)]

        diff_time_hours = diff_time_hours.groupby(level="post_key").sum()
        
        t1 = tablename + "_in_" + str(t) + "_hours" 
        
        # rename column in order to do operation between dataframes 
        diff_time_hours = diff_time_hours.rename(columns = {'count': t1})
        
        # add data we get to the feature dataframe
        feat1 = feat1.add(diff_time_hours, fill_value = 0)
        feat1.dropna(inplace=True)

    return feat1

def organize_data(database_post, engine, type_):


    # Build label Dataframe
    database_name = database_post

    label = []

    if type_== 'train':
        # label 1 with like count >= 1000
        query = "SELECT * FROM " + database_name + " WHERE like_count_36_hour >= 1000"

        table1 = pd.read_sql(query, engine, index_col = 'post_key')
        table1['like_count_36_hour'] = table1['like_count_36_hour'].apply(lambda x : 1)
        one_label_count = len(table1.index) # count one label numbers

        # label 0 with like count < 1000
        query = "SELECT * FROM " + database_name + " WHERE like_count_36_hour < 1000"

        table2 = pd.read_sql(query, engine, index_col = 'post_key')

        ## do random sample in bigger datasets, but perform not good enough
        # table2 = table2.sample(n=one_label_count, random_state=1)

        table2['like_count_36_hour'] = table2['like_count_36_hour'].apply(lambda x : 0)

        # concat dataframe
        label = pd.concat([table1, table2])

        label['count'] = 0
    # test data have no like_count_36_hour
    elif type_ == 'test':
        query = "SELECT * FROM " + database_name
        
        label = pd.read_sql(query, engine, index_col = 'post_key')
        label['count'] = 0
    
    # Build feature Dataframe
    feat1 = pd.DataFrame(index = label.index)

    threshold = math.ceil(10.0 / offset)
    thresholds = [ (t+1)*offset for t in range(threshold)]
    table_names = ['shared', 'liked', 'comment', 'collected']

    # create columns in feat1
    for name in table_names:
        for t in thresholds:
            new_column_name = name + "_in_" + str(t) + "_hours" 
            feat1[new_column_name] = 0

    if type_=="train":
        database_name = ["post_shared_train", "post_comment_created_train", "post_liked_train", "post_collected_train"]
    else:
        database_name = ["post_shared_test", "post_comment_created_test", "post_liked_test", "post_collected_test"]

    # create features according to the threshold
    for db_name, tb_name in zip(database_name, table_names):
        print("Collecting data in " + db_name + "...")
        feat1 = add_to_feature_table(db_name, tb_name, feat1, label, engine, thresholds)
    
    feat1.sort_index(inplace=True)
    label.sort_index(inplace=True)

    X = []
    Y = []
    for col in feat1.columns:
        X.append( feat1[col].values.tolist() )
    if type_ == "train":
        Y = label['like_count_36_hour'].values.tolist()
    
    # transpose data to the right shape
    Xp = np.array(X)
    X = Xp.T
    keys = label.index
    return X, Y, keys

if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("database_name")
    parser.add_argument("model_path")
    args = parser.parse_args()

    db_name = args.database_name.split(':')
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
    
    # set begin time
    begin_time = time.time()

    X, Y, keys = organize_data("posts_train", engine, "train")

    # build model and fit 
    print("Start to do model fitting...")
    clf = RandomForestClassifier()
    clf.fit(X, Y)

    end_time = time.time()
    print("Train Cost time: {} sec.".format(end_time - begin_time))

    pickle.dump(clf, open(model_filepath, "wb"))

