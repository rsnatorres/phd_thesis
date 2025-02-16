#Importing libraries and packages
import pandas as pd
import sqlalchemy as db # pandas works better with sqlalchemy
from sqlalchemy import create_engine, types
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import re
import time
import json
from IPython.display import clear_output
import hashlib
from general_functions import json_to_sqlalchemy_dtypes

#Checks and optimizes SQL Alchemy engine
#@param engine: a Python object, a SQL Alchemy engine
def checks_optimizes_engine(engine):
    with engine.connect():
        print("Connection successful!")
    
    #Optimization
    check_server_status = """SHOW GLOBAL VARIABLES LIKE
                            'innodb_buffer_pool_size';"""
    with engine.connect() as connection:
        result1 = connection.execute(db.text(check_server_status)).fetchone()
    if int(result1[1]) == 134217728: # 128mb
        with engine.connect() as connection: # change to 4GB
            connection.execute(db.text("""SET GLOBAL 
                        INNODB_BUFFER_POOL_SIZE = 2147483648;"""))
    with engine.connect() as connection:
        result2 = connection.execute(db.text(check_server_status)).fetchone()
    print(result1, result2) 

#Creates a SQL Alchemy engine to interact with MySQL
#@param user: a string with the username of your MySQL database
#@param password: a string with the password of your MySQL database
#@param host: a string with the host used at your MySQL database
#@param port: a string with the port used at your MySQL database
#@param database_name: a string with the name of your MySQL database
#@return engine: a Python object, a SQL Alchemy engine
def create_engine(user, password, host, port, database_name):
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}"
    engine = db.create_engine(db_url, 
                          #echo = True # print sql statements into console if needed
                         )
    print('Engine created successfully!')
    checks_optimizes_engine(engine)
    return engine

#Creates a sql table to store data
#@param engine: a Python object, a SQL Alchemy engine
#@param sql: a string containing a set of SQL instructions to create a table
def create_table(engine, path_home, sql_file, table_name):
    with open(os.path.join(path_home, sql_file), 'r') as fd:
        sql = fd.read()
    with engine.connect() as connection:
        connection.execute(db.text(sql %table_name))
    print("Table created with success!")

#Creates a unique hash value to a given object
#@param value: a Python object to be hashed
#@return hash_object.hexdigest(): an integer, the unique hash value of a Python object
def create_hash(value):
    hash_object = hashlib.sha256()  # a better algorithm, also demands more mem
    hash_object.update(value.encode('utf-8'))
    return hash_object.hexdigest()


#Transfers data from a .csv file into a SQL table
#@param i: an integer, an iterator
#@param csv: a string, a reference to a .csv file
#@param engine: a Python object, a SQL Alchemy engine
#@param table_name: a string, containing the name of the table to store the data
#@param sql_alchemy_dtypes: a dictionary, containing the data types of the variables to be stored
#@param if_table_sql_exists: a string, a binary variable with default value "append"
#@return query: a string, containing the query used to form the entry
#@return year: an intenger, refering to the year the tweet was posted
#@return length: an integer, refering to to the size of the tweet
#@return time: a timestamp, refering to the time elapsed during the operation
def read_and_write_df_into_sql(i,
                               csv,
                               engine,
                               table_name,
                               sql_alchemy_dtypes, 
                               if_table_sql_exists = 'append'):
    
    start_time = time.time()
    global df
    df = pd.read_csv(csv, 
                    usecols = ['datetime', 'username','content', 'display name', 'user description',
                            'verified', 'follower count', 'friends count', 'statuses count', 'favourites count',
                            'listed count', 'media count', 'location', 'reply count', 'retweet count', 'like count',
                            'quote count', 'lang', 'retweeted tweet'], # 25% less memmory)
                     dtype = {'datetime': str, 'username': str, 'content': str,
                            'display name': str, 'user description': object, 'location': str,
                             'lang': str})
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(columns={'datetime': 'date_time'}, inplace = True)
    # So I can define datatypes (specially varchars len) when creating mysql table and reduce mem allocation
    df = df.query("""(username.str.len() < 50 or username.isnull() ) & \
                    (content.str.len() < 500 or content.isnull() ) & \
                    (display_name.str.len() < 100 or display_name.isnull() ) & \
                    (user_description.str.len() < 300 or user_description.isnull() ) & \
                    (location.str.len() < 200 or location.isnull() ) & \
                    (lang.str.len() < 10 or lang.isnull() ) """)
    # creating the hash
    df['raw_hash'] = df['date_time'] + df['username'] + df['content']
    df['hash'] = df['raw_hash'].apply(create_hash)
    df.drop('raw_hash', axis = 1, inplace = True)
    # registering the search string
    match = re.search(r'df query (.*?)(\d{4})', csv)
    df['mining_string'] = match.group(1).strip()

    # Save the DataFrame to the MySQL database
    df.to_sql(name=table_name, 
              con=engine,
              if_exists=if_table_sql_exists,
              index=False,
              dtype = sql_alchemy_dtypes
             )

    # Report data
    query =  match.group(1).strip()
    try:
        year = pd.to_datetime(df['date_time'].head(1)).dt.year[0]
    except:
        year = 0
    length = len(df)    
    end_time = time.time()
    time_length = end_time - start_time    
    del df
    return query, year, length, time_length


#Uploads a set of files into a SQL table
#@param path_dir: a string containing the absolute path to where the .csv files are stored
#@param engine: a Python object, a SQL Alchemy engine
#@param table_name: a string, containing the name of the table to store the data
#@param sql_alchemy_dtypes: a dictionary, containing the data types of the variables to be stored
def upload_table(path_dir, engine, table_name, path_home, file_json, path_export): # into mysql
    csv_pattern = "*.csv"
    csv_files = glob.glob(os.path.join(path_dir, csv_pattern))
    mining_strings = []
    for csv in csv_files:
        mining_strings.append(re.search(r'df query (.*?)(\d{4})', csv).group(1))
    mining_strings_array = np.unique(np.array(mining_strings))
    files = list(enumerate(csv_files))
    sql_alchemy_dtypes = json_to_sqlalchemy_dtypes(path_home, file_json)

    report = {'i': [],
        'query': [],
        'year': [],
        'len': [],
        'time': []}

    for i, csv in files:
        if i == 0: 
            query, year, length, time_length = read_and_write_df_into_sql(i,
                                                                        csv,
                                                                        engine,
                                                                        table_name,
                                                                        sql_alchemy_dtypes, 
                                                                        if_table_sql_exists = 'replace')
            time.sleep(1) # wait for mysql to create and make table available
        else:
           query, year, length, time_length = read_and_write_df_into_sql(i,
                                                                        csv,
                                                                        engine,
                                                                        table_name,
                                                                        sql_alchemy_dtypes, 
                                                                        if_table_sql_exists = 'append')
        
        report['i'].append(1)
        report['query'].append(query)
        report['year'].append(year)
        report['len'].append(length)
        report['time'].append(time_length)
    
        progress = (i+1)/len(csv_files)*100
        clear_output(wait=True)
        print(f"[INFO] Processing {i+1} of {len(csv_files)} ({int(progress)}%)")
    
    for key in report.keys():
        print(key, len(report[key]))
    
    df_report = pd.DataFrame(report)
    df_report['time'].sum()/60 # minutes taken
    path_file = os.path.join(path_export, 'df_report.xlsx')
    df_report.to_excel(path_file, index=False)

#Checks the table created by the data upload and removes duplicates
#@param engine: a Python object, a SQL Alchemy engine
#@param table_name: a string, containing the name of the table to store the data
#@param sql_alchemy_dtypes: a dictionary, containing the data types of the variables to be stored
def check_remove_duplicates(engine, table_name, path_home, file_json):
    query = f"""SELECT *
            FROM {table_name}"""
    with engine.connect() as conn: # guarantees connection is closed after execution
        df = pd.read_sql(db.text(query), con = conn)
    df.info()

    sql_alchemy_dtypes = json_to_sqlalchemy_dtypes(path_home, file_json)

    #Hash checking
    len_pandas_drop_duplic_multiple_columns = len(df.drop_duplicates(
        subset=['date_time', 'username', 'content']))

    len_pandas_drop_duplic_hash = len(df.drop_duplicates(subset=['hash']))

    if len_pandas_drop_duplic_multiple_columns == len_pandas_drop_duplic_hash:
        print("There was no hash Collision")
        tab_hash = df.pivot_table(index='hash',
                             values='date_time',
                             aggfunc='count')
        tab_hash.reset_index(inplace=True)
        max_hash_len = tab_hash['hash'].apply(lambda row: len(row)).max()

        create_index = f""" CREATE INDEX idx_hash 
                            ON {table_name} (hash({max_hash_len}));"""
        with engine.connect() as connection:
            connection.execute(db.text(create_index))
        df.drop_duplicates(subset=['hash'], inplace=True)
        df.to_sql(name=f"{table_name}_unique", 
          con=engine,
          if_exists='replace', 
          index=False,
          dtype = sql_alchemy_dtypes)

    else:
        print("Attention!!! Hash collision")
    
