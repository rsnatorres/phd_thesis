import numpy as np
import pandas as pd
import sqlalchemy as db 
import re
import os
from tqdm import tqdm
from IPython.display import clear_output
from leia import SentimentIntensityAnalyzer

#Imports data from MySQL database
#@param table_name: a string containing the table at MySQL database name
#@param start_date: a datetime with the start date of the period
#@param end_date: a datetime with the end date of the period
#@param mining_strings: a list of strings to be searched
#@param engine: a Python objetc with a SQLAlchemy engine
#@param columns a list of strings with column labels
#@return df: a Pandas dataframe withe queried data
def import_data(table_name, start_date, end_date, mining_strings, engine, columns):
    if not columns:
        columns = ['tokens', 'lemmas', 'adjs_verbs', 'nouns', 'noun_phrases', 
                    'adj_noun_phrases', 'entities', 'hashs', 'users', 'noun_chunks',
                    'bigrams', 'trigrams', 'acronyms', 'keywords_yake', 'keywords_scake']

    query = f"""SELECT *
            FROM {table_name}
            WHERE date_time BETWEEN '{start_date}' AND '{end_date}'
            AND mining_string IN {mining_strings}
            """ 
    with engine.connect() as conn: # guarantees connection is closed after execution
        df = pd.read_sql(db.text(query), con = conn)
    
    try:
        df.drop_duplicates('hash', inplace=True)
    except Exception as e:
        print('Operation aborted!')
        print(e)
    
    for column in columns:
        try:
            df[column] = df[column].str.split()
            if column == 'tokens':
                df['len_tokens'] = df['tokens'].apply(lambda list_tokens: len(list_tokens))
        except Exception as e:
            print('Operation aborted!')
            print(e)

    df['content_lower'] = df['content'].str.lower()
    df['date_time_month'] = pd.to_datetime(df[f'date_time_day'].dt.strftime('%Y-%m'))
    return df

#Creates a OpLex data dataframe
#@param path_language_resources: a string with a folder absolute path
#@param oplex_file: a string with the OpLex .csv file name
#@param df: a Pandas dataframe
#@return df: a Pandas dataframe
def set_oplex(path_language_resources, oplex_file, df):
    oplex = pd.read_csv(os.path.join(path_language_resources, oplex_file))
    positive_terms_op = set(oplex.query('polarity == 1')['term'].unique())
    negative_terms_op = set(oplex.query('polarity == -1')['term'].unique())
    def positive_words(tokens, positive_terms = positive_terms_op):
        positive_tokens = []
        for t in tokens:
            if t.lower() in positive_terms:
                positive_tokens.append(t.lower())
            else:
                pass
        return positive_tokens
        
    def negative_words(tokens, negative_terms = negative_terms_op):
        negative_tokens = []
        for t in tokens:
            if t.lower() in negative_terms:
                negative_tokens.append(t.lower())
            else:
                pass
        return negative_tokens
        
    def sentiment_index(tokens, positive_terms, negative_terms):
        sentiment = 0
        if len(tokens) != 0:
            for t in tokens:
                if t in positive_terms: 
                    sentiment += 1
                elif t in negative_terms:
                    sentiment += -1
            return sentiment/len(tokens)
        else:
            return sentiment
    try:
        df['positive_words'] = df['tokens'].apply(positive_words)
    except Exception as e:
        print('Error!')
        print(e)
    try:
        df['negative_words'] = df['tokens'].apply(negative_words)
    except Exception as e:
        print('Error!')
        print(e)
    try:
        df['sentiment_op'] = df['tokens'].apply(sentiment_index, 
                                            args = (positive_terms_op, negative_terms_op))
    except Exception as e:
        print('Error!')
        print(e)

    conditions = [df['sentiment_op'] < 0,
              df['sentiment_op'] == 0,
              df['sentiment_op'] > 0]

    values = ['negative',
            'neutral',
            'positive']

    df['sentiment_label_op'] = np.select(conditions, values)
    conditions = [df['sentiment_op'] < 0,
              df['sentiment_op'] == 0,
              df['sentiment_op'] > 0]

    values = [-1,
            0,
            1]

    df['sentiment_cat_op'] = np.select(conditions, values)
    df['weight_sentiment_cat_op'] = df['sentiment_cat_op']*(df['like_count'] + 1)
    df['weight_sentiment_op'] = df['sentiment_op']*(df['like_count'] + 1)
    return df

#@Creates a pivoted table with aggregated sentiment data
#@param df: a Pandas dataframe
#@param day_month: a datetime with the day/month of the perioded
#@param weighted: a boolean. 'True' if there are weights applied to the model.
#@param lexicon: a string. Could be one of the following: 'op', 'sentilex', 'lm' or 'vader'
#@return tab_sent: a Pandas dataframe containing a pivoted table
def tab_sent_func(df, day_month, weighted, lexicon):
    tab_sent = df.pivot_table(
        index = f'date_time_{day_month}',
        values = f'sentiment_cat_{lexicon}' if weighted == False else f'weight_sentiment_cat_{lexicon}',
        columns = f'sentiment_label_{lexicon}',
        aggfunc = 'sum')
    tab_sent['negative'] = tab_sent['negative'].replace(np.nan, 0)
    tab_sent['positive'] = tab_sent['positive'].replace(np.nan, 0)
    tab_sent.reset_index(inplace=True) 
    tab_sent.columns.name = None
    # compute index
    tab_sent[ 
            [f'sentiment_agg_carosia_by_{day_month}_with_{lexicon}' if weighted == False else
            f'sentiment_weighted_agg_carosia_by_{day_month}_with_{lexicon}'
            ][0]
            ] = (
        tab_sent['positive'] + tab_sent['negative']) / (tab_sent['positive'] - tab_sent['negative']).replace(
            np.nan, 0)
    tab_sent = tab_sent[[f'date_time_{day_month}', 
                        [f'sentiment_agg_carosia_by_{day_month}_with_{lexicon}'  if weighted == False else
                        f'sentiment_weighted_agg_carosia_by_{day_month}_with_{lexicon}'
                        ][0]
                        ]] 
    return tab_sent

#@Computes aggregated sentiment data by different strategies
#@param df: a Pandas dataframe
#@param day_month: a datetime with the day/month of the perioded
#@param weighted: a boolean. 'True' if there are weights applied to the model.
#@param start_date: a datetime with the start date of the period
#@param end_date: a datetime with the end date of the period
#@param frequency: a string with the frequency of data collection.
#@param lexicon: a string. Could be one of the following: 'op', 'sentilex', 'lm' or 'vader'
#@return tab_sent: a Pandas dataframe containing a pivoted table
#@param agg_strategy: a string with the aggregation strategy. Could be either 'carosia' or 'picault'
#@return tab_sent: a Pandas dataframe, with aggregated sentiment data
def compute_strategies_sentiment_by(df, day_month, start_date, end_date, frequency, lexicon, agg_strategy):
    if agg_strategy == 'carosia':
        tab_sent = tab_sent_func(df = df,
                                day_month = day_month,
                                weighted = True,
                                lexicon = lexicon).merge(tab_sent_func(
                                                                      df = df,
                                                                      day_month = day_month,
                                                                      weighted = False,
                                                                      lexicon = lexicon),
                                      on = 'date_time_month',
                                      how = 'left')
    else:
        tab_sent = df.pivot_table(index = f'date_time_{day_month}',
                                   values = f'sentiment_{lexicon}',
                                   aggfunc = 'mean')
        tab_sent.reset_index(inplace=True)
        tab_sent[f'sentiment_{lexicon}'] = tab_sent[f'sentiment_{lexicon}']*100 
        tab_sent.columns = [f'date_time_{day_month}',
                            f'sentiment_agg_picault_by_{day_month}_with_{lexicon}']
    expected_dates = pd.date_range(start = start_date, end = end_date, freq = frequency)
    if expected_dates[~expected_dates.isin(tab_sent[f'date_time_{day_month}'])].size != 0:
            print('Warning: Some points in time interval maybe missing')
    else:
        pass
    return tab_sent

#Creates combined strategy dataframe
#@param df: a Pandas dataframe
#@param day_month: a datetime with the day/month of the perioded
#@param weighted: a boolean. 'True' if there are weights applied to the model.
#@param start_date: a datetime with the start date of the period
#@param end_date: a datetime with the end date of the period
#@param frequency: a string with the frequency of data collection.
#@param lexicon: a string. Could be one of the following: 'op', 'sentilex', 'lm' or 'vader'
#@return tab_sent: a Pandas dataframe containing a pivoted table
#@return strategies_op: a Pandas dataframe, with aggregated sentiment data
def create_strategies_dataframe(df, day_month, start_date, end_date, frequency, lexicon):
    strategies_df = compute_strategies_sentiment_by(df = df,
                                                day_month = day_month,
                                                start_date = start_date,
                                                end_date = end_date,
                                                frequency = frequency,
                                                lexicon = lexicon,
                                                agg_strategy = 'carosia').merge(compute_strategies_sentiment_by(df = df,
                                                                                                                day_month = day_month,
                                                                                                                start_date = start_date,
                                                                                                                end_date = end_date,
                                                                                                                frequency = frequency,
                                                                                                                lexicon = lexicon,
                                                                                                                agg_strategy = 'picault'),
                                                                                on = f'date_time_{day_month}', 
                                                                                how = 'left')
    return strategies_df

#Creates a SentiLex flexion data dataframe
#@param path_language_resources: a string with a folder absolute path
#@param sentilex_file: a string with the OpLex .csv file name
#@return sentilex_flex: a Pandas dataframe
#@return positive_terms_flex: a Pandas dataframe
#@return negative_terms_flex: a Pandas dataframe
def create_sentilex_flex(path_language_resources, sentilex_flex_file):
    pattern = re.compile(r'([^\.]+)\.PoS=([^;]+);FLEX=([^;]+);TG=([^;]+);POL:N0=([^;]*);(?:POL:N1=([^;]*);)?ANOT=([^\n]+)')
    # Initialize an empty list to store dictionaries
    data_list = []
    # Read the lines from the text file
    with open(os.path.join(path_language_resources, sentilex_flex_file), 'r') as file:
        for line in file:
            # Find all matches in the line
            matches = pattern.search(line)
            if matches:
                for term in matches.group(1).split(','):
                    # Create a dictionary to store the matches
                    data_dict = {'term': term.strip(),
                                'pos': matches.group(2).strip(),
                                'flex': matches.group(3).strip(), # not in lemma.txt
                                'tg': matches.group(4).strip() if matches.group(4) else np.nan,
                                'polarity': matches.group(5).strip() if matches.group(5) else np.nan,
                                'polarity_complement': matches.group(6).strip() if matches.group(6) else np.nan,
                                'anot': matches.group(7).strip() if matches.group(7) else np.nan  }
                    # Append the dictionary to the list
                    data_list.append(data_dict)
    # Create a DataFrame from the list of dictionaries
    sentilex_flex = pd.DataFrame(data_list)
    sentilex_flex['polarity'] = sentilex_flex['polarity'].astype(int)
    #sentilex_flex['term'] = sentilex_flex['term'].str.encode('latin-1').str.decode('utf-8')
    positive_terms_flex = set(sentilex_flex.query('polarity == 1')['term'].unique())
    negative_terms_flex = set(sentilex_flex.query('polarity == -1')['term'].unique())
    return sentilex_flex, positive_terms_flex, negative_terms_flex

#Creates a SentiLex lemma data dataframe
#@param path_language_resources: a string with a folder absolute path
#@param sentilex_file: a string with the OpLex .csv file name
#@return sentilex: a Pandas dataframe
#@return positive_terms_lem: a Pandas dataframe
#@return negative_terms_lem: a Pandas dataframe
def create_sentilex_lemma(path_language_resources, sentilex_lem_file):
    pattern = re.compile(r'([^\.]+)\.PoS=([^;]+);TG=([^;]+);POL:N0=([^;]*);(?:POL:N1=([^;]*);)?ANOT=([^\n]+)')
    # Initialize an empty list to store dictionaries
    data_list = []
    # Read the lines from the text file
    with open(os.path.join(path_language_resources,sentilex_lem_file), 'r') as file:
        for line in file:
            # Find all matches in the line
            matches = pattern.search(line)
            if matches:
                    # Create a dictionary to store the matches
                data_dict = {   'term': matches.group(1).strip(),
                                'pos': matches.group(2).strip(),
                                'tg': matches.group(3).strip(),
                                'polarity': matches.group(4).strip() if matches.group(4) else np.nan,
                                'polarity_complement': matches.group(5).strip() if matches.group(5) else np.nan,
                                'anot': matches.group(6).strip()    }
                # Append the dictionary to the list
                data_list.append(data_dict)
    # Create a DataFrame from the list of dictionaries
    sentilex = pd.DataFrame(data_list)
    sentilex['polarity'] = sentilex['polarity'].astype(int)
    #sentilex['term'] = sentilex['term'].str.encode('latin-1').str.decode('utf-8')
    positive_terms_lem = set(sentilex.query('polarity == 1')['term'].unique())
    negative_terms_lem = set(sentilex.query('polarity == -1')['term'].unique())
    return sentilex, positive_terms_lem, negative_terms_lem

#Creates a SentiLex lemma data dataframe
#@param path_language_resources: a string with a folder absolute path
#@param sentilex_file: a string with the OpLex .csv file name
#@return positive_terms_sentilex: a Pandas dataframe
#@return negative_terms_sentilex: a Pandas dataframe
def create_pos_neg_words_sentilex(path_language_resources, sentilex_flex_file, sentilex_lem_file):
    sentilex_flex, positive_terms_flex, negative_terms_flex = create_sentilex_flex(path_language_resources, sentilex_flex_file)
    sentilex, positive_terms_lem, negative_terms_lem = create_sentilex_lemma(path_language_resources, sentilex_lem_file)
    positive_terms_sentilex = positive_terms_lem.union(positive_terms_flex)
    negative_terms_sentilex = negative_terms_lem.union(negative_terms_flex)
    return positive_terms_sentilex, negative_terms_sentilex

# Compile the regex pattern outside the function
#@param terms: a string
#@return re.compile(...): a string
def compile_pattern(terms):
    return re.compile(r'\b(?:%s)\b' % '|'.join(map(re.escape, terms)))

# Extract positive terms using vectorized operations
#@param text: a string
#@return matches: a list of string
def extract_terms(text, compile_pattern):
    matches = compile_pattern.findall(text)
    return matches

#Extract terms in chunks
#@param df: a Pandas dataframe
#@param new_column: a string with the new column name
#@param terms_to_compile: a Python object with the terms to be compiled
#@param chunk_size: an integer with the number of rows of the chunk
#@return df: a Pandas dataframe
def extract_terms_in_chunks_and_monitor_progress(df, new_column, terms_to_compile, chunk_size :int = 10000):
    chunk_size = chunk_size
    # Tqdm progress bar
    total_chunks = len(df) // chunk_size + 1
    pbar = tqdm(total = total_chunks, desc = 'Processing chunks')
    # Process DataFrame in chunks
    for i in range(0, len(df), chunk_size):
        chunk = df.loc[i:i+chunk_size]['content_lower']
        terms = chunk.apply(extract_terms,
                                     compile_pattern = compile_pattern(terms_to_compile))
        df.loc[i:i+chunk_size, new_column] = terms
        pbar.update(1)  # Update progress bar
        clear_output(wait=True)
    return df

#Calculates the number of words of a text
#@param text: a string
#@return len(...) an integer
def length_text(text):
    return len(text.split())

#Cretas sentiment data by SentiLex
#@param df: a Pandas dataframe
#@param positive_terms: a Pandas dataframe
#@param negative_terms: a Pandas dataframe
#@param chunk_size: an integer with the number of rows of the chunk
#@return df: a Pandas dataframe
def create_sentiment_sentiflex(df, positive_terms, negative_terms, chunk_size):
    df = extract_terms_in_chunks_and_monitor_progress(df,
                                                      new_column = 'positive_terms_sentilex_regex',
                                                      terms_to_compile = positive_terms,
                                                      chunk_size = chunk_size)
    df = extract_terms_in_chunks_and_monitor_progress(df,
                                                      new_column = 'negative_terms_sentilex_regex',
                                                      terms_to_compile = negative_terms,
                                                      chunk_size = chunk_size)
    df['len_words'] = df['content_lower'].apply(length_text)
    df['len_positive_terms_sentilex_regex'] = df['positive_terms_sentilex_regex'].apply(lambda row: len(row))
    df['len_negative_terms_sentilex_regex'] = df['negative_terms_sentilex_regex'].apply(lambda row: len(row))
    df['sentiment_sentilex'] = df.apply(
                                        lambda row: (row['len_positive_terms_sentilex_regex']- row['len_negative_terms_sentilex_regex'])
                                                    / row['len_words'] if row['len_words'] != 0 else None
                                                                ,
                                        axis=1)
    conditions = [df['sentiment_sentilex'] < 0,
              df['sentiment_sentilex'] == 0,
              df['sentiment_sentilex'] > 0]
    values = ['negative',
            'neutral',
            'positive']
    df['sentiment_label_sentilex'] = np.select(conditions, values)
    conditions = [df['sentiment_sentilex'] < 0,
              df['sentiment_sentilex'] == 0,
              df['sentiment_sentilex'] > 0]
    values = [-1,
            0,
            1]
    df['sentiment_cat_sentilex'] = np.select(conditions, values)
    df['weight_sentiment_cat_sentilex'] = df['sentiment_cat_sentilex']*(df['like_count'] + 1)
    df['weight_sentiment_sentilex'] = df['sentiment_sentilex']*(df['like_count'] + 1)
    return df

#Creates a LM data dataframe
#@param path_language_resources: a string with a folder absolute path
#@param lm_file: a string with the lm .csv file name
#@return lm_portugues: a Pandas dataframe
#@return positive_terms_lm: a Pandas dataframe
#@return negative_terms_lm: a Pandas dataframe
def create_lm(path_language_resources, lm_file):
    lm_portuguese = pd.read_csv(os.path.join(path_language_resources, lm_file))
    positive_terms_lm = set(lm_portuguese.query("polarity == 1")['word_pt'].apply(
                lambda row: [word.strip() for word in row.split(';')[0:2] if word != '']).sum())
    negative_terms_lm = set(lm_portuguese.query("polarity == -1")['word_pt'].apply(
                lambda row: [word.strip() for word in row.split(';')[0:2] if word != '']).sum())
    commom_terms = []
    for term in positive_terms_lm:
        if term in negative_terms_lm:
            commom_terms.append(term)
    for term in commom_terms:
        positive_terms_lm.remove(term)
        negative_terms_lm.remove(term)
    return lm_portuguese, positive_terms_lm, negative_terms_lm

#Creates sentiment data by LM
#@param df: a Pandas dataframe
#@param positive_terms: a Pandas dataframe
#@param negative_terms: a Pandas dataframe
#@param chunk_size: an integer with the number of rows of the chunk
#@return df: a Pandas dataframe
def create_sentiment_lm(df, positive_terms, negative_terms, chunk_size):
    df = extract_terms_in_chunks_and_monitor_progress(df,
                                                      new_column = 'positive_terms_lm',
                                                      terms_to_compile = positive_terms,
                                                      chunk_size = chunk_size)
    df = extract_terms_in_chunks_and_monitor_progress(df,
                                                      new_column = 'negative_terms_lm',
                                                      terms_to_compile = negative_terms,
                                                      chunk_size = chunk_size)
    df['len_positive_terms_lm'] = df['positive_terms_lm'].apply(lambda row: len(row))
    df['len_negative_terms_lm'] = df['negative_terms_lm'].apply(lambda row: len(row))
    df['sentiment_lm'] = df.apply(
                                lambda row: (row['len_positive_terms_lm'] - row['len_negative_terms_lm'])
                                            / row['len_words'] if row['len_words'] != 0 else None
                                            , axis=1)
    conditions = [df['sentiment_lm'] < 0,
              df['sentiment_lm'] == 0,
              df['sentiment_lm'] > 0]

    values = ['negative',
            'neutral',
            'positive']

    df['sentiment_label_lm'] = np.select(conditions, values)
    conditions = [df['sentiment_lm'] < 0,
              df['sentiment_lm'] == 0,
              df['sentiment_lm'] > 0]

    values = [-1,
            0,
            1]

    df['sentiment_cat_lm'] = np.select(conditions, values)
    df['weight_sentiment_cat_lm'] = df['sentiment_cat_lm']*(df['like_count'] + 1)
    df['weight_sentiment_lm'] = df['sentiment_lm']*(df['like_count'] + 1)
    return df

#Gets sentiment data by Vader
#@param text: a string
#@return sentiment['compound']: a Series object
def get_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

#Creates sentiment data by Vader
#@param df: a Pandas dataframe
#@param chunk_size: an integer with the number of rows of the chunk
#@param limit: a float. Default: 0.05
#@return df: a Pandas dataframe
def create_sentiment_vader(df, chunk_size, limit):
    total_chunks = len(df) // chunk_size + 1
    pbar = tqdm(total = total_chunks, desc='Processing chunks')
    # process in chunks
    for i in range(0, len(df), chunk_size):
        df.loc[i:i+chunk_size, 'sentiment_vader'] = df.loc[i:i+chunk_size]['content_lower'].apply(get_sentiment_vader)
        pbar.update(1)  # Update progress bar
        clear_output(wait=True)
    conditions = [df['sentiment_vader'] < -limit,
                (df['sentiment_vader'] >= -limit) & (df['sentiment_vader'] <= limit),
                df['sentiment_vader'] > limit]

    values = ['negative',
            'neutral',
            'positive']

    df['sentiment_label_vader'] = np.select(conditions, values)
    conditions = [df['sentiment_vader'] < -limit,
              (df['sentiment_vader'] >= -limit) & (df['sentiment_vader'] <= limit),
              df['sentiment_vader'] > limit]

    values = [-1,
            0,
            1]

    df['sentiment_cat_vader'] = np.select(conditions, values)
    df['weight_sentiment_cat_vader'] = df['sentiment_cat_vader']*(df['like_count'] + 1)
    df['weight_sentiment_vader'] = df['sentiment_vader']*(df['like_count'] + 1)
    return df

#Creates a general strategies dataframe
#@param df: a Pandas dataframe
#@param day_month: a datetime with the day/month of the perioded
#@param weighted: a boolean. 'True' if there are weights applied to the model.
#@param start_date: a datetime with the start date of the period
#@param end_date: a datetime with the end date of the period
#@param frequency: a string with the frequency of data collection.
def create_all_strategies(df, day_month, start_date, end_date, frequency, path_exports):
    strategy_op = create_strategies_dataframe(df, day_month, start_date, end_date, frequency, lexicon = 'op')
    strategy_sentilex = create_strategies_dataframe(df, day_month, start_date, end_date, frequency, lexicon = 'sentilex')
    strategy_lm = create_strategies_dataframe(df, day_month, start_date, end_date, frequency, lexicon = 'lm')
    strategy_vader = create_strategies_dataframe(df, day_month, start_date, end_date, frequency, lexicon = 'vader')
    
    strategies_all = strategy_op.merge(strategy_sentilex,
                                       on = f'date_time_{day_month}', 
                                       how = 'left').merge(strategy_lm,
                                                            on = f'date_time_{day_month}', 
                                                            how = 'left').merge(strategy_vader,
                                                                                on = f'date_time_{day_month}', 
                                                                                how = 'left')
    for col in strategies_all.iloc[:, 1:].columns:
        col_mean, col_std = strategies_all[col].mean(), strategies_all[col].std()
        strategies_all['normalized_' + col] = round((strategies_all[col] - col_mean)/col_std, 4)
    strategies_all.to_csv(os.path.join(path_exports, 'strategies_all.csv'), index=False)

#Creates a general strategies file
#@param table_name: a string with the table name
#@param df: a Pandas dataframe
#@param day_month: a datetime with the day/month of the perioded
#@param weighted: a boolean. 'True' if there are weights applied to the model.
#@param start_date: a datetime with the start date of the period
#@param end_date: a datetime with the end date of the period
#@param frequency: a string with the frequency of data collection.
#@param mining_strings: a list of strings to be mined
#@param engine: a Python SQLAlchemy object
#@param columns: a list of strings with the NLP items
#param path_language_resources: a string with the absolute path of the language resources folder
#@param chunk_size: an integer with the number of rows of the chunk
#@param limit: a float
#@param oplex_file: a string with the name of the OpLex .csv file
#@param sentilex_file: a string with the name of the SentiLex .csv file
#@param lm_file: a string with the name of the LM .csv file
#@param path_exports: a string with the absolute path of the exports folder
def create_all_strategies_file(table_name, start_date, end_date, mining_strings, day_month, frequency, engine, columns, path_language_resources, chunk_size, limit, oplex_file, sentilex_flex_file, sentilex_lem_file,lm_file, path_exports):
    df = import_data(table_name, start_date, end_date, mining_strings, engine, columns)
    df = set_oplex(path_language_resources, oplex_file, df)
    positive_terms_sentilex, negative_terms_sentilex = create_pos_neg_words_sentilex(path_language_resources, sentilex_flex_file, sentilex_lem_file)
    df = create_sentiment_sentiflex(df, positive_terms_sentilex, negative_terms_sentilex, chunk_size)
    lm_portuguese, positive_terms_lm, negative_terms_lm = create_lm(path_language_resources, lm_file)
    df = create_sentiment_lm(df, positive_terms_lm, negative_terms_lm, chunk_size)
    df = create_sentiment_vader(df, chunk_size, limit)
    create_all_strategies(df, day_month, start_date, end_date, frequency, path_exports)


#Compares lexicon sentiment labels
#@param df: a Pandas dataframe
#@return lexicon_labels: a Pandas dataframe
def lexicon_comparison(df):
    sentiment_label_op = df['sentiment_label_op'].value_counts()
    sentiment_label_sentilex = df['sentiment_label_sentilex'].value_counts()
    sentiment_label_lm = df['sentiment_label_lm'].value_counts()
    lexicon_labels = pd.DataFrame({'sentiment_label_op': sentiment_label_op, 
                               'sentiment_label_sentilex': sentiment_label_sentilex, 
                               'sentiment_label_lm': sentiment_label_lm}).transpose()
    lexicon_labels.apply(lambda x: round(x / x.sum(), 2), axis=1)
    return lexicon_labels