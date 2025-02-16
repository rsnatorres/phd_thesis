import time
import numpy as np
import pandas as pd
import plotly.express as px
import csv
import re
import html # for auxiliary into cleaning steps
from collections import Counter
import spacy
from spacy.tokenizer import Tokenizer
from spacy.language import Language
import spacy.tokenizer
from spacy.util import compile_prefix_regex, \
                       compile_infix_regex, compile_suffix_regex
from spacy import displacy # spaCy's displacy module also provides VISUALIZATIONS for the named-entity recognition:
from IPython.display import clear_output
import textacy # for preprocessing
import textacy.preprocessing as tprep
from textacy.preprocessing.resources import RE_URL
import sqlalchemy as db 
from sqlalchemy import create_engine, types
from general_functions import json_to_sqlalchemy_dtypes


#Safely constructs a NLP model from spacy
#@param model_name: a string, containing the name of the language model used by spacy´
#@return nlp: a Python object containing a NLP model
def nlp_model(model_name):
    try:
        nlp = spacy.load(model_name)
        print(f"{model_name} model is already installed.")
    except OSError:
        print(f"{model_name} model is not installed.")
    
    if spacy.prefer_gpu():
        print('Working on GPU.')
    else:
        print('No GPU found, working on CPU')
    return nlp

#Prepares dataframe for language processing
#@param df: a Pandas dataframe containing the data downloaded from the database
#@return df: a Pandas dataframe containing the data downloaded from the database
def prepare_data(df):
    df = df.loc[df['content'].notna()]
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['date_time_day'] = pd.to_datetime(df['date_time'].dt.strftime('%d-%m-%Y'), dayfirst = True)
    df = df[['date_time', 'date_time_day', 'username', 'content', 'user_description', 
             'follower_count', 'reply_count', 'retweet_count', 'like_count', 'lang',
            'hash', 'mining_string']]
    return df

#Cleans text from unnecessary characters
#@param text: a string containing text to be cleaned
#@return text: a string with clean text
def clean_text(text):
  # convert html escapes like &amp; to characters.
  text = html.unescape(text)
  # tags like <tab>
  text = re.sub(r'<[^<>]*>', ' ', text)
  # markdown URLs like [Some text](https://....)
  text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
  # text or code in brackets like [0]
  text = re.sub(r'\[[^\[\]]*\]', ' ', text)
  # standalone sequences of specials, matches &# but not #cool
  text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
  # standalone sequences of hyphens like --- or ==
  text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
  # sequences of white spaces
  text = re.sub(r'\s+', ' ', text)
  return text.strip()

#Normalizes a text
#@param text: a string to be normalized
#@return text: a normalized string
def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    text = tprep.replace.urls(text, '_url_') # a very complete url hunter
    text = tprep.normalize.whitespace(text)
    # text = tprep.replace.emojis(text, '_emoji_') # maintain as vader handles them
    text = text.replace('\\n', '\n') # some linebreaks were using \\n and were not tokenized accordingly
    text = text.lower()
    return text

#Creates a custom NLP tokenizer
#@param nlp: a Python object containing a NLP model
#@return nlp: a Python object containing a NLP model
def custom_tokenization(nlp):
    # use default patterns except the ones matched by re.search
    prefixes = [pattern for pattern in nlp.Defaults.prefixes 
                if pattern not in ['-', '_', '#']]
    suffixes = [pattern for pattern in nlp.Defaults.suffixes
                if pattern not in ['_']]
    infixes  = [pattern for pattern in nlp.Defaults.infixes
                if not re.search(pattern, 'xx-xx')]

    percent_pattern = re.compile(r'\d+(\.\d+)?%') # percent pattern --> treat this as a token
    nlp.tokenizer = Tokenizer(vocab          = nlp.vocab, 
                        rules          = nlp.Defaults.tokenizer_exceptions,
                        prefix_search  = compile_prefix_regex(prefixes).search,
                        suffix_search  = compile_suffix_regex(suffixes).search,
                        infix_finditer = compile_infix_regex(infixes).finditer,
                        token_match    = percent_pattern.match)
    
    @Language.component('_masks_pos_')
    def _url_pos_x(doc):
        for token in doc:
            if token.text == '_url_' or token.text == '_emoji_':
                token.pos_ = 'X'
            if token.text.startswith('@'):
                token.pos_ = 'X'
            if token.text.startswith('#'):
                token.pos_ = 'X'
        return doc

    nlp.add_pipe('_masks_pos_', after='parser')

    return nlp

#Adds stop words to NLP
#@param nlp: a Python object containing a NLP model
#@param words: a list of strings containing stop words
#@return nlp: a Python object containing a NLP model
def add_stop_words(nlp, words):
    for word in words:
        nlp.vocab[word].is_stop = True
    return nlp

#Defines a custom function to extract words, based on textacy.extract.words
#@param doc: a string, text to have words extracted
#@return: words extracted
def extract_word(doc, **kwargs):
    return [t.text for t in textacy.extract.words(doc, **kwargs)]

#Defines a custom function to extract lemmas, based on textacy.extract.words
#@param doc: a string, text to have words extracted
#@return: filtered lemmas extracted 
def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.words(doc, **kwargs)]

#Defines a custom function to extract nouns
#@param doc: a string, text to have words extracted
#@return: filtered nouns extracted 
def extract_noun_phrases(doc, preceding_pos = ['NOUN'], sep = '_'):
    patterns = []
    for pos in preceding_pos:
        patterns.append(f"POS:{pos} POS:NOUN:+")
    spans = textacy.extract.token_matches(doc, patterns = patterns)
    return [sep.join([t.lemma_ for t in s]) for s in spans]

#Defines a custom function to extract named entities
#@param doc: a string, text to have words extracted
#@return: filtered nouns extracted 
def extract_entities(doc, include_types = None, sep = '_'):
    ents = textacy.extract.entities(doc, 
                                    include_types = include_types,
                                    exclude_types = None,
                                    drop_determiners = True,
                                    min_freq = 1)
    return [sep.join([t.lemma_ for t in e]) + '/' + e.label_ for e in ents]

#Defines a custom function to extract noun chunks
#@param doc: a string, text to have words extracted
#@return: filtered nouns extracted 
def extract_noun_chunks(doc, sep = '_', **kwargs):
    return [s.text for s in textacy.extract.basics.noun_chunks(doc, **kwargs)]

#Defines a custom function to extract N-grams chunks
#@param doc: a string, text to have words extracted
#@return: filtered N-grams extracted 
def extract_ngrams(doc, n):
    ngrams = textacy.extract.ngrams(doc, n, filter_stops=True, filter_punct=True, filter_nums=False)
    return [ngram.text for ngram in ngrams]

#Defines a custom function to extract acronyms chunks
#@param doc: a string, text to have words extracted
#@return: filtered acronyms extracted 
def extract_acronyms(doc):
    return [t for t in textacy.extract.acronyms(doc)]

#Defines a custom function to extract yakes chunks
#@param doc: a string, text to have words extracted
#@return: filtered yakes extracted 
def keywords_yake(doc):
    return [t[0] for t in textacy.extract.keyterms.yake(doc, ngrams = (1,2), topn = 3)]

#Defines a custom function to extract scakes chunks
#@param doc: a string, text to have words extracted
#@return: filtered scakes extracted
def keywords_scake(doc):
    return [t[0] for t in textacy.extract.keyterms.scake(doc, topn = 3)]

#Defines a custom function to find hashes
#@param doc: a string, text to have words extracted
#@return: filtered hashes extracted 
def find_hashs(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.matches.regex_matches(doc, **kwargs)]

#Defines a custom function to find users
#@param doc: a string, text to have words extracted
#@return: filtered users extracted 
def find_users(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.matches.regex_matches(doc, **kwargs)]

#Creates a centralized NLP extractor
#@param doc: a string, text to have words extracted
#@param items: a list of strings, containing the operations to be performed in the text
#@return: a dictionary with the NLP extracted 
def extract_nlp(doc, items):
    results = {}
    items_allowed = ['tokens', 'lemmas', 'adjs_verbs', 'nouns', 'noun_phrases', 
                    'adj_noun_phrases', 'entities', 'hashs', 'users', 'noun_chunks',
                    'bigrams', 'trigrams', 'acronyms', 'keywords_yake', 'keywords_scake']
    
    if not items:
        items = items_allowed
    else:
        if not all(x in items_allowed for x in items):
            print('Extraction not allowed! Rerun.')
            return
          
    for item in items:
        if item == 'tokens':
            addition = extract_word(doc, 
                                        exclude_pos = ['PART', 'PUNCT', 'DET', 
                                                    'PRON', 'SYM', 'SPACE'],
                                    filter_stops = True)
        elif item == 'lemmas':
            addition = extract_lemmas(doc,
                                        exclude_pos = ['PART', 'PUNCT', 'DET', 
                                                    'PRON', 'SYM', 'SPACE'],
                                        filter_stops = True)
        elif item == 'adjs_verbs':
            addition = extract_lemmas(doc, include_pos = ['ADJ', 'VERB'])
        elif item == 'nouns':
            addition = extract_lemmas(doc, include_pos = ['NOUN', 'PROPN'])
        elif item == 'noun_phrases':
            addition = extract_noun_phrases(doc, ['NOUN'])
        elif item == 'adj_noun_phrases':
            addition = extract_noun_phrases(doc, ['ADJ'])
        elif item == 'entities':
            addition = extract_entities(doc, ['PERSON', 'ORG', 'GPE', 'LOC'])
        elif item == 'hashs':
            addition = find_hashs(doc, pattern = r'#\w+')
        elif item == 'users':
            addition = find_users(doc, pattern = r'@\w+')
        elif item == 'noun_chunks':
            addition = extract_noun_chunks(doc, drop_determiners = True)                    
        elif item == 'bigrams':
            addition = extract_ngrams(doc, n = 2)
        elif item == 'trigrams':
            addition = extract_ngrams(doc, n = 3)
        elif item == 'acronyms':
            addition = extract_acronyms(doc)
        elif item == 'keywords_yake':
            addition = keywords_yake(doc)
        else:
            addition = keywords_scake(doc)
        results.update({item: addition})        
        
    return results

#Creates a NLP pipeline
#@param df: a dataframe containing treated data
#@param nlp: a Python object containing a NLP model
#@param batch_size: an integer containing the size of the batch. Default: 500
def nlp_pipe_batch(df, nlp, items, batch_size):
    for i in range(0, len(df), batch_size):
        docs = nlp.pipe(df['content'][i:i+batch_size])
    
        for j, doc in enumerate(docs):
        # inner loop: extract features from the processed doc; write into df
            for col, values in extract_nlp(doc, items).items():
                df[col].iloc[i+j] = values
    
            progress = (i+j)/len(df)*100
            clear_output(wait=True)
            print(f"[INFO] Processing chunk: {i + j} of {len(df)} ({int(progress)}%)")
    return df

#Preprocess a chunk of text
#@param df: a dataframe containing the chunk of text to be preprocessed
#@param nlp: a Python object containing a NLP model
#@param items: a list of strings, containing the operations to be performed in the text
#@param batch_size: an integer containing the size of the batch. Default: 500
#@return df: a dataframe containing a preprocessed chunk of text
def preprocessing(df, nlp, items, batch_size):
    df = prepare_data(df)
    df['content_preprocessed'] = df['content'].apply(clean_text)
    df['content_preprocessed'] = df['content'].apply(normalize)
    # extract columns labels
    nlp_columns  = list(extract_nlp(nlp.make_doc(''), items).keys())
    # before we start nlp processing, we initialize the new DataFrame columns
    for col in nlp_columns:
        df[col] = None
    # extract features in batch
    df = nlp_pipe_batch(df, nlp, items, batch_size)
    # we need to serialize the extracted lists to space-separated strings
    # as lists are not supported by most databases
    df[nlp_columns] = df[nlp_columns].applymap(lambda items: ' '.join(items))
    return df

#Preprocess a chunk of text and inserts it into a table
#@param engine: a Python object, a SQL Alchemy engine
#@param model: a string, containing the name of the language model used by spacy´
#@param query: a string containing a set of SQL instructions to create a table
#@param chunk_size: an integer containing the number of rows in the chunk
#@param table_name: a string, containing the name of the table to store the data
#@param sql_alchemy_dtypes: a dictionary, containing the data types of the variables to be stored
#@param items: a list of strings, containing the operations to be performed in the text
#@param batch_size: an integer containing the size of the batch. Default: 500
#@param stop_words: a list of strings containing stop words
def preprocess_table(engine, model, chunk_size, table_name, path_home, file_json, stop_words, items, batch_size):
    query = f"""SELECT *
            FROM {table_name}
            """
    sql_alchemy_dtypes = json_to_sqlalchemy_dtypes(path_home, file_json)
    nlp = custom_tokenization(nlp_model(model))
    if stop_words:
        nlp = add_stop_words(nlp, stop_words)
    with engine.connect() as conn: # guarantees connection is closed after execution
        chunk_number = 1
        for chunk in pd.read_sql(db.text(query), 
                                con=conn,
                                chunksize=chunk_size):
            #clear_output(wait=True)
            print(f"[INFO] Working on chunk: {chunk_number}")
            df = preprocessing(chunk, nlp, items, batch_size)
            # Save the DataFrame to the MySQL database
            df.to_sql(name=f"{table_name}_preprocessed", 
                    con=engine,
                    if_exists='append', 
                    index=False,
                    dtype = sql_alchemy_dtypes)
            chunk_number += 1

#Auxiliary function to count word frequency
#@param df: a dataframe containing the chunk of text whose words 
#@param model: a string, containing the name of the language model used by spacy´
#@param items: a list of strings, containing the operations to be performed in the text
#@param column: a string with the name of the dataframe column whose tokens will be processed
#@param preprocess: a binary, informing if preprocessing was already done. Default None.
#@param stop_words: a list of strings containing stop words
#@param batch_size: an integer containing the size of the batch. Default: 500
#@param min_frequency: an integer containing the minimum word frequency to be counted. Default: 2
#@return freq_df: a dataframe containing the frequency of the words counted
def count_words(df, items, model, column, preprocess=None, stop_words = None, batch_size = 500, min_freq=2):
    nlp = custom_tokenization(nlp_model(model))
    if stop_words is not None:
        nlp = add_stop_words(nlp, stop_words)
    counter = Counter()
    # process tokens and update counter
    def update(doc):
        tokens = doc if preprocess is None else preprocessing(doc, nlp, items, batch_size)
        counter.update(tokens)

    # create counter and run through all data
    try:
        df[column].map(update)
    except Exception as e:
        print('Error updating dataframe column!')
        print(e)

    # transform counter into a DataFrame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq>=@min_freq')
    freq_df.index.name=f'{column}'

    return freq_df.sort_values('freq', ascending=False)
