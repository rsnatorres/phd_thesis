import os
import json
from dotenv import load_dotenv
from general_functions import parse_tuple, safely_env

class SQL_config:
    load_dotenv()
    user: str = os.environ['USER']
    password: str = os.environ['PASSWORD']
    host: str = os.environ['HOST']
    port: str = os.environ['PORT']
    database_name: str = os.environ['DATABASE_NAME']
    table_name: str = os.environ['TABLE_NAME']
    table_name_unique: str = os.environ['TABLE_NAME_UNIQUE']
    table_name_preprocessed: str = os.environ['TABLE_NAME_PREPROCESSED']

class path_config:
    load_dotenv()
    path_home: str = os.environ['PATH_HOME']
    path_storage: str = os.environ['PATH_TO_STORAGE_FOLDER']
    path_resources: str = os.environ['PATH_LANGUAGE_RESOURCES']
    path_economic: str = os.environ['PATH_ECONOMIC_DATA']
    path_statistic: str = os.environ['PATH_STATISTIC_RESOURCES']
    path_lexicons: str = os.environ['PATH_LEXICONS']
    path_exports: str = os.environ['PATH_TO_EXPORTS']

class file_config:
    load_dotenv()
    create_table: str = os.environ['CREATE_TABLE_FILE']
    table_json_file: str = os.environ['SQL_DTYPES_TABLE_FILE']
    table_preprocessed_json_file: str = os.environ['SQL_DTYPES_TABLE_PREPROCESSED_FILE']
    oplex_file: str = os.environ['OPLEX_FILE']
    sentilex_flex_file: str = os.environ['SENTILEX_FLEX_FILE']
    sentilex_lem_file: str = os.environ['SENTILEX_LEM_FILE']
    lm_file: str = os.environ['LM_FILE']
    treasury_file: str = os.environ['TREASURY_FILE']
    selic_file: str = os.environ['SELIC_FILE']
    strategies_file: str = os.environ['STRATEGIES_FILE']
    confidence_file: str = os.environ['CONFIDENCE_FILE']
    industrial_file: str = os.environ['INDUSTRIAL_FILE']
    inflation_file: str = os.environ['INFLATION_FILE']
    ptax_file: str = os.environ['PTAX_FILE']
    pnad_file: str = os.environ['PNAD_FILE']
    employment_file: str = os.environ['EMPLOYMENT_FILE']
    ibovespa_1_file: str = os.environ['IBOVESPA_1_FILE']
    ibovespa_2_file: str = os.environ['IBOVESPA_2_FILE']
    expectations_file: str = os.environ['EXPECTATIONS_FILE']
    finance_file: str = os.environ['FINANCE_FILE']
    cboe_file: str = os.environ['CBOE_FILE']
    embi_file: str = os.environ['EMBI_FILE']
    fbcf_file: str = os.environ['FBCF_FILE']
    ibc_file: str = os.environ['IBC_FILE']
    indicators_file: str = os.environ['INDICATORS_FILE']
    pca_plt_file: str = os.environ['PCA_PLT_FILE']
    var_file: str = os.environ['VAR_FILE']
    booster_file: str = os.environ['BOOSTER_FILE']
    emoji_file: str = os.environ['EMOJI_FILE']
    negate_file: str = os.environ['NEGATE_FILE']
    vader_file: str = os.environ['VADER_FILE']                

class time_config:
    load_dotenv()
    start_date: str = os.environ['START_DATE']
    end_date: str = os.environ['END_DATE']
    day_month: str = os.environ['DAY_MONTH']
    frequency: str = os.environ['FREQUENCY']

class lists_config:
    load_dotenv()
    key_words: list = json.loads(os.environ['KEY_WORDS'])
    year_start_end_tuples: list = [parse_tuple(x) for x in json.loads(os.environ['YEAR_TUPLES'])]

class nlp_model_config:
    load_dotenv()
    model_name: str = os.environ['MODEL_NAME']
    stop_words: list = safely_env('STOP_WORDS')
    items: list = safely_env('ITEMS')
    chunk_size: int = int(os.getenv('CHUNK_SIZE_PREPROCESS', 100000))
    batch_size: int = int(os.getenv('BATCH_SIZE', 500))

class sentiment_config:
    load_dotenv()
    mining_strings: list = os.environ['MINING_STRINGS']
    chunk_size: int = int(os.getenv('CHUNK_SIZE_SENTIMENT', 10000))
    limit: float = float(os.getenv('LIMIT', 0.05))

