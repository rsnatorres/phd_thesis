import pandas as pd
from twitter_scraping import crawl_tweets
from storage_manipulation import create_engine, create_table, upload_table, check_remove_duplicates
from preprocessing_tools import preprocess_table
from sentiment_analysis import create_all_strategies_file
from config import SQL_config, path_config, time_config, file_config, nlp_model_config, lists_config, sentiment_config

crawl_tweets(key_words = lists_config.key_words,
             year_start_end_tuples = lists_config.year_start_end_tuples,
             today = pd.to_datetime('today').strftime("%m-%d-%Y"),
             path_dir = path_config.path_storage)


engine = create_engine(user = SQL_config.user,
                        password = SQL_config.password,
                        host = SQL_config.host,
                        port = SQL_config.port,
                        database_name = SQL_config.database_name)
    
create_table(engine = engine,
                path_home = path_config.path_home,
                sql_file = file_config.create_table,
                table_name = SQL_config.table_name)

upload_table(path_dir = path_config.path_storage,
                engine = engine,
                table_name = SQL_config.table_name,
                path_home = path_config.path_home,
                file_json = file_config.table_json_file,
                path_export = path_config.path_exports)

check_remove_duplicates(engine = engine,
                        table_name = SQL_config.table_name,
                        path_home = path_config.path_home,
                        file_json = file_config.table_json_file)

preprocess_table(engine = engine,
                model = nlp_model_config.model_name,
                chunk_size = nlp_model_config.chunk_size,
                table_name = SQL_config.table_name_unique,
                path_home = path_config.path_home,
                file_json = file_config.table_preprocessed_json_file,
                stop_words = nlp_model_config.stop_words,
                items = nlp_model_config.items,
                batch_size = nlp_model_config.batch_size)
        
create_all_strategies_file(table_name = SQL_config.table_name_preprocessed,
                            start_date = time_config.start_date,
                            end_date = time_config.end_date,
                            mining_strings = sentiment_config.mining_strings,
                            day_month = time_config.day_month,
                            frequency = time_config.frequency,
                            engine = engine,
                            columns = nlp_model_config.items,
                            path_language_resources = path_config.path_resources,
                            chunk_size = sentiment_config.chunk_size,
                            limit = sentiment_config.limit,
                            oplex_file = file_config.oplex_file,
                            sentilex_flex_file =  file_config.sentilex_flex_file,
                            sentilex_lem_file =  file_config.sentilex_lem_file,
                            lm_file = file_config.lm_file,
                            path_exports = path_config.path_exports)


    

    

    

    
    



