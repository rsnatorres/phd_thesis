import re
import os
import ast
import json
from sqlalchemy import types
from dotenv import load_dotenv

#Safely retrieves an environmental variable list
#@param key: a string with key to an env variable
#@return variable: a list or string
def safely_env(key):
    load_dotenv()
    try:
        variable = json.loads(os.environ[key])
    except KeyError:
        variable = []
    except:
        variable = os.environ[key]
    return variable

#Parses a string and turns it into a tuple
#@param string
#@return s: a tuple
def parse_tuple(string):
    try:
        s = ast.literal_eval(str(string))
        if type(s) == tuple:
            return s
        return
    except:
        return

#Converts a JSON file to a Python dictionary
#@param path_json: a string containing the absolute path to a JSON file
#@param file_json: a string containing the name of a JSON file
#@return sql_alchemy_dtypes: a Python dictionary
def json_to_sqlalchemy_dtypes(path_json, file_json):
    with open(os.path.join(path_json, file_json)) as json_file:
        sql_alchemy_dtypes = json.load(json_file)
    for key in sql_alchemy_dtypes:
        sql_alchemy_dtypes[key] = eval(sql_alchemy_dtypes[key])
    return sql_alchemy_dtypes

#Abbreviates the name of a month
#@param x: a string containing the name of a month
def get_month_abv(x):
    pattern_month = re.compile(r'^.{0,3}')
    return re.search(pattern_month, x).group(0)

#Gets a year from a text
#@param x: a string containing a year
def get_year(x):
    pattern_year = re.compile(r'\d+')
    return re.search(pattern_year, x).group(0)

#Finds month references
#@param x: a string containing the name of a month
def find_month_reference(x):
    pattern = re.compile(r'-(\w+)\s')
    return re.search(pattern, x).group(1)