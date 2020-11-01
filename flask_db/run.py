from time import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
import requests
res = requests.get('http://localhost:8088')
print(res.content)
#connect to our cluster
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 8088}])



def index_data(data_path, index_name, doc_type):
       import json
   f = open(data_path)
   csvfile = pd.read_csv(f, iterator=True, encoding=”utf8") 
   r = requests.get(‘http://localhost:9200')
   for i,df in enumerate(csvfile): 
       records=df.where(pd.notnull(df), None).T.to_dict()
       list_records=[records[it] for it in records]
       try :
          for j, i in enumerate(list_records):
              es.index(index=index_name, doc_type=doc_type, id=j, body=i)
        except :
           print(‘error to index data’)