import pandas as pd
import os  
import sys
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging


## intialize data ingestion configguration

@dataclass

class DataingestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")
    
    
    
## Create data ingestion class

class Dataingestion:
    def __init__(self):
        self.ingestion_config = DataingestionConfig()
        
    def initiate_data_inegstion(self):
        logging.info("Data Ingestion method is starts")
        
        try:
            df=pd.read_csv(os.path.join("notebooks/data","filter_data.csv"))
            logging.info("Dataset read as Pandas DataFrame")
        
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info("Train test_split")
            
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False) 
            
            
            logging.info("Ingestion of Data is Completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                    )
            
            
        
        
        except Exception as e:
            logging.info("Error Occured in Data Ingestion Config")