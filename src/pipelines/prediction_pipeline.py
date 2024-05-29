import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            logging.info("predict_pipeline intiate")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            logging.info("find path of preprocessor and model")
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            logging.info("Finding model and preprocessor")  
            # preprocessor=load_object(preprocessor_path)
            # model=load_object(model_path)
            logging.info(features)
            data_Scaled=preprocessor.transform(features)
            logging.info("predict output using model")
            pred=model.predict(data_Scaled)
            logging.info("Predictio is done")
            return pred
        

        

    
        
            
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                age:float,
                workclass:str,
                fnlwgt:float,
                education:str,
                marital_status:str,
                occupation:str,
                relationship:str,
                race:str,
                sex:str,
                capital_gain:float,
                capital_loss:float,
                hours_per_week:float,
                country:str
                ):
        
        self.age=age
        self.workclass=workclass
        self.fnlwgt=fnlwgt
        self.education=education
        self.marital_status=marital_status
        self.occupation=occupation
        self.relationship=relationship
        self.race=race	
        self.sex=sex		
        self.capital_gain=capital_gain
        self.capital_loss=capital_loss
        self.hours_per_week=hours_per_week
        self.country=country

        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'workclass':[self.workclass],
                'fnlwgt':[self.fnlwgt],
                'education':[self.education],
                'marital_status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'race':[self.race],
                'sex':[self.sex],
                'capital_gain':[self.capital_gain],
                'capital_loss':[self.capital_loss],
                'hours_per_week':[self.hours_per_week],
                'country':[self.country]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
