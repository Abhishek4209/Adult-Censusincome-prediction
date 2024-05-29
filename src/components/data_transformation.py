from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import sys
import os 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.utils import save_object




## Data Transformation Config

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    


## Data Transformation Config

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transfomration intiated")
            # Define which columns should be oridinal -encoded and which should be scaled
        
            numeric_features =['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
            categorical_features =['workclass', 'education', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'country']
        
        
            logging.info("Pipeline intiated")
            
            
            ## Pipeline intialize
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder()),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numeric_features),
            ('cat_pipeline',cat_pipeline,categorical_features)
            ])

            return preprocessor
        
            logging.info("Pipeline Completed")
            
                    
        except Exception as e:
            logging.info("Error Occured in Data Transformation")
    
    
    
    
    def  initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
        
        
            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")
        
            logging.info("Obtaing preprocessing object")
        
            preprocessing_obj=self.get_data_transformation_object()
        
            target_column_name="salary"
        
            drop_columns=[target_column_name]
        
            # Feature devide  into independet and depedent features
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name] 



            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name] 
        
            ## apply the transformation 
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datsets.")
            
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor pickle in create and saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
            raise CustomException(e,sys)