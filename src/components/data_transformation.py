from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imPipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    
    def get_data_transformer_obj(self, df: pd.DataFrame):
        """
        This function is responsible for data transformation
        """
        try:
            df = self.handle_missing_values(df= df)

            logging.info(f"Categorical columns: {df.columns[df.dtypes == 'object']}")
            logging.info(f"Numerical columns: {df.columns[df.dtypes =='float64']}")

            df = self.encoding_catgeories(df = df)

            return df
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Imputation process started")
            for i in df.columns:
                if i in df.select_dtypes(include=['float64', 'int64']):
                    df[i] = df[i].replace(np.nan, df[i].median())
                elif i in df.select_dtypes(include='object'):
                    df[i] = df[i].replace(np.nan, df[i].mode()[0])
                else:
                    raise CustomException("While imputing missing values, something went wring", sys)
            logging.info("Imputation process completed successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def encoding_catgeories(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Arguments:
            df: It takes dataframe and do the categorical columns encoding
        Returns:
            returns the final output in Dataframe
        """
        try:
            logging.info("Categorical Encoding started")
            for i in df.select_dtypes(include='object'):
                df[i] = df[i].astype('category').cat.codes
            logging.info("Categorical Encoding completed successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    def handle_class_imbalance(self, df: pd.DataFrame, target_feat: str) ->pd.DataFrame:
        """
        Parameters:
            df: Dataframe, requires train dataframe
            target_feat: str, requires the name of target variable
        Returns:
            An array of train and target features
        """
        try:
            logging.info("Class balancing process started")
            X = df.drop([target_feat], axis=1)
            Y = df[target_feat]
            X_res, Y_res = SMOTE(random_state=42).fit_resample(X, Y)
            logging.info("Class balancing completed")
            return X_res, Y_res
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data completed")
            logging.info("Obtaining preprocessing object")

            target_column = "went_on_backorder"
            
            input_feature_train_df = train_df.drop(columns=["sku"], axis=1)
            preprocessed_train_df = self.get_data_transformer_obj(df=input_feature_train_df)

            input_feature_train_arr, target_feature_train_df = self.handle_class_imbalance(df=preprocessed_train_df, target_feat=target_column)

            input_feature_test_df = test_df.drop(columns=["sku"], axis=1)
            input_feature_test_arr = self.get_data_transformer_obj(df=input_feature_test_df).to_numpy()
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframe")


            train_arr = np.c_[
                input_feature_train_arr, target_feature_train_df
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Save preprocessing object")
            # save_object(
            #     file_path= self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj
            # )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e, sys) from e
