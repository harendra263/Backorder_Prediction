import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from imblearn.over_sampling import SMOTE
from src.utils import evaluate_model, save_object


from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "XgBoost Classifier": XGBClassifier()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Classifier": {
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XgBoost Classifier": {
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_model(X_train= X_train, y_train= y_train, X_test= X_test, y_test=  y_test, 
                                               models = models, params= params)

            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No Best model found")
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            return accuracy_score(y_test, predicted)
        except Exception as e:
            raise CustomException(e, sys) from e

