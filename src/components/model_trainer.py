import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
       try:
        logging.info("splitting training and test data")
        X_train, y_train,X_test,y_test=(
           train_array[:,:-1],
           train_array[:,-1],
           test_array[:,:-1],
           test_array[:,-1]
        )
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Gradient Boosting":GradientBoostingRegressor(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "XGBoost Regressor": XGBRegressor(),
            "Catboost Regressor": CatBoostRegressor(verbose=False),
            "Random Forest Regressor": RandomForestRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }

        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

        best_model_score=max(sorted(model_report.values()))

        best_model_name=list(model_report.keys())[
           list(model_report.values()).index(best_model_score)
        ]
        best_model=models[best_model_name]

        if (best_model_score<0.6):
           raise CustomException("No best model found")
        logging.info("Best model found on both training and testing dataset")
        
        save_objects(
           file_path=self.model_trainer_config.trained_model_file_path,
           obj=best_model
        )
         
        predicted=best_model.predict(X_test)

        r2_square=r2_score(predicted,y_test)

        return r2_square
       except Exception as e:
        raise CustomException(e,sys)