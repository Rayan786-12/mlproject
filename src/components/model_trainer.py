# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor
# )
# import sys
# import os
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from dataclasses import dataclass
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from src.components.data_transformation import DataTransformation
# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object,evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifact",'model.pkl')
# class ModelTrainer:
#     def __init__(self):
#         self.model_train_config=ModelTrainerConfig()


#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("Splitting training and test input data")
#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
#             models = {
#     "Random Forest": RandomForestRegressor(),
#     "Decision Tree": DecisionTreeRegressor(),
#     "Gradient Boosting": GradientBoostingRegressor(),
#     "Linear Regression": LinearRegression(),
#     "K-Nearest Neighbors": KNeighborsRegressor(),
#     "XGBoost": XGBRegressor(),
#     "CatBoost": CatBoostRegressor(verbose=0),
#     "AdaBoost": AdaBoostRegressor(),
# }
#             # params={
#             #     "Decision Tree": {
#             #         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#             #         # 'splitter':['best','random'],
#             #         # 'max_features':['sqrt','log2'],
#             #     },
#             #     "Random Forest":{
#             #         # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
#             #         # 'max_features':['sqrt','log2',None],
#             #         'n_estimators': [8,16,32,64,128,256],
#             #         "max_depth": [None, 10, 20],
#             #     },
#             #     "Gradient Boosting":{
#             #         # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
#             #         'learning_rate':[.1,.01,.05,.001],
#             #         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
#             #         # 'criterion':['squared_error', 'friedman_mse'],
#             #         # 'max_features':['auto','sqrt','log2'],
#             #         'n_estimators': [8,16,32,64,128,256]
#             #     },
#             #     "Linear Regression":{},
#             #     "XGBRegressor":{
#             #         'learning_rate':[.1,.01,.05,.001],
#             #         'n_estimators': [8,16,32,64,128,256]
#             #     },
#             #     "CatBoosting Regressor":{
#             #         'depth': [6,8,10],
#             #         'learning_rate': [0.01, 0.05, 0.1],
#             #         'iterations': [30, 50, 100]
#             #     },
#             #     "AdaBoost Regressor":{
#             #         'learning_rate':[.1,.01,0.5,.001],
#             #         # 'loss':['linear','square','exponential'],
#             #         'n_estimators': [8,16,32,64,128,256]
#             #     }
                
#             # }
#             params = {
#                 "Random Forest": {
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                     "max_depth": [None, 10, 20],
#                 },
#                 "Decision Tree": {
#                     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                 },
#                 "Gradient Boosting": {
#                     'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                     'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                 },
#                 "Linear Regression": {},
#                 "K-Nearest Neighbors": {
#                     'n_neighbors': [3, 5, 7, 9],
#                 },
#                 "XGBoost": {
#                     'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                 },
#                 "CatBoost": {
#                     'depth': [6, 8, 10],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'iterations': [30, 50, 100],
#                 },
#                 "AdaBoost": {
#                     'learning_rate': [0.1, 0.01, 0.5, 0.001],
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                 }
#                 }


#             model_report,best_model_name,best_model=evaluate_models(X_train=X_train,y_train=y_train,y_test=y_test,X_test=X_test,models=models,params=params)
#             best_model_score=model_report[best_model_name]
#             # best_model_name=list(model_report.keys())[
#             #     list(model_report.values()).index(best_model_score)
#             # ]
#             best_model=model_report[best_model_name]
#             if best_model_score<0.6:
#                 raise CustomException("No Best Model Found")
#             logging.info(f"Best found model on both training and testing dataset {best_model_name}")

#             save_object(
#                 file_path=self.model_train_config.trained_model_file_path,
#                 obj=best_model
#             )
#             predicted=best_model.predict(X_test)
#             r2_square=r2_score(y_test,predicted)
#             return r2_square
#         except Exception as e:
#             raise CustomException(e,sys)
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    "max_depth": [None, 10, 20],
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                },
                "XGBoost": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                }
            }

            model_report, best_model_name, best_model = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                y_test=y_test,
                X_test=X_test,
                models=models,
                params=params
            )

            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


# if __name__=='__main__':
#     obj=DataIngestion()
#     train_data,test_data=obj.initiate_data_ingestion()

#     data_transformation=DataTransformation()
#     train_arr,test_arr=data_transformation.initiate_data_transformation(train_data,test_data)


#     modeltrainer=ModelTrainer()
#     modeltrainer.initiate_model_trainer(train_arr,test_arr)




