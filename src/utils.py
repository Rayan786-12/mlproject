import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
# def evaluate_models(X_train,y_train,X_test,y_test,models,params):
#     try:
#         report={}
#         best_score=-1
#         best_model=None
#         best_model_name=None
#         model_names = list(models.keys())
#         model_objects = list(models.values())

#         for model_name,model in enumerate(model_objects):
#             model_name=model_names[i]
#             para=params[model_name]
#             gs=GridSearchCV(model,para,cv=3,n_jobs=-1,verbose=0)
#             gs.fit(X_train,y_train)
#             # model.fit(X_train,y_train)
#             best_model = gs.best_estimator_
#             best_model.fit(X_train, y_train)
#             # model_name.set_params(**gs.best_params_)
#             # model_name.fit(X_train,y_train)

#             y_train_pred=best_model.predict(X_train)
#             y_test_pred=best_model.predict(X_test)
#             train_model_score=r2_score(y_train,y_train_pred)
#             test_model_score=r2_score(y_test,y_test_pred)
#             report[model_name]=test_model_score
#         return report
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_score = -1
        best_model = None
        best_model_name = None

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            best_estimator = gs.best_estimator_
            best_estimator.fit(X_train, y_train)

            y_test_pred = best_estimator.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

            if test_score > best_score:
                best_score = test_score
                best_model = best_estimator
                best_model_name = model_name

        return report, best_model_name, best_model

    except Exception as e:
        raise CustomException(e, sys)

    
