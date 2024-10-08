import pandas as pd
import numpy as np
import datetime

# maquina de soporte vectorial
from sklearn.svm import SVR 
# metodo ensembler 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:

    def __init__(self):
        self.reg = {
            'SVR' : SVR(),
            'GRADIENT' : GradientBoostingRegressor()
        }

        #DEFINIR DICCIONARIO
        self.params = {
            'SVR' : {
                'kernel' : ['linear', 'poly', 'rbf'],
                'gamma' : ['auto', 'scale'],
                'C' : [1,5,10]
            },'GRADIENT' : {
                'loss' : ['absolute_error', 'squared_error'],
                'learning_rate' : [0.01, 0.05, 0.1]
            }
        }
    #función de ejecución
    def grid_training(self, X, y):
        print("Iniciamos el entrenamiento del modelo")
        print(datetime.datetime.now())
        best_score = 999
        best_model = None
        print("Entrenando modelo")
        for name, reg in self.reg.items():
            print(" ...... ")
            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        
        print(datetime.datetime.now())
        utils = Utils()
        print("Se inicia la exportación del modelo")
        utils.model_export(best_model, best_score)
