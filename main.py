import pandas as pd
import numpy as np
import sklearn
from utils import Utils
from models import Models
from preprocesamiento import Preprocesamiento

if __name__ == "__main__":

    utils = Utils()
    models = Models()
    prepropresamiento = Preprocesamiento()

    print("La carga de información ha comenzado")

    data = utils.load_from_csv('./in/EstudiantesDiscapacidades.csv')
    data = prepropresamiento.preparar_dataset(data)
    ##Establecemos las variables que se describen como caracteristicas de los estudiantes para el entrenamiento del modelo  
    drop_columns=['cedula','apellidos_nombres','fecha_nacimiento','nivel_educativo']
    X, y = utils.features_target(data, drop_columns, ['nivel_educativo'])
    ##Definimos el proceso de entrenamiento
    models.grid_training(X,y)

    print(data)