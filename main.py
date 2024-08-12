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

    print("** 1. Empieza carga de data set")

    data = utils.load_from_csv('./in/EstudiantesDiscapacidades.csv')
    data = prepropresamiento.preparar_dataset(data)
    drop_columns=['cedula','apellidos_nombres','fecha_nacimiento','nivel_educativo']
    X, y = utils.features_target(data, drop_columns, ['nivel_educativo'])

    models.grid_training(X,y)

    print(data)