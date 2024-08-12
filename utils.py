import pandas as pd
import numpy as np
import sklearn
import joblib
#escalamiento
class Utils:

    def load_from_csv(self, path):
        return pd.read_csv(path, delimiter=';')
        #return pd.read_csv(path)
    
    def features_target(self, dataset, drop_cols, y):
        print("** 2. Selección variables")
        X = dataset.drop(drop_cols, axis = 1)
        y = dataset[y]
        return X, y
    
    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf,'./models/best_student_model_'+str(round(score,3))+'.pkl')
        print("** 5. Proceso finalizado")
    
    def valor_boolean(self, valor):
        if valor == "SI":
            return 1
        else:
            return 0

    def valor_discapacidad(self, tipo_disc):
        if tipo_disc=="AUDITIVA":
            return 0
        elif tipo_disc=="FISICA":
            return 1
        elif tipo_disc=="INTELECTUAL":
            return 2
        elif tipo_disc=="PSICOSOCIAL":
            return 3
        elif tipo_disc=="VISUAL":
            return 4
        else:
            return 2

    def valor_grado(self, grado):
        if grado=="10MO EGB":
            return 0
        elif grado=="1ERO BACHILLERATO":
            return 1
        elif grado=="2DO BACHILLERATO":
            return 2
        elif grado=="2DO EGB":
            return 3
        elif grado=="3ERO BACHILLERATO":
            return 4
        elif grado=="3ERO EGB":
            return 5
        elif grado=="4TO EGB":
            return 6
        elif grado=="5TO EGB":
            return 7
        elif grado=="6TO EGB":
            return 8
        elif grado=="7MO EGB":
            return 9
        elif grado=="8VO EGB":
            return 10
        elif grado=="9NO EGB":
            return 11
        elif grado=="INICIAL II":
            return 12
        
    def valor_institucion(self, institucion):
        if institucion=="AMAZONAS":
            return 0
        elif institucion=="CIUDAD DE COCA":
            return 1
        elif institucion=="JORGE RODRIGUEZ":
            return 2
        elif institucion=="UNIDAD EDUCATIVA ESPECIALIZADA MANUELA CAÑIZARES":
            return 3
        elif institucion=="PRESIDENTE TAMAYO":
            return 4
        else:
            return 0