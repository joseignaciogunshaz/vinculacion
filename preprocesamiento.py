import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

class Preprocesamiento:

    def preparar_dataset(self, df):
        print("**   1.1 Preparando modelo")
        df = pd.DataFrame(df)
        df['porcentaje'] = df['porcentaje'].replace({'%':''}, regex=True)
        df['tipo_discapacidad'] = df['tipo_discapacidad'].str.strip()
        df['nivel_educativo'] = df['nivel_educativo'].str.strip()
        df['grado_curso'] = df['grado_curso'].replace({'II':'INICIAL 2'}, regex=True)
        df['porcentaje'] = df['porcentaje'].replace({'NO APLICA':0.0}, regex=True)
        imputer = SimpleImputer(strategy='most_frequent')
        df['asistencia'] = imputer.fit_transform(df['asistencia'].values.reshape(-1,1))[:,0]
        df['regularizado'] = imputer.fit_transform(df['regularizado'].values.reshape(-1,1))[:,0]
        df['resago_3_anios'] = imputer.fit_transform(df['resago_3_anios'].values.reshape(-1,1))[:,0]
        df['carne'] = imputer.fit_transform(df['carne'].values.reshape(-1,1))[:,0]
        df['cedula']=df['cedula'].fillna(value = "0")
        df['fecha_nacimiento']=df['fecha_nacimiento'].fillna(value = "00/00/0000")
        imputer = SimpleImputer(strategy='mean')
        df['edad'] = imputer.fit_transform(df[['edad']])
        df = df.dropna()
        df['carne'] = df['carne'].map({'SI':1, 'Si':1, 'si':1, 'NO': 0, 'No': 0, 'no': 0, 'NO APLICA': 0})
        df['regularizado'] = df['regularizado'].map({'SI':1, 'Si':1, 'si':1, 'NO': 0, 'No': 0, 'no': 0})
        df['asistencia'] = df['asistencia'].map({'SI':1, 'Si':1, 'si':1, 'NO': 0, 'No': 0, 'no': 0})
        df['resago_3_anios'] = df['resago_3_anios'].map({'SI RESAGO':1, 'RESAGO':1, 'NO RESAGO': 0})
        df['carne'] = df['carne'].astype(int)
        df['porcentaje'] = df['porcentaje'].astype(int)
        label_encoder = LabelEncoder()
        label_encoder
        df.loc[:, 'tipo_discapacidad'] = label_encoder.fit_transform(df['tipo_discapacidad'])
        df.loc[:, 'nivel_educativo'] = label_encoder.fit_transform(df['nivel_educativo'])
        df.loc[:, 'grado_curso'] = label_encoder.fit_transform(df['grado_curso'])
        df.loc[:, 'institucion'] = label_encoder.fit_transform(df['institucion'])
        df['tipo_discapacidad'] = df['tipo_discapacidad'].astype(int)
        df['nivel_educativo'] = df['nivel_educativo'].astype(int)
        df['grado_curso'] = df['grado_curso'].astype(int)
        df['institucion'] = df['institucion'].astype(int)
        df['porcentaje'] = df['porcentaje']/100

        return df