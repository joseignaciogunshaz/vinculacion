import joblib
import numpy as np

from flask import Flask
from flask import jsonify
from flask import render_template, request, redirect, url_for
from utils import Utils

app = Flask(__name__)

posts = []
#@app.route("/")
#def index():
#    return "PROYECTO DE VINCULACIÓN (Diríjase a http://localhost:5000/formulario_estudiantes/)"

@app.route('/')
def index():
    return "Hello, World!"

#POSTMAN PARA PRUEBAS
@app.route("/formulario_estudiantes/", methods=["GET", "POST"])
def show__formulario_est():
    utils = Utils()

    model = joblib.load('./models/best_student_model_0.082.pkl')

    discapacidades = ['FISICA','INTELECTUAL','AUDITIVA','PSICOSOCIAL','VISUAL','TRASTORNO MIXTO DE HABILIDADES ESCOLARES','RETRASO MENTAL LEVE','EPILEPSIA']
    values = ['SI','NO']
    grados = ['INICIAL I','INICIAL II','2DO EGB','3ERO EGB','4TO EGB','5TO EGB','6TO EGB','7MO EGB','8VO EGB','9NO EGB','10MO EGB','1ERO BACHILLERATO','2DO BACHILLERATO','3ERO BACHILLERATO']
    instituciones = ['PRESIDENTE TAMAYO','UNIDAD EDUCATIVA ESPECIALIZADA MANUELA CAÑIZARES','AMAZONAS','CIUDAD DE COCA','JORGE RODRIGUEZ','FRANCISCO DE ORELLANA','NARCISA DE JESUS']

    if request.method == 'POST':
        edad = request.form['edad']
        discapacidad_f = request.form['discapacidades']
        porcentaje = request.form['porcentaje']
        carne_f = request.form['carne']
        regularizado_f = request.form['regularizado']
        resago_f = request.form['resago']
        asistencia_f = request.form['asistencia']
        grado_f = request.form['grados']
        institucion_f = request.form['instituciones']

        #x_test = np.array([15,1,2,0.5,2,1,1,1,2])
        carne = utils.valor_boolean(carne_f)
        discapacidad  = utils.valor_discapacidad(discapacidad_f)
        grado = utils.valor_grado(grado_f)
        regularizado = utils.valor_boolean(regularizado_f)
        resago = utils.valor_boolean(resago_f)
        asistencia = utils.valor_boolean(asistencia_f)
        institucion = utils.valor_institucion(institucion_f)

        x_test = np.array([edad,carne,discapacidad,porcentaje,grado,regularizado,resago,asistencia,institucion])
        prediction = model.predict(x_test.reshape(1,-1))
        return jsonify({'prediction' : list(prediction)}) 
        #0 BACHILLERATO - 1 BASICA - 2 INICIAL

    return render_template("prediccion_form.html", discapacidades=discapacidades, values=values, grados=grados,instituciones=instituciones)
    
if __name__ == '__main__':
    #model = joblib.load('./models/best_student_model_0.932.pkl')
    model = joblib.load('./models/best_student_model_0.082.pkl')
    app.run(port=5000)