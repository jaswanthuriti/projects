from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import bz2
app = Flask(__name__)   
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://username:password@localhost/db_name'
db = SQLAlchemy(app)
ifile = bz2.BZ2File('m.pkl', 'rb')
model = pickle.load(ifile)
ifile.close()
@app.route("/")
def hello():
    return render_template("login.html")
@app.route("/predict", methods=['POST'])
def predict(): 
    features=[]
    features.append(float(request.form.get("age")))
    features.append(float(request.form.get("gender")))
    features.append(float(request.form.get("fd")))
    features.append(float(request.form.get("hbp")))
    features.append(float(request.form.get("pa")))
    features.append(float(request.form.get("bmi")))
    features.append(float(request.form.get("smoke")))
    features.append(float(request.form.get("alco")))
    features.append(float(request.form.get("slp")))
    features.append(float(request.form.get("rmc")))
    features.append(float(request.form.get("jfc")))
    features.append(float(request.form.get("stress")))
    features.append(float(request.form.get("pr")))
    features.append(float(request.form.get("ufc")))
    final = np.array(features) 
    stat = model.predict_proba(final.reshape(1,-1))[0]
    txt = int((round(stat[1]*100, 0)))
    return render_template('index1.html', pred=txt)
if(__name__=="__main__"):
    app.run(debug=True)