from flask import Flask,request,render_template
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def survival():
    Pclass=int(request.form.get("Pclass",False))
    Sex=int(request.form.get("Sex",False))
    Age=int(request.form.get("Age",False))
    SibSp=int(request.form.get("SibSp",False))
    Parch=int(request.form.get("Parch",False))
    Fare=int(request.form.get("Fare",False))
    Embarked=int(request.form.get("Embarked",False))

    result=model.predict(np.array([Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]).reshape(1,7))
    if result[0]==1:
        return "<h1 style='color:green'>Survived</h1>"
    else:
        return "<h1 style='color:red'>NOT Survived</h1>"
    

