from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
scalar = pickle.load(open('scalar.pkl', 'rb'))
model=load_model('model.h5')
@app.route("/", methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        result=""
        self_employed_yes,self_employed_no,graduated_yes,graduated_no=0,0,0,0
        dependents = int(request.form['dependents'])
        income = int(request.form['income'])
        loan_amount = int(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])
        cibil = int(request.form['cibil'])
        if request.form['graduated']=='1':
            graduated_yes=1
        else:
            graduated_no=1
        if request.form['self_employed']=='1':
            self_employed_yes=1
        else:
            self_employed_no=1
        residential_assets = int(request.form['residential_assets'])
        commercial_assets = int(request.form['commercial_assets'])
        luxury_assets = int(request.form['luxury_assets'])
        bank_assets = int(request.form['bank_assets'])
     
        features={' no_of_dependents':dependents,' income_annum':income,' loan_amount':loan_amount,' loan_term':loan_term,' cibil_score':cibil,' residential_assets_value':residential_assets,' commercial_assets_value':commercial_assets,' luxury_assets_value':luxury_assets,' bank_asset_value':bank_assets,' education_ Graduate':graduated_yes,' education_ Not Graduate':graduated_no,' self_employed_ No':self_employed_no,' self_employed_ Yes':self_employed_yes}
        df=pd.DataFrame(features,index=[0])
        df=scalar.transform(df)
        
        prediction = model.predict(df)
        prediction=(prediction>0.5).astype(int)
        if prediction[0]==1:
            result='Approved'
        else:
            result='Not Approved'
        return render_template('result_view.html',result=result)
    return render_template('loan_prediction.html', result="")


if __name__ == "__main__":
    app.run(debug=True)