# import necessary libraries
import numpy as np
from flask import Flask,request,render_template
import pickle

# create an object app taking current module as argument
app = Flask(__name__)
# load the pickled file 
model = pickle.load(open('model.pkl','rb'))

#decorator to route to main page
@app.route("/")
def home():
    result = ''
    return render_template('index.html',**locals())#returns the home page
    
# decorator to route to prediction page    
@app.route("/predict", methods=['POST'])
def predict():
    age = float(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    ap_hi = float(request.form['ap_hi'])
    ap_lo = float(request.form['ap_lo'])
    gender = float(request.form['gender'])
    cholesterol = float(request.form['cholesterol'])
    gluc = float(request.form['gluc'])
    smoke = float(request.form['smoke'])
    active = float(request.form['active'])
    prediction = model.predict([[age,height,weight,ap_hi,ap_lo,gender,cholesterol,gluc,smoke,active]])[0]
    if prediction == 0:
           result = 'no cardiovascular disease'
    elif prediction == 1:
           result = 'cardiovascular disease'
    
    
    #returns home page with the prediction
    return render_template('index.html', prediction_text='You might have {}!'.format(result))

# to run 
if __name__ == "__main__":
    app.run()
   

    
    

