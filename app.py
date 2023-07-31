from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor



app = Flask(__name__,template_folder="template",static_folder='static')

def load_model():
	df = pd.read_csv('AQI Data.csv')
	df = df.dropna()
	X = df.iloc[:,:-1].values
	y = df.iloc[:,-1].values
	x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
	xgr = XGBRegressor(subsample= 0.8,n_estimators= 1100,min_child_weight= 3,max_depth= 30,learning_rate= 0.05)
	xgr.fit(x_train,y_train)
	return xgr

	

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('real_2018.csv')
    model = load_model()
    my_prediction=model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
