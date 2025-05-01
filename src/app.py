# API para previsões em tempo real (Flask/FastAPI)
from flask import Flask, request, jsonify  
import mlflow.sklearn  
import pandas as pd  

app = Flask(__name__)  
model = mlflow.sklearn.load_model("models/linear_regression")  

@app.route('/predict', methods=['POST'])  
def predict():  
    data = request.json  
    temp = data['temperatura']  
    prediction = model.predict([[temp]])  
    return jsonify({'vendas_previstas': prediction[0]})  

if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0')  
