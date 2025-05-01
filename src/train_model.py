# Pipeline de treinamento 
import mlflow  
import mlflow.sklearn  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
import pandas as pd  

# Carregar dados  
df = pd.read_csv('data/raw/dados.csv')  
X = df[['temperatura']]  
y = df['vendas']  

# Experimente no MLflow  
mlflow.set_experiment("GelatoMagico-Predict")  

with mlflow.start_run():  
    # Treinar modelo  
    model = LinearRegression()  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
    model.fit(X_train, y_train)  

    # Log de métricas e parâmetros  
    mlflow.log_param("modelo", "LinearRegression")  
    mlflow.log_metric("R2", model.score(X_test, y_test))  
    mlflow.sklearn.log_model(model, "model")  

    # Salvar modelo  
    mlflow.sklearn.save_model(model, "models/linear_regression")  
