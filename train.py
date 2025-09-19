# import pandas as pd
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib


# # Cargar el conjunto de datos
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# # Inicializar y entrenar el modelo
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Guardar el modelo entrenado en un archivo .pkl
# joblib.dump(model, 'model.pkl')

# print("Modelo entrenado y guardado como 'model.pkl'")



import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

import dagshub
dagshub.init(repo_owner='rdgzmanuel', repo_name='our-model-mlops', mlflow=True)

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Iniciar un experimento de MLflow
with mlflow.start_run():
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Inicializar y entrenar el modelo (Logistic Regression)
    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicciones y precisión
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Guardar el modelo
    joblib.dump(model, 'our_model.pkl')

    # Registrar en MLflow
    mlflow.sklearn.log_model(model, "logistic-regression-model")
    mlflow.log_param("max_iter", 100)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")

