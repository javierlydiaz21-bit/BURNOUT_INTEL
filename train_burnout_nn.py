import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import json

# 1. Cargar Datos
df = pd.read_csv('mental_health_burnout_tech_2026.csv')

# 2. Mapeos
label_map = {'Low': 0, 'Moderate': 1, 'High': 2, 'Severe': 3}

# Forzamos la conversión a int (por si hay strings o ya están convertidos)
df['burnout_level'] = df['burnout_level'].replace(label_map)
df['burnout_level'] = pd.to_numeric(df['burnout_level'], errors='coerce').fillna(0).astype(int)

# 3. Features
feature_cols = [
    'work_hours_per_week', 'meetings_per_day', 'sleep_hours_per_night', 
    'stress_score', 'work_life_balance_score', 'job_satisfaction_score', 
    'manager_support_score', 'vacation_days_taken'
]
X = df[feature_cols].copy()
y = df['burnout_level'].copy()

X.fillna(X.mean(), inplace=True)
y.fillna(0, inplace=True)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train with RandomizedSearchCV (Limitando iteraciones y samples por ser 100k filas, para que no tarde demasiado en demostración)
# Sample smaller for RandomizedSearch to run fast
X_train_sample, _, y_train_sample, _ = train_test_split(X_train_scaled, y_train, train_size=5000, random_state=42, stratify=y_train)

param_grid = {
    "hidden_layer_sizes": [(16, 16), (32,)],
    "activation": ["relu"],
    "alpha": [0.001, 0.01],
    "learning_rate": ["constant"]
}

mlp = MLPClassifier(max_iter=500, random_state=42)

grid_search = RandomizedSearchCV(
    mlp, param_grid, cv=2, n_jobs=-1, scoring="accuracy", n_iter=2, random_state=42, verbose=1
)

print("Buscando hiperparámetros de Red Neuronal...")
grid_search.fit(X_train_sample, y_train_sample)

# Entrenar en todo el dataset de train
print("Entrenando Red Neuronal en dataset completo...")
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# 7. Evaluación
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

cm = confusion_matrix(y_test, y_pred)

# 8. Guardar
joblib.dump(best_model, 'burnout_nn_model.pkl')
# scaler ya fue guardado en la logistica, pero mantenemos la paridad
joblib.dump(scaler, 'burnout_scaler.pkl')

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "labels": list(label_map.keys()),
    "best_params": grid_search.best_params_
}

with open('burnout_nn_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nRed Neuronal entrenada. Accuracy: {accuracy*100:.2f}%")
