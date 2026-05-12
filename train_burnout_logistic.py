import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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

# 6. Modelo
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluación
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

# 8. Guardar
joblib.dump(model, 'burnout_logistic_model.pkl')
joblib.dump(scaler, 'burnout_scaler.pkl')

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}
with open('burnout_logistic_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Logística Multiclase entrenada. Accuracy: {accuracy*100:.2f}%")
