import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("health_fitness_dataset.csv")

data = pd.get_dummies(data, columns=['gender', 'activity_type', 'intensity', 'smoking_status'], drop_first=True)

entry_counts = data['participant_id'].value_counts()
data['churn'] = data['participant_id'].map(lambda pid: 1 if entry_counts[pid] < 230 else 0)

agg_funcs = {
    'age': 'first',
    'height_cm': 'first',
    'weight_kg': 'mean',
    'calories_burned': 'mean',
    'avg_heart_rate': 'mean',
    'bmi': 'mean',
    'fitness_level': 'mean'
}

categorical_cols = [col for col in data.columns if col.startswith(('gender_', 'activity_type_', 'intensity_', 'smoking_status_'))]
for col in categorical_cols:
    agg_funcs[col] = 'mean'

aggregated = data.groupby('participant_id').agg(agg_funcs).reset_index()

aggregated['churn'] = aggregated['participant_id'].map(lambda pid: 1 if entry_counts[pid] < 230 else 0)

X = aggregated.drop(columns=['participant_id', 'churn'])
y = aggregated['churn']

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train_raw)

model = RandomForestClassifier(n_estimators=100, random_state=None)
model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test_raw, y_pred))
print("ROC-AUC:", roc_auc_score(y_test_raw, y_pred))
print(classification_report(y_test_raw, y_pred))

churn_count = pd.Series(y_train_bal).value_counts()
total_users = len(y_train_bal)
   
labels = ['Retained (0)', 'At Risk of Quitting (1)']
counts = [churn_count.get(0, 0), churn_count.get(1, 0)]

plt.figure(figsize=(6, 4))
plt.bar(labels, counts, color=['green', 'red'])
plt.title('User Churn Summary in Training Data')
plt.ylabel('Number of Users')
plt.xlabel('Churn Status')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(y_test_raw, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Retained (0)', 'At Risk of Quitting (1)'], yticklabels=['Retained (0)', 'At Risk of Quitting (1)'])
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
