import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data
training_data = pd.read_csv("C:/Users/Ayush/Downloads/Smart Energy Management System/datatraining.txt")
print(training_data.head())

# Load test data
test_data = pd.read_csv("C:/Users/Ayush/Downloads/Smart Energy Management System/datatest.txt")
print(test_data.head())

# Load validation data
validation_data = pd.read_csv("C:/Users/Ayush/Downloads/Smart Energy Management System/datatest2.txt")
print(validation_data.head())

# Load training data
training_data = pd.read_csv("datatraining.txt")

# Load test data
test_data = pd.read_csv("datatest.txt")

# Load validation data
validation_data = pd.read_csv("datatest2.txt")

# Preprocess data (e.g., drop timestamp and humidity ratio)
X_train = training_data.drop(columns=["date", "HumidityRatio", "Occupancy"])
y_train = training_data["Occupancy"]

X_test = test_data.drop(columns=["date", "HumidityRatio", "Occupancy"])
y_test = test_data["Occupancy"]

X_val = validation_data.drop(columns=["date", "HumidityRatio", "Occupancy"])
y_val = validation_data["Occupancy"]

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Occupancy distribution

plt.figure(figsize=(7, 5))
sns.countplot(x='Occupancy', data=training_data, hue='Occupancy')
plt.title('Distribution of Occupancy (Target Variable)')
plt.xlabel('Occupancy')
plt.ylabel('Count')
plt.show()

# Correlation matrix

plt.figure(figsize=(10, 8))
corr_matrix = training_data.drop(columns=['date', 'HumidityRatio']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Plot distributions of each feature by occupancy

features = ['Temperature', 'Humidity', 'Light', 'CO2']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Occupancy', y=feature, data=training_data,hue='Occupancy')
    plt.title(f'{feature} by Occupancy')
plt.tight_layout()
plt.show()

# Convert date to datetime
training_data['date'] = pd.to_datetime(training_data['date'])

# Plot time series
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.lineplot(x='date', y='Temperature', hue='Occupancy', data=training_data)
plt.title('Temperature Over Time')

plt.subplot(2, 2, 2)
sns.lineplot(x='date', y='Humidity', hue='Occupancy', data=training_data)
plt.title('Humidity Over Time')

plt.subplot(2, 2, 3)
sns.lineplot(x='date', y='Light', hue='Occupancy', data=training_data)
plt.title('Light Over Time')

plt.subplot(2, 2, 4)
sns.lineplot(x='date', y='CO2', hue='Occupancy', data=training_data)
plt.title('CO2 Over Time')

plt.tight_layout()
plt.show()

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Occupied", "Occupied"]))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Enhanced feature importance plot

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Get predicted probabilities

y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Occupied", "Occupied"],
            yticklabels=["Not Occupied", "Occupied"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("\n\n")
# Plot feature importance (if using Random Forest)
feature_importance = model.feature_importances_
features = training_data.drop(columns=["date", "HumidityRatio", "Occupancy"]).columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

import joblib

# Save the trained Random Forest model
joblib.dump(model, "occupancy_model.h5")
print("✅ Occupancy model saved as occupancy_model.h5")

joblib.dump(scaler, "occupancy_scaler.h5")
print("✅ Occupancy scaler saved as occupancy_scaler.h5")
