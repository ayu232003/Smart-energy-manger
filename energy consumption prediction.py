import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import eli5
import shap
from eli5.sklearn import PermutationImportance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc
import gc
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# GPU Configuration for RTX 3050 (4GB VRAM)
print("=== GPU CONFIGURATION ===")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory limit to 3.5GB (leaving some buffer for system)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3584)]  # 3.5GB in MB
        )
        
        print(f"✅ GPU Found: {gpus[0]}")
        print(f"✅ Memory growth enabled")
        print(f"✅ Memory limit set to 3.5GB")
        
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")
else:
    print("❌ No GPU found. Running on CPU.")

# Verify GPU is being used
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Load the dataset
file_path = "C:/Users/Ayush/Downloads/Smart Energy Management System/household_power_consumption.txt"
data = pd.read_csv(file_path, sep=";", low_memory=False, on_bad_lines='skip')

print(f"Dataset shape: {data.shape}")

# Data Preprocessing
# Replace missing values (marked with "?") with NaN
data.replace("?", np.nan, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Convert columns to numeric
data["Global_active_power"] = pd.to_numeric(data["Global_active_power"])

# Combine Date and Time columns into a single DateTime column
data["DateTime"] = pd.to_datetime(data["Date"] + " " + data["Time"], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Drop rows with invalid DateTime values (NaT)
data.dropna(subset=['DateTime'], inplace=True)

# Sort data by DateTime
data.sort_values("DateTime", inplace=True)

print(f"Dataset shape after preprocessing: {data.shape}")

# OPTIMIZATION: Reduce dataset size for 4GB GPU if dataset is too large
MAX_SAMPLES = 500000  # Adjust based on your GPU memory
if len(data) > MAX_SAMPLES:
    print(f"⚠️  Dataset too large ({len(data)} samples). Sampling {MAX_SAMPLES} recent samples for GPU efficiency.")
    data = data.tail(MAX_SAMPLES).reset_index(drop=True)

# Select the target variable: Global_active_power
target = data["Global_active_power"].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler.fit_transform(target.reshape(-1, 1))

# Prepare the dataset for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# OPTIMIZATION: Reduced time step for memory efficiency
time_step = 30  # Reduced from 50 to save GPU memory
X, y = create_dataset(target_scaled, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"Training data shape: X={X.shape}, y={y.shape}")

# Line Plot for Energy Consumption Over Time
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Global_active_power'], label='Energy Consumption')
plt.title('Energy Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of Energy Consumption Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Global_active_power'], kde=True, bins=30)
plt.title('Distribution of Energy Consumption')
plt.xlabel('Energy Consumption (kWh)')
plt.ylabel('Frequency')
plt.show()

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# GPU-OPTIMIZED LSTM MODEL
print("\n=== BUILDING GPU-OPTIMIZED LSTM MODEL ===")

# Clear any existing models from memory
tf.keras.backend.clear_session()
gc.collect()

# Build the LSTM model (optimized for 4GB GPU)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_step, 1), 
             dropout=0.2, recurrent_dropout=0.2),  # Reduced units from 50 to 32
        LSTM(32, return_sequences=False, 
             dropout=0.2, recurrent_dropout=0.2),  # Reduced units from 50 to 32
        Dense(16),  # Reduced from 25 to 16
        Dense(1)
    ])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mae'])

# Model summary
model.summary()

# GPU-OPTIMIZED TRAINING
print("\n=== TRAINING ON GPU ===")

# Callbacks for memory management
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
]

# Train the model with optimized batch size for GPU
batch_size = 64  # Optimized for RTX 3050
epochs = 15      # Increased slightly since we have GPU acceleration

history = model.fit(
    X_train, y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# Make predictions
print("\n=== MAKING PREDICTIONS ===")
train_predict = model.predict(X_train, batch_size=batch_size)
test_predict = model.predict(X_test, batch_size=batch_size)

# Inverse transform the predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform the actual values to original scale
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:1000], label="Actual Power Consumption", alpha=0.7)  # Show first 1000 points
plt.plot(test_predict[:1000], label="Predicted Power Consumption", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Global Active Power (kW)")
plt.title("Energy Consumption Prediction using GPU-Optimized LSTM")
plt.legend()
plt.show()

# Actual vs Predicted Values Plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test_actual, test_predict, alpha=0.3)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Energy Consumption')
plt.show()

# Define thresholds for classification
def classify_energy_consumption(value):
    if value < 1:
        return 0  # Low
    elif 1 <= value < 3:
        return 1  # Medium
    else:
        return 2  # High

# Classify actual and predicted values
y_test_actual_class = np.array([classify_energy_consumption(val) for val in y_test_actual.flatten()])
test_predict_class = np.array([classify_energy_consumption(val) for val in test_predict.flatten()])

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test_actual_class, test_predict_class)
precision = precision_score(y_test_actual_class, test_predict_class, average="weighted")
recall = recall_score(y_test_actual_class, test_predict_class, average="weighted")

# Print results
print(f"\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Calculate additional metrics
mse = np.mean((y_test_actual - test_predict) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test_actual - test_predict))

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Save Keras LSTM model
model.save("energy_consumption_model_gpu.h5")
print("✅ GPU-optimized energy consumption LSTM model saved as energy_consumption_model_gpu.h5")

# Memory cleanup
del model, X_train, X_test, y_train, y_test, train_predict, test_predict
gc.collect()
tf.keras.backend.clear_session()

print("\n🚀 GPU training completed successfully!")