import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Veri setini yükleme
file_path = rf'.\Housing.csv'  # Dosya yolunuza göre güncelleyin
df = pd.read_csv(file_path)

# Gereksiz sütunları çıkarma
df = df.drop(['id', 'date', "lat", "long"], axis=1)

# Kategorik ve sayısal sütunları ayırma
categorical_features = ['zipcode', "waterfront", "view", "condition", "grade", "bedrooms", "bathrooms", "floors", "yr_built", "yr_renovated"]
numerical_features = df.columns.difference(categorical_features + ['price'])

# Pipeline oluşturma
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Özellikler ve hedef değişkeni ayırma
X = df.drop('price', axis=1)
y = df['price']

# Verileri eğitim ve test setlerine ayırma
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X, y, test_size=0.5, random_state=42)

# Pipeline'ı kullanarak veriyi işleme
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_train_80_processed = pipeline.fit_transform(X_train_80)
X_test_20_processed = pipeline.transform(X_test_20)
X_train_50_processed = pipeline.fit_transform(X_train_50)
X_test_50_processed = pipeline.transform(X_test_50)

# Hedef değişkeni ölçeklendirme
scaler_80 = StandardScaler()
y_train_80_scaled = scaler_80.fit_transform(y_train_80.values.reshape(-1, 1))
y_test_20_scaled = scaler_80.transform(y_test_20.values.reshape(-1, 1))

scaler_50 = StandardScaler()
y_train_50_scaled = scaler_50.fit_transform(y_train_50.values.reshape(-1, 1))
y_test_50_scaled = scaler_50.transform(y_test_50.values.reshape(-1, 1))

def build_model(input_dim, hidden_layers, neurons_per_layer):
    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    for _ in range(hidden_layers):
        model.add(Dense(neurons_per_layer, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_model(model, X_train, y_train_scaled, X_test, y_test_scaled, epochs=5, batch_size=32):
    history = model.fit(X_train, y_train_scaled, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    # Test seti üzerinde tahmin yapma
    y_pred_scaled = model.predict(X_test)
    
    # Kayıp (loss) ve ortalama mutlak hata (MAE) hesaplama
    loss = np.mean(np.square(y_pred_scaled - y_test_scaled))
    mae = np.mean(np.abs(y_pred_scaled - y_test_scaled))
    
    # R-kare değeri hesaplama
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    
    return history, loss, mae, r2

# Model 1: Tek Katmanlı YSA (ADELINE) - 128 nöron - %80 eğitim / %20 test
model_1_128_80 = build_model(X_train_80_processed.shape[1], hidden_layers=0, neurons_per_layer=128)
history_1_128_80, loss_1_128_80, mae_1_128_80, r2_1_128_80 = train_and_evaluate_model(model_1_128_80, X_train_80_processed, y_train_80_scaled, X_test_20_processed, y_test_20_scaled)

# Model 1: Tek Katmanlı YSA (ADELINE) - 128 nöron - %50 eğitim / %50 test
model_1_128_50 = build_model(X_train_50_processed.shape[1], hidden_layers=0, neurons_per_layer=128)
history_1_128_50, loss_1_128_50, mae_1_128_50, r2_1_128_50 = train_and_evaluate_model(model_1_128_50, X_train_50_processed, y_train_50_scaled, X_test_50_processed, y_test_50_scaled)

# Model 2: Çok Katmanlı YSA (MADELINE)(MLP) - 128 nöron - %80 eğitim / %20 test
model_2_128_80 = build_model(X_train_80_processed.shape[1], hidden_layers=3, neurons_per_layer=128)
history_2_128_80, loss_2_128_80, mae_2_128_80, r2_2_128_80 = train_and_evaluate_model(model_2_128_80, X_train_80_processed, y_train_80_scaled, X_test_20_processed, y_test_20_scaled)

# Model 2: Çok Katmanlı YSA (MADELINE)(MLP) - 128 nöron - %50 eğitim / %50 test
model_2_128_50 = build_model(X_train_50_processed.shape[1], hidden_layers=3, neurons_per_layer=128)
history_2_128_50, loss_2_128_50, mae_2_128_50, r2_2_128_50 = train_and_evaluate_model(model_2_128_50, X_train_50_processed, y_train_50_scaled, X_test_50_processed, y_test_50_scaled)


# R-kare değerlerini yazdırma
print("Model 1 (Tek Katmanlı YSA - 128 nöron) %80/20 R-squared:", r2_1_128_80)
print("Model 1 (Tek Katmanlı YSA - 128 nöron) %50/50 R-squared:", r2_1_128_50)
print("Model 2 (Çok Katmanlı YSA - 128 nöron) %80/20 R-squared:", r2_2_128_80)
print("Model 2 (Çok Katmanlı YSA - 128 nöron) %50/50 R-squared:", r2_2_128_50)

# Tahminlerin gerçek değerlerle karşılaştırılması
unscaled_prediction_1_128_80 = scaler_80.inverse_transform(model_1_128_80.predict(X_test_20_processed))
unscaled_prediction_1_128_50 = scaler_50.inverse_transform(model_1_128_50.predict(X_test_50_processed))
unscaled_prediction_2_128_80 = scaler_80.inverse_transform(model_2_128_80.predict(X_test_20_processed))
unscaled_prediction_2_128_50 = scaler_50.inverse_transform(model_2_128_50.predict(X_test_50_processed))


# Tek bir giriş için gerçek ve tahmin edilen fiyatı karşılaştırma
print("datasetindeki 42.verinin Gerçek Fiyatı:", y_test_20.iloc[0])
print("Model 1 (Tek Katmanlı YSA - 128 nöron) - %80/20 Tahmin Edilen Fiyat:", unscaled_prediction_1_128_80[0])
print("Model 1 (Tek Katmanlı YSA - 128 nöron) - %50/50 Tahmin Edilen Fiyat:", unscaled_prediction_1_128_50[0])
print("Model 2 (Çok Katmanlı YSA - 128 nöron) - %80/20 Tahmin Edilen Fiyat:", unscaled_prediction_2_128_80[0])
print("Model 2 (Çok Katmanlı YSA - 128 nöron) - %50/50 Tahmin Edilen Fiyat:", unscaled_prediction_2_128_50[0])

# Tahmin edilen değerleri karşılaştırma
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test_20, unscaled_prediction_1_128_80, color='green', edgecolors=(0, 0, 0), alpha=0.6, label=f'Model 1 R-squared: {r2_1_128_80:.4f} ({int(0.2*100)}% Test Size)')
plt.scatter(y_test_50, unscaled_prediction_1_128_50, color='orange', edgecolors=(0, 0, 0), alpha=0.6, label=f'Model 1 R-squared: {r2_1_128_50:.4f} ({int(0.5*100)}% Test Size)')
plt.scatter(y_test_20, unscaled_prediction_2_128_80, color='blue', edgecolors=(0, 0, 0), alpha=0.6, label=f'Model 2 R-squared: {r2_2_128_80:.4f} ({int(0.2*100)}% Test Size)')
plt.scatter(y_test_50, unscaled_prediction_2_128_50, color='red', edgecolors=(0, 0, 0), alpha=0.6, label=f'Model 2 R-squared: {r2_2_128_50:.4f} ({int(0.5*100)}% Test Size)')
plt.plot([min(y_test_20.min(), y_test_50.min()), max(y_test_20.max(), y_test_50.max())], [min(y_test_20.min(), y_test_50.min()), max(y_test_20.max(), y_test_50.max())], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Comparison of Predicted Prices')
plt.legend()

# Loss değerlerini karşılaştırma
plt.subplot(1, 2, 2)
plt.plot(history_1_128_80.history['loss'], label='Model 1 (128 Neurons) - %80/20 Training Loss', color='green')
plt.plot(history_1_128_80.history['val_loss'], label='Model 1 (128 Neurons) - %80/20 Validation Loss', color='lightgreen', linestyle='--')
plt.plot(history_1_128_50.history['loss'], label='Model 1 (128 Neurons) - %50/50 Training Loss', color='orange')
plt.plot(history_1_128_50.history['val_loss'], label='Model 1 (128 Neurons) - %50/50 Validation Loss', color='navajowhite', linestyle='--')
plt.plot(history_2_128_80.history['loss'], label='Model 2 (128 Neurons) - %80/20 Training Loss', color='blue')
plt.plot(history_2_128_80.history['val_loss'], label='Model 2 (128 Neurons) - %80/20 Validation Loss', color='lightblue', linestyle='--')
plt.plot(history_2_128_50.history['loss'], label='Model 2 (128 Neurons) - %50/50 Training Loss', color='red')
plt.plot(history_2_128_50.history['val_loss'], label='Model 2 (128 Neurons) - %50/50 Validation Loss', color='lightcoral', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Comparison of Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

def train_and_evaluate_with_parameters(X_train, y_train, X_test, y_test, model_name, test_size=0.2, epochs=50, hidden_layers=0, neurons_per_layer=128, batch_size=32, validation_split=0.2):
    # Verileri eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
    
    # Pipeline'ı kullanarak veriyi işleme
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Hedef değişkeni ölçeklendirme
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
    
    # Modeli oluşturup eğitme
    model = build_model(X_train_processed.shape[1], hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer)
    history, loss, mae, r2 = train_and_evaluate_model(model, X_train_processed, y_train_scaled, X_test_processed, y_test_scaled, epochs=epochs, batch_size=batch_size)
    
    # Tahmin edilen değerleri hesaplama
    unscaled_prediction = scaler.inverse_transform(model.predict(X_test_processed))
    
    # Grafik oluşturma
    plt.figure(figsize=(12, 12))
    
    # Tahmin edilen değerleri karşılaştırma
    plt.subplot(2, 1, 1)
    plt.scatter(y_test, unscaled_prediction, color='green', edgecolors=(0, 0, 0), alpha=0.6, label=f'{model_name} - Test Size: {int(test_size*100)}%')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Comparison of Predicted Prices')
    plt.legend()
    
    # Loss değerlerini karşılaştırma
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Comparison of Training and Validation Loss')
    plt.legend()
    
    # Doğruluk oranlarını ekleme
    plt.annotate(f'R-squared: {r2:.4f} ({int(test_size*100)}% Test Size)', xy=(0.2, 0.9), xycoords='axes fraction', fontsize=12, color='black')
    
    plt.tight_layout()
    plt.show()



test_size = float(input("Enter test size (0-1): "))
epochs = int(input("Enter number of epochs: "))
hidden_layers = int(input("Enter number of hidden layers: "))
neurons_per_layer = int(input("Enter number of neurons per layer: "))
batch_size = int(input("Enter batch size: "))
validation_split = float(input("Enter validation split (0-1): "))

# Yeni grafiği oluşturalım
train_and_evaluate_with_parameters(X, y, X, y, model_name="Model New", test_size=test_size, epochs=epochs, hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer, batch_size=batch_size, validation_split=validation_split)


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, loss, mae, r2

# RandomForest modeli eğitimi ve değerlendirmesi
rf_model_80, rf_loss_80, rf_mae_80, rf_r2_80 = train_and_evaluate_rf(X_train_80_processed, y_train_80, X_test_20_processed, y_test_20)
rf_model_50, rf_loss_50, rf_mae_50, rf_r2_50 = train_and_evaluate_rf(X_train_50_processed, y_train_50, X_test_50_processed, y_test_50)

def train_and_evaluate_rf_with_plot(X_train, y_train, X_test, y_test, model_name):
    model, loss, mae, r2 = train_and_evaluate_rf(X_train, y_train, X_test, y_test)
    
    # Tahmin edilen değerleri hesaplama
    y_pred = model.predict(X_test)
    
    # Grafik oluşturma
    plt.figure(figsize=(12, 6))
    
    # Tahmin edilen değerleri karşılaştırma
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, color='blue', edgecolors=(0, 0, 0), alpha=0.6, label=f'{model_name}')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Comparison of Predicted Prices')
    plt.legend()
    
    # Doğruluk oranlarını ekleme
    plt.annotate(f'R-squared: {r2:.4f}', xy=(0.2, 0.9), xycoords='axes fraction', fontsize=12, color='black')
    
    plt.tight_layout()
    plt.show()

# Random Forest modeli eğitimi ve grafiği
train_and_evaluate_rf_with_plot(X_train_80_processed, y_train_80, X_test_20_processed, y_test_20, 'Random Forest (80/20)')
train_and_evaluate_rf_with_plot(X_train_50_processed, y_train_50, X_test_50_processed, y_test_50, 'Random Forest (50/50)')

def print_rf_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    for i in range(len(predictions)):
        print(f"Gerçek Değer: {y_test.iloc[i]}, Tahmin Edilen Değer: {predictions[i]}, Fark: {abs(y_test.iloc[i] - predictions[i])}")

# Random Forest modeli tahminlerini ve gerçek değerleri yazdırma
print("Random Forest (80/20) Test Seti Tahminleri ve Gerçek Değerler:")
print_rf_predictions(rf_model_80, X_test_20_processed, y_test_20)

print("\nRandom Forest (50/50) Test Seti Tahminleri ve Gerçek Değerler:")
print_rf_predictions(rf_model_50, X_test_50_processed, y_test_50)



