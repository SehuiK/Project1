# 필요한 라이브러리 임포트
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow_addons as tfa  # TensorFlow-Addons 추가
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from math import sqrt
import joblib

########################### 하이퍼 파라미터 설정 #############################
time_steps_list = [3, 6, 12, 24, 48, 96, 288]
outer_n_splits_options = {
    4: [4, 28, 56],
    8: [4, 14, 28],
    16: [7, 14]
}
learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
# 하이퍼 파라미터 설정
optimizer_options = {
    'adam': Adam,
    'rmsprop': RMSprop
}
batch_size_list = [16, 32, 64, 128]
lstm_neurons_1_list = [128]
lstm_neurons_2_list = [64]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4]
window_size_list = [6, 12, 24, 48, 96, 288]

# 사용자 선택에 따라 변수 설정
time_steps = 12
outer_n_splits = 4
inner_n_splits = outer_n_splits_options[outer_n_splits][0]  # 예시로 첫 번째 값을 선택
learning_rate = 0.001
optimizer_choice = 'adam'
optimizer_class = optimizer_options[optimizer_choice]
batch_size = 64
lstm_neurons_1 = 64
lstm_neurons_2 = 32
dropout_rate = 0.2 # Dropout 비율 선택
window_size = 12

# 의도한 대로 맞도록 1 빼주기
outer_n_splits -= 1
inner_n_splits -= 1

########################################################

# 데이터 로드 및 전처리
data = pd.read_csv("202304~202307 온실데이터.csv")

plt.rcParams['font.family'] = 'Malgun Gothic'

data['datetime'] = pd.to_datetime(data['날짜'] + ' ' + data['시간'], dayfirst=True)
data.set_index('datetime', inplace=True)
data.drop(['날짜', '시간'], axis=1, inplace=True)

full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='5T')
data = data.reindex(full_range).interpolate(method='time')

# 이상치 처리 함수
def rolling_avg(data, columns, window_size=12):
    for column in columns:
        if column == '누적광량(J/cm²)':
            continue

        feature = data[column]
        rollingavg = feature.rolling(window=window_size, min_periods=1).mean()
        residuals = feature - rollingavg
        threshold = 2 * residuals.std()
        outliers = abs(residuals) > threshold
        feature[outliers] = np.nan
        data[column] = feature.interpolate(method='time')

features = ['외부온도(℃)', '내부온도(℃)', '누적광량(J/cm²)', '광량(W/m²)', '풍향', '풍속(m/s)', '풍하측 천창 개폐도(%)', '풍상측 천창 개폐도(%)']
rolling_avg(data, features, window_size=window_size)

data['누적광량(J/cm²)_diff'] = data['누적광량(J/cm²)'].diff()
data.loc[(data['누적광량(J/cm²)_diff'] < 0) & (data['누적광량(J/cm²)'] != 0), '누적광량(J/cm²)'] = 0
data.drop(['누적광량(J/cm²)_diff'], axis=1, inplace=True)

data = data.interpolate(method='time')
data.ffill(inplace=True)
data.bfill(inplace=True)

# 온도 예측 모델 로드 및 내부 온도 예측 수행
temp_model = load_model('temp_model.h5')
temp_scaler_X = joblib.load('temp_scaler_X.pkl')
temp_scaler_Y = joblib.load('temp_scaler_Y.pkl')

X_temp = data[features]
X_temp_scaled = temp_scaler_X.transform(X_temp)

def create_temp_dataset(X, time_steps=12):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

X_temp_seq = create_temp_dataset(X_temp_scaled, time_steps)

temp_predictions_scaled = temp_model.predict(X_temp_seq)
temp_predictions = temp_scaler_Y.inverse_transform(temp_predictions_scaled)

# 예측된 내부 온도를 데이터프레임에 추가
time_intervals = [5, 10, 15, 30, 45, 60]
for i, minutes in enumerate(time_intervals):
    col_name = f'예측된 내부온도(℃)_{minutes}min'
    predicted_temp = pd.Series(temp_predictions[:, i], index=data.index[time_steps:time_steps + len(temp_predictions)])
    data[col_name] = data['내부온도(℃)'].shift(-minutes // 5)
    data[col_name].iloc[time_steps-1:-1] = predicted_temp

# 양 끝 결측치 처리
for col_name in [f'예측된 내부온도(℃)_{minutes}min' for minutes in time_intervals]:
    data[col_name] = data[col_name].interpolate(method='time')  # 시간 보간법 사용
    data[col_name] = data[col_name].fillna(method='bfill')  # backward fill로 시작 부분 결측치 처리
    data[col_name] = data[col_name].fillna(method='ffill')  # forward fill로 끝 부분 결측치 처리

# 인덱스를 열로 변환하여 CSV 파일로 저장 (UTF-8 인코딩)
output_file = "updated_data_with_predicted_temp.csv"
data_with_index = data.reset_index()
data_with_index.to_csv(output_file, encoding='utf-8-sig', index=False)

# 개폐도 예측 모델의 입력 데이터 생성
features_with_pred_temp = features + [f'예측된 내부온도(℃)_{minutes}min' for minutes in time_intervals]
X_vent = data[features_with_pred_temp]
Y_vent = data[['풍하측 천창 개폐도(%)', '풍상측 천창 개폐도(%)']]

vent_scaler_X = MinMaxScaler()
vent_scaler_Y = MinMaxScaler()
X_vent_scaled = vent_scaler_X.fit_transform(X_vent)
Y_vent_scaled = vent_scaler_Y.fit_transform(Y_vent)

# 데이터셋 생성 함수
def create_dataset(X, Y, time_steps=12):
    Xs, Ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        Ys.append(Y[i + time_steps])
    return np.array(Xs), np.array(Ys)

X_vent_seq, Y_vent_seq = create_dataset(X_vent_scaled, Y_vent_scaled, time_steps)

# 개폐도 예측 모델 정의 및 훈련
def create_vent_model():
    model = Sequential([
        LSTM(lstm_neurons_1, return_sequences=True, activation='linear'),
        Dropout(dropout_rate),
        LSTM(lstm_neurons_2, return_sequences=False, activation='linear'),
        Dropout(dropout_rate),
        Dense(Y_vent_seq.shape[1])
    ])
    optimizer = optimizer_class(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

vent_model = create_vent_model()

# TimeSeriesSplit 설정 및 훈련
outer_tscv = TimeSeriesSplit(n_splits=outer_n_splits)
mse_scores_vent = []
mae_scores_vent = []
r2_scores_vent = []

for fold, (train_index, test_index) in enumerate(outer_tscv.split(X_vent_seq)):
    X_train, X_test = X_vent_seq[train_index], X_vent_seq[test_index]
    Y_train, Y_test = Y_vent_seq[train_index], Y_vent_seq[test_index]

    inner_tscv = TimeSeriesSplit(n_splits=inner_n_splits)
    for inner_train_index, inner_val_index in inner_tscv.split(X_train):
        X_inner_train, X_inner_val = X_train[inner_train_index], X_train[inner_val_index]
        Y_inner_train, Y_inner_val = Y_train[inner_train_index], Y_train[inner_val_index]

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = vent_model.fit(X_inner_train, Y_inner_train, epochs=50, batch_size=batch_size,
                                  validation_data=(X_inner_val, Y_inner_val), callbacks=[early_stopping], verbose=1)

        if np.isnan(history.history['loss']).any() or np.isnan(history.history['val_loss']).any():
            raise ValueError("NaN loss encountered during training")

    # 외부 검증을 위한 모델 평가
    predictions = vent_model.predict(X_test)
    predictions_rescaled = vent_scaler_Y.inverse_transform(predictions)
    Y_test_rescaled = vent_scaler_Y.inverse_transform(Y_test)

    actual = Y_test_rescaled[:, 0]
    pred = predictions_rescaled[:, 0]

    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    mse_scores_vent.append(mse)
    mae_scores_vent.append(mae)
    r2_scores_vent.append(r2)

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index[-len(actual):], actual, label='Actual', color='blue')
    ax.plot(data.index[-len(pred):], pred, label='Predicted', color='red')
    ax.set_title(f'5 Minutes After: Predict vs Actual')
    ax.set_ylabel('Ventilation Opening (%)')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
    plt.xticks(rotation=45)
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f'prediction_vs_actual_vent_plots_fold_{fold + 1}_5min.png')
    plt.close(fig)

# 최종 성능 지표 계산
mean_mse_vent = np.mean(mse_scores_vent)
mean_rmse_vent = sqrt(mean_mse_vent)
mean_mae_vent = np.mean(mae_scores_vent)
mean_r2_vent = np.mean(r2_scores_vent)

# 성능 지표를 파일로 저장
with open('vent_model_performance.txt', 'w') as f:
    f.write(f"time_steps: {time_steps}\n")
    f.write(f"outer_n_splits: {outer_n_splits}\n")
    f.write(f"inner_n_splits: {inner_n_splits}\n")
    f.write(f"learning_rate: {learning_rate}\n")
    f.write(f"optimizer: {optimizer_choice}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"lstm_neurons_1: {lstm_neurons_1}\n")
    f.write(f"lstm_neurons_2: {lstm_neurons_2}\n")
    f.write(f"dropout_rate: {dropout_rate}\n")
    f.write(f"window_size: {window_size}\n")
    f.write(f"scaler X : min-max\n")
    f.write(f"scaler Y : min-max\n")
    f.write(f"activation: linear\n")
    f.write(f"no He 가중치 초기값\n")
    f.write("\n5 Minutes After:\n")
    f.write(f"  Mean MSE: {mean_mse_vent}\n")
    f.write(f"  Mean RMSE: {mean_rmse_vent}\n")
    f.write(f"  Mean MAE: {mean_mae_vent}\n")
    f.write(f"  Mean R2: {mean_r2_vent}\n")

# 모델 저장
vent_model.save('vent_model.h5')

# 스케일러 저장
joblib.dump(vent_scaler_X, 'vent_scaler_X.pkl')
joblib.dump(vent_scaler_Y, 'vent_scaler_Y.pkl')