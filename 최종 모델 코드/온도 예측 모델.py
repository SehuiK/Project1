# -*- coding: utf-8 -*-
""" 기초 모델 작성
1. 설치 라이브러리 : pandas, matplotlib, keras, sklearn, tensorflow (저는 이것만 설치해도 돌아가긴 했는데, 안되면 더 설치해보시는 것을 추천합니다)
참고로 빨간줄 있어도 저는 잘돌아가긴 했습니다.
2. 프로젝트 실행 경로에 '202304~202307 온실데이터'를 넣을 것
3. 본인이 원하는 환경에 따라 아래 있는 변수를 조정할 것
4. 실행 완료 후 만들어진 txt 파일과 모델을 공유할 것 (파일로 만들어서 공유해 주세요)
5. 참고로 실행 도중에 loss 부분에 nan이 나오는 경우는 그냥 데이터가 부족한 거니까 중단하고 그 부분은 넘어가 주세요!
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from math import sqrt
import joblib

########################### 변수 설정 #############################
time_steps_list = [3, 6, 12, 24, 48, 96, 288]
outer_n_splits_options = {
    4: [4, 28, 56],
    8: [4, 14, 28],
    16: [7, 14]
}
learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
optimizer_options = {
    'adam': Adam,
    'rmsprop': RMSprop
}
batch_size_list = [16, 32, 64, 128]
lstm_neurons_1_list = [64, 128, 256]
lstm_neurons_2_list = [32, 64, 128]
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
lstm_neurons_1 = 128
lstm_neurons_2 = 64
dropout_rate = 0.2  # Dropout 비율 선택
window_size = 12

########################################################

data = pd.read_csv("202304~202307 온실데이터.csv")

plt.rcParams['font.family'] = 'Malgun Gothic'

# 의도한 대로 맞도록 1 빼주기
outer_n_splits -= 1
inner_n_splits -= 1

####  데이터 전처리 ####

data['datetime'] = pd.to_datetime(data['날짜'] + ' ' + data['시간'], dayfirst=True)
data.set_index('datetime', inplace=True)
data.drop(['날짜', '시간'], axis=1, inplace=True)

# 누락된 데이터를 시간 보간
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

#### 내부 온도 예측 모델 ####

# 시간 보간
data = data.interpolate(method='time')
data.ffill(inplace=True)
data.bfill(inplace=True)

# 타겟 변수 생성
for minutes in [5, 10, 15, 30, 45, 60]:
    shift_steps = minutes // 5
    data[f'내부온도(℃)_{minutes}min'] = data['내부온도(℃)'].shift(-shift_steps + 1)

data.dropna(inplace=True)  # 최종적인 결측치 제거
X = data[features]
Y = data[[f'내부온도(℃)_{m}min' for m in [5, 10, 15, 30, 45, 60]]]

# 스케일링
temp_scaler_X = StandardScaler()
temp_scaler_Y = StandardScaler()
X_scaled = temp_scaler_X.fit_transform(X)
Y_scaled = temp_scaler_Y.fit_transform(Y)


# 데이터셋 생성 함수
def create_dataset(X, Y, time_steps=12):
    Xs, Ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        Ys.append(Y[i + time_steps])
    return np.array(Xs), np.array(Ys)


# 시퀀스 데이터셋 생성
X_seq, Y_seq = create_dataset(X_scaled, Y_scaled, time_steps)

# 시퀀스 데이터셋 크기 확인
print(f'X_seq shape: {X_seq.shape}')
print(f'Y_seq shape: {Y_seq.shape}')


# 모델 정의 함수
def create_model():
    model = Sequential([
        LSTM(lstm_neurons_1, return_sequences=True, activation='linear'),
        Dropout(dropout_rate),
        LSTM(lstm_neurons_2, return_sequences=False, activation='linear'),
        Dropout(dropout_rate),
        Dense(Y_seq.shape[1])
    ])
    optimizer = optimizer_class(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# TimeSeriesSplit 설정
outer_tscv = TimeSeriesSplit(n_splits=outer_n_splits)
mse_scores = {5: [], 10: [], 15: [], 30: [], 45: [], 60: []}
mae_scores = {5: [], 10: [], 15: [], 30: [], 45: [], 60: []}
r2_scores = {5: [], 10: [], 15: [], 30: [], 45: [], 60: []}

# 외부 TimeSeriesSplit 루프
for fold, (train_index, test_index) in enumerate(outer_tscv.split(X_seq)):
    X_train, X_test = X_seq[train_index], X_seq[test_index]
    Y_train, Y_test = Y_seq[train_index], Y_seq[test_index]

    # 내부 TimeSeriesSplit 설정
    inner_tscv = TimeSeriesSplit(n_splits=inner_n_splits)
    for inner_train_index, inner_val_index in inner_tscv.split(X_train):
        X_inner_train, X_inner_val = X_train[inner_train_index], X_train[inner_val_index]
        Y_inner_train, Y_inner_val = Y_train[inner_train_index], Y_train[inner_val_index]

        # 모델 정의 및 컴파일
        final_model = create_model()

        # EarlyStopping 콜백 설정
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # 모델 훈련
        history = final_model.fit(X_inner_train, Y_inner_train, epochs=50, batch_size=batch_size,
                                  validation_data=(X_inner_val, Y_inner_val), callbacks=[early_stopping], verbose=1)

        # 손실 값 확인
        print(f'History loss: {history.history["loss"]}')
        print(f'History val_loss: {history.history["val_loss"]}')

        # NaN 값이 발생하는지 확인
        if np.isnan(history.history['loss']).any() or np.isnan(history.history['val_loss']).any():
            raise ValueError("NaN loss encountered during training")

    # 외부 검증을 위한 모델 평가
    predictions = final_model.predict(X_test)
    predictions_rescaled = temp_scaler_Y.inverse_transform(predictions)
    Y_test_rescaled = temp_scaler_Y.inverse_transform(Y_test)

    # 각 시간 간격에 대한 성능 지표 계산
    time_intervals = [5, 10, 15, 30, 45, 60]
    indices = [0, 1, 2, 3, 4, 5]  # 각 시간 간격에 대응하는 인덱스

    for i, minutes in enumerate(time_intervals):
        index = indices[i]
        actual = Y_test_rescaled[:, index]
        pred = predictions_rescaled[:, index]

        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        mse_scores[minutes].append(mse)
        mae_scores[minutes].append(mae)
        r2_scores[minutes].append(r2)

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index[-len(actual):], actual, label='Actual', color='blue')
        ax.plot(data.index[-len(pred):], pred, label='Predicted', color='red')
        ax.set_title(f'{minutes} Minutes After: Predict vs Actual')
        ax.set_ylabel('Temperature')
        ax.legend()
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
        plt.xticks(rotation=45)
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(f'prediction_vs_actual_plots_fold_{fold + 1}_{minutes}min.png')  # 각 폴드의 그래프를 이미지 파일로 저장
        plt.close(fig)  # 저장 후 창을 닫음

# 최종 성능 지표 계산
mean_mse = {minutes: np.mean(mse_scores[minutes]) for minutes in time_intervals}
mean_rmse = {minutes: sqrt(mean_mse[minutes]) for minutes in time_intervals}
mean_mae = {minutes: np.mean(mae_scores[minutes]) for minutes in time_intervals}
mean_r2 = {minutes: np.mean(r2_scores[minutes]) for minutes in time_intervals}

# 성능 지표를 파일로 저장
with open('temp_model_performance.txt', 'w') as f:
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
    f.write(f"scaler X : minmax\n")
    f.write(f"scaler Y : minmax\n")
    f.write(f"activation: linear\n")
    f.write(f"no He 가중치 초기값\n")
    for minutes in time_intervals:
        f.write(f"\n{minutes} Minutes After:\n")
        f.write(f"  Mean MSE: {mean_mse[minutes]}\n")
        f.write(f"  Mean RMSE: {mean_rmse[minutes]}\n")
        f.write(f"  Mean MAE: {mean_mae[minutes]}\n")
        f.write(f"  Mean R2: {mean_r2[minutes]}\n")

# 모델 저장
final_model.save('temp_model.h5')

# 스케일러 저장
joblib.dump(temp_scaler_X, 'temp_scaler_X.pkl')
joblib.dump(temp_scaler_Y, 'temp_scaler_Y.pkl')
