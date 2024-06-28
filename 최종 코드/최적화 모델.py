## 앞으로 수정 사항 예정
## 1. 온도 예측 병렬화
## 2. custom 손실 함수 목적에 맞도록 변경


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.optimizers import Adam, RMSprop
import warnings
# 경고창 숨기기
warnings.filterwarnings('ignore')

############### 초기 설정 및 데이터 전처리 ###################
# 초기 설정
time_steps = 12  # time_steps 변수 설정

# 모델 및 스케일러 불러오기
temp_model = load_model('temp_model.h5')
vent_model = load_model('vent_model.h5')
temp_scaler_X = joblib.load('temp_scaler_X.pkl')
temp_scaler_Y = joblib.load('temp_scaler_Y.pkl')
vent_scaler_X = joblib.load('vent_scaler_X.pkl')
vent_scaler_Y = joblib.load('vent_scaler_Y.pkl')

# 데이터 준비
data = pd.read_csv("202304~202307 온실데이터.csv")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # Minus sign 문제 해결

# 데이터 전처리 함수 설정
def preprocess_data(data, features, window_size=12):
    # 누락된 데이터를 시간 보간
    full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='5min')
    data = data.reindex(full_range).interpolate(method='time')

    # 이상치 처리 함수
    def rolling_avg(data, columns, window_size):
        for column in columns:
            feature = data[column]
            rollingavg = feature.rolling(window=window_size, min_periods=1).mean()
            residuals = feature - rollingavg
            threshold = 2 * residuals.std()
            outliers = abs(residuals) > threshold
            feature[outliers] = np.nan
            data[column] = feature.interpolate(method='time')

    # '누적광량(J/cm²)'을 제외한 features 목록 생성
    features_excluding_누적광량 = [feature for feature in features if feature != '누적광량(J/cm²)']

    rolling_avg(data, features_excluding_누적광량, window_size)

    # 누적광량(J/cm²) 컬럼 처리
    data['누적광량(J/cm²)_diff'] = data['누적광량(J/cm²)'].diff()
    data.loc[(data['누적광량(J/cm²)_diff'] < 0) & (data['누적광량(J/cm²)'] != 0), '누적광량(J/cm²)'] = 0
    data.drop(['누적광량(J/cm²)_diff'], axis=1, inplace=True)

    return data

# 전처리 시작
data['datetime'] = pd.to_datetime(data['날짜'] + ' ' + data['시간'], dayfirst=True)
data.set_index('datetime', inplace=True)
data.drop(['날짜', '시간'], axis=1, inplace=True)

features = ['외부온도(℃)', '내부온도(℃)', '누적광량(J/cm²)', '광량(W/m²)', '풍향', '풍속(m/s)', '풍하측 천창 개폐도(%)', '풍상측 천창 개폐도(%)']
data = preprocess_data(data, features, window_size=12)

################### 내부 온도 예측 #####################
# 입력 데이터 준비하기
X = data[features]

# 스케일링
X_scaled_temp = temp_scaler_X.transform(X)

# 데이터셋 생성 함수
def create_dataset(X, time_steps):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

# 데이터셋 생성
X_seq_temp = create_dataset(X_scaled_temp, time_steps)

# 내부 온도 예측
temp_predictions_scaled = temp_model.predict(X_seq_temp)
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

# 데이터 전처리 다시 수행
features_with_pred_temp = features + [f'예측된 내부온도(℃)_{minutes}min' for minutes in time_intervals]
data = preprocess_data(data, features_with_pred_temp, window_size=12)

# 인덱스를 열로 변환하여 CSV 파일로 저장 (UTF-8 인코딩)
output_file = "updated_data_with_predicted_temp.csv"
data_with_index = data.reset_index()
data_with_index.to_csv(output_file, encoding='utf-8-sig', index=False)

########################### 개폐도 최적화 모델 학습 부분 ##################################
def custom_loss(y_true, y_pred):
    # 입력에 쓰였던 X_seq_vent 복사
    X_seq_vent_copy = X_seq_vent.copy()
    X_seq_vent_copy = np.roll(X_seq_vent_copy, shift=-1, axis=1)
    X_seq_vent_copy[-1, :, :] = X_seq_vent_copy[-2, :, :]

    # y_pred를 텐서에서 numpy로 변환
    y_pred_np = y_pred.numpy()
    print(type(y_pred_np), y_pred_np.shape, X_seq_vent_copy[:, -1, 6:8].shape)
    X_seq_vent_copy[:, -1, 6:8] = y_pred_np

    # temp_model을 사용해서 예측
    X_seq_temp = vent_scaler_X.inverse_transform(X_seq_vent_copy.reshape(-1, X_seq_vent_copy.shape[-1])).reshape(X_seq_vent_copy.shape)
    X_seq_temp = X_seq_temp[:, :, :8]
    X_seq_temp = temp_scaler_X.transform(X_seq_temp.reshape(-1, X_seq_temp.shape[-1])).reshape(X_seq_temp.shape)
    print(X_seq_temp.shape)

    # 텐서플로우 모델의 predict 호출 부분을 tf.function으로 감싸기
    @tf.function
    def predict_temp_model(input_data):
        return temp_model(input_data, training=False)

    prediction = predict_temp_model(X_seq_temp)

    # numpy 배열을 텐서로 변환
    X_seq_vent_tensor = tf.convert_to_tensor(X_seq_vent, dtype=tf.float32)
    prediction_tensor = tf.convert_to_tensor(prediction, dtype=tf.float32)
    X_seq_temp_tensor = tf.convert_to_tensor(X_seq_temp, dtype=tf.float32)

    # 예측된 개폐도와 정답 개폐도(실제 값)와의 손실
    predict_penalty = tf.reduce_sum(tf.abs(y_pred - y_true))

    # 예측된 개폐도와 현재 개폐도와의 손실
    vent_change_penalty = tf.reduce_sum(tf.square(y_pred - X_seq_vent_tensor[:, -1, 6:8]))

    # 0.5도 이하로 움직였을 때의 손실 추가 (0에 가까울수록 크게)
    small_movement = tf.abs(y_pred - X_seq_vent_tensor[:, -1, 6:8])
    small_movement_penalty = tf.reduce_sum(tf.where(small_movement <= 0.1, 1 / (small_movement + 0.5), 0.0))

    # 최종 vent_change_penalty에 small_movement_penalty를 더함
    vent_change_penalty -= small_movement_penalty

    # 예측된 개폐도를 포함한 1시간 뒤 내부온도 예측과 현재의 차이가 4도 이상일 때의 차이
    prediction_unscaled = temp_scaler_Y.inverse_transform(prediction)
    X_seq_temp_unscaled = temp_scaler_X.inverse_transform(X_seq_temp[:, -1, :])
    X_diff = X_seq_temp_unscaled[:, 1] - prediction_unscaled[:, 5]
    X_diff_tensor = tf.convert_to_tensor(X_diff, dtype=tf.float32)

    one_hour_temp_diff_penalty = tf.reduce_sum(tf.maximum(tf.abs(X_diff_tensor), 4) - 4)

    # 예측된 개폐도를 포함한 5분 뒤 내부온도 예측과 현재의 차이
    five_min_temp_diff = tf.abs(X_seq_temp_tensor[:, -1, 1] - prediction_tensor[:, 0])
    five_min_temp_diff_penalty = tf.reduce_sum(five_min_temp_diff)

    # 1 1 5 10 (수)
    # 5 1 10 20 (수)
    # 3 50 1 2 (수)
    # 1 1 40 20 (정)
    # 1 1 80 10 (정)
    # 5 1 20 20 (정)

    # 변수 정의
    vt = 1
    pp = 100
    oh = 160
    fv = 10

    # 개별 손실 요소 출력
    tf.print("vent_change_penalty:", vent_change_penalty, vent_change_penalty * vt)
    tf.print("predict_penalty:", predict_penalty, predict_penalty * pp)
    tf.print("one_hour_temp_diff_penalty:", one_hour_temp_diff_penalty, one_hour_temp_diff_penalty * oh)
    tf.print("five_min_temp_diff_penalty:", five_min_temp_diff_penalty, five_min_temp_diff_penalty * fv)

    # 최종 손실 계산 및 출력
    loss = tf.reduce_mean(vent_change_penalty * vt + predict_penalty * pp + one_hour_temp_diff_penalty * oh + five_min_temp_diff_penalty * fv)

    tf.print("loss:", loss)
    return loss / (len(data) * 10)

# 데이터셋 생성 함수
def create_vent_dataset(X, Y, time_steps):
    Xs, Ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        Ys.append(Y[i + time_steps])
    return np.array(Xs), np.array(Ys)

# 모델 재학습 및 예측 수행
epochs = 10
learning_rate = 0.001

for epoch in range(epochs):
    print("now epoch :" + str(epoch))
    # 개폐도 예측 모델의 입력 데이터 생성
    features_with_pred_temp = features + [f'예측된 내부온도(℃)_{minutes}min' for minutes in time_intervals]
    X_vent = data[features_with_pred_temp]
    Y_vent = data[['풍하측 천창 개폐도(%)', '풍상측 천창 개폐도(%)']]

    # 스케일링
    X_vent_scaled = vent_scaler_X.transform(X_vent)
    Y_vent_scaled = vent_scaler_Y.transform(Y_vent)

    # 시퀀스 데이터셋 생성
    X_seq_vent, Y_seq_vent = create_vent_dataset(X_vent_scaled, Y_vent_scaled, time_steps)

    batch_size = X_seq_vent.shape[0]  # 배치 사이즈 수정

    tf.config.run_functions_eagerly(True)

    vent_model.compile(optimizer=Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: custom_loss(y_true, y_pred))
    vent_model.fit(X_seq_vent, Y_seq_vent, epochs=1, batch_size=batch_size)

    y_pred = vent_model.predict(X_seq_vent)
    print(y_pred.shape)
    print(data['풍하측 천창 개폐도(%)'].iloc[time_steps:].shape)
    y_pred_actual = vent_scaler_Y.inverse_transform(y_pred)  # 예측 값을 실제 값으로 변환

    # 개폐도 값을 0~100 범위로 맞추기
    y_pred_actual[:, 0] = y_pred_actual[:, 0].clip(0, 100)
    y_pred_actual[:, 1] = y_pred_actual[:, 1].clip(0, 100)

    data.loc[data.index[time_steps:], '풍하측 천창 개폐도(%)'] = y_pred_actual[:, 0]  # 풍하측 천창 개폐도(%) 열 대체
    data.loc[data.index[time_steps:], '풍상측 천창 개폐도(%)'] = y_pred_actual[:, 1]  # 풍상측 천창 개폐도(%) 열 대체

    tf.config.run_functions_eagerly(False)

    # 인덱스를 열로 변환하여 CSV 파일로 저장 (UTF-8 인코딩)
    output_file = "final_data.csv"
    data_with_index = data.reset_index()
    data_with_index.to_csv(output_file, encoding='utf-8-sig', index=False)

    # 내부 온도 예측 및 기존 값 갱신
    temp_features = data[features]
    for i in range(time_steps, len(temp_features)):
        if(i % 1000 == 0):
            print(i)
        input_seq = temp_scaler_X.transform(temp_features.iloc[i - time_steps:i]).reshape(1, time_steps, len(features))  # 입력 데이터 스케일링
        predicted_temp_scaled = temp_model.predict(input_seq, verbose=0)
        predicted_temp = temp_scaler_Y.inverse_transform(predicted_temp_scaled)  # 예측 값을 실제 값으로 변환
        temp_features.iloc[i, temp_features.columns.get_loc('내부온도(℃)')] = predicted_temp[0][0]  # 예측된 값으로 대체
        for j, minutes in enumerate(time_intervals):
            data.loc[data.index[i - 1], f'예측된 내부온도(℃)_{minutes}min'] = predicted_temp[0][j]

    # 데이터 전처리 다시 수행
    features_with_pred_temp = features + [f'예측된 내부온도(℃)_{minutes}min' for minutes in time_intervals]
    data = preprocess_data(data, features_with_pred_temp, window_size=12)

# 데이터 비교 및 그래프 그리기
initial_data = pd.read_csv("202304~202307 온실데이터.csv")  # 초기 데이터를 새로 불러옴
initial_data['datetime'] = pd.to_datetime(initial_data['날짜'] + ' ' + initial_data['시간'], dayfirst=True)
initial_data.set_index('datetime', inplace=True)
initial_data.drop(['날짜', '시간'], axis=1, inplace=True)

################### 그래프 및 결과 저장 ######################
days = pd.date_range(start=data.index.min().date(), end=data.index.max().date(), freq='D')

# 온도 데이터의 전체 최소값과 최대값 계산
temp_min = initial_data['내부온도(℃)'].min()
temp_max = initial_data['내부온도(℃)'].max()

for day in days:
    daily_initial_data = initial_data[day:day + pd.Timedelta(days=1)]
    daily_data = data[day:day + pd.Timedelta(days=1)]
    if not daily_data.empty and not daily_initial_data.empty:
        # 하나의 그림에 3개의 서브플롯 생성
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # 풍상측 천창 개폐도 비교 그래프
        axs[0].plot(daily_initial_data.index, daily_initial_data['풍상측 천창 개폐도(%)'], label='Initial Windward Vent',
                    color='blue')
        axs[0].plot(daily_data.index, daily_data['풍상측 천창 개폐도(%)'], label='Predicted Windward Vent', color='red')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Windward Vent Position')
        axs[0].set_ylim(0, 100)  # Y 스케일 고정
        axs[0].legend()
        axs[0].set_title(f'Windward Vent Position on {day.date()}')

        # 풍하측 천창 개폐도 비교 그래프
        axs[1].plot(daily_initial_data.index, daily_initial_data['풍하측 천창 개폐도(%)'], label='Initial Leeward Vent',
                    color='blue')
        axs[1].plot(daily_data.index, daily_data['풍하측 천창 개폐도(%)'], label='Predicted Leeward Vent', color='red')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Leeward Vent Position')
        axs[1].set_ylim(0, 100)  # Y 스케일 고정
        axs[1].legend()
        axs[1].set_title(f'Leeward Vent Position on {day.date()}')

        # 내부 온도 비교 그래프
        axs[2].plot(daily_initial_data.index, daily_initial_data['내부온도(℃)'], label='Initial Internal Temperature',
                    color='blue')
        axs[2].plot(daily_data.index, daily_data['내부온도(℃)'], label='Predicted Internal Temperature', color='red')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Internal Temperature (℃)')
        axs[2].set_ylim(temp_min, temp_max)  # Y 스케일 고정
        axs[2].legend()
        axs[2].set_title(f'Internal Temperature on {day.date()}')

        # 그림 저장
        plt.tight_layout()
        plt.savefig(f"comparison_{day.date()}.png")
        plt.close()

# 인덱스를 열로 변환하여 CSV 파일로 저장 (UTF-8 인코딩)
output_file = "final_data.csv"
data_with_index = data.reset_index()
data_with_index.to_csv(output_file, encoding='utf-8-sig', index=False)

# 최종 모델 평가 및 성능 지표 저장
vent_model.save('final_vent_model.h5')
joblib.dump(vent_scaler_X, 'final_vent_scaler_X.pkl')
joblib.dump(vent_scaler_Y, 'final_vent_scaler_Y.pkl')