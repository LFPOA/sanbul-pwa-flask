import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras


fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] + 1)

''' (1-2)
# fires.head() 출력
print("=== fires.head() ===")
print(fires.head(), "\n")

# fires.info() 출력
print("=== fires.info() ===")
fires.info()    # info() 자체가 출력해 줍니다.
print()

# fires.describe() 출력
print("=== fires.describe() ===")
print(fires.describe(), "\n")

# 범주형 특성 month, day 의 value_counts() 출력
print("=== month value_counts() ===")
print(fires['month'].value_counts(), "\n")

print("=== day value_counts() ===")
print(fires['day'].value_counts())
'''

''' (1-3)
fires.hist(column=['avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind', 'burned_area'],
           bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(fires['max_temp'], fires['burned_area'], alpha=0.5)
plt.xlabel('max_temp')
plt.ylabel('burned_area (log)')
plt.title('max_temp vs. burned_area')
plt.show()
'''

''' (1-4)
orig = fires["burned_area"]
log = np.log(orig + 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(orig, bins=50)
plt.title("burned_area")
plt.xlabel("burned_area")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.hist(log, bins=50)
plt.title("burned_area")
plt.xlabel("ln(burned_area + 1)")
plt.ylabel("")  
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
'''

## (1-5)
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
test_set.head()
fires["month"].hist()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
 strat_train_set = fires.loc[train_index]
 strat_test_set = fires.loc[test_index]
'''
print("\nMonth category proportion: \n",
 strat_test_set["month"].value_counts()/len(strat_test_set))
print("\nOverall month category proportion: \n",
 fires["month"].value_counts()/len(fires))
'''

''' (1-6)
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
data_for_plot = strat_train_set[attributes]

plt.figure(figsize=(10, 10))
scatter_matrix(data_for_plot, figsize=(10, 10), diagonal="hist", alpha=0.6)
plt.tight_layout()
plt.show()
'''

'''(1-7)
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=fires["max_temp"], label="max_temp",
 c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend(title="max_temp")
plt.title("Location vs. Burned Area")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(alpha=0.3)
plt.show()
'''

'''(1-8)
fires_cat = strat_train_set[["day", "month"]]

# --- Day 인코딩 ---
cat_day_encoder = OneHotEncoder()
day_1hot = cat_day_encoder.fit_transform(fires_cat[["day"]])

# COO 포맷으로 변환하여 (row, col) 값 출력
coo = day_1hot.tocoo()
for r, c, v in zip(coo.row, coo.col, coo.data):
    print(f"({r}, {c}) {v}")

# 카테고리 목록
print("\ncat_day_encoder.categories_:")
print(cat_day_encoder.categories_[0])  # day 카테고리 배열

# --- Month 인코딩 ---
cat_month_encoder = OneHotEncoder()
month_1hot = cat_month_encoder.fit_transform(fires_cat[["month"]])

coo2 = month_1hot.tocoo()
for r, c, v in zip(coo2.row, coo2.col, coo2.data):
    print(f"({r}, {c}) {v}")

print("\ncat_month_encoder.categories_:")
print(cat_month_encoder.categories_[0])  # month
'''
##(1-8)##
fires_features = strat_train_set.drop("burned_area", axis=1)
fires_labels   = strat_train_set["burned_area"].copy()

num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["month", "day"]

# 5) 숫자용 파이프라인
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

# 6) 전체 전처리 파이프라인
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# 7) 전처리 적용
fires_prepared = full_pipeline.fit_transform(fires_features)

##(1-9)##
X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels, test_size=0.2, random_state=42
)

fires_test_features = strat_test_set.drop("burned_area", axis=1)
fires_test_labels   = strat_test_set["burned_area"].copy()

fires_test_prepared = full_pipeline.transform(fires_test_features)

X_test, y_test = fires_test_prepared, fires_test_labels

# 재현성을 위한 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

# 모델 정의
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)  # 출력층 (회귀)
])

model.summary()

# 모델 컴파일 및 훈련
model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid)
)

# Keras 모델 저장
model.save('fires_model.keras')

# 모델 평가 (샘플 3개 예측)
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2):\n",
      np.round(model.predict(X_new), 2))