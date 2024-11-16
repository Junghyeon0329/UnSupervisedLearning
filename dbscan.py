# from sklearn import datasets
# import pandas as pd
# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# import numpy as np

# # Load dataset
# iris = datasets.load_iris()
# labels = pd.DataFrame(iris.target, columns=['labels'])
# data = pd.DataFrame(iris.data, columns=['Sepal length','Sepal width','Petal length','Petal width'])
# data = pd.concat([data, labels], axis=1)

# # Feature
# feature = data[['Sepal length', 'Sepal width', 'Petal length', 'Petal width']]

# # 데이터 표준화
# scaler = StandardScaler()
# scaled_feature = scaler.fit_transform(feature)

# # NearestNeighbors로 k-distance plot 생성
# neigh = NearestNeighbors(n_neighbors=5)
# nbrs = neigh.fit(scaled_feature)
# distances, indices = nbrs.kneighbors(scaled_feature)

# # k-distance plot
# distances = np.sort(distances[:, -1], axis=0)
# plt.plot(distances)
# plt.ylabel('k-distance')
# plt.xlabel('Points sorted by distance')
# plt.title('k-distance Plot to Find eps')
# plt.show()

# # DBSCAN 모델을 최적화된 eps 값으로 학습
# # 여기서는 k-distance plot에서 '엘보우' 포인트를 찾아 eps 값을 설정합니다.
# # 예를 들어, 0.3과 같은 값으로 설정해볼 수 있습니다.
# eps_value = 0.3  # k-distance plot에서 적당한 값으로 설정하세요.
# model = DBSCAN(eps=eps_value, min_samples=5)
# predict = pd.DataFrame(model.fit_predict(scaled_feature), columns=['predict'])

# # Noise를 'Noise'로 라벨링
# r = pd.concat([pd.DataFrame(scaled_feature, columns=feature.columns), predict], axis=1)
# r['predict'] = r['predict'].map(lambda x: 'Noise' if x == -1 else f'Cluster {x}')

# # Seaborn 스타일 설정
# sns.set(style="whitegrid")

# # Pairplot with DBSCAN results
# sns.pairplot(r, hue='predict', palette='Set2')
# plt.show()

# # Pairplot with true labels
# sns.pairplot(data, hue='labels', palette='Set2')
# plt.show()

from sklearn import datasets
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the wine dataset
wine = datasets.load_wine()
labels = pd.DataFrame(wine.target, columns=['labels'])
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data = pd.concat([data, labels], axis=1)

# Feature
feature = data[wine.feature_names]

# 데이터 표준화
scaler = StandardScaler()
scaled_feature = scaler.fit_transform(feature)

# NearestNeighbors로 k-distance plot 생성
neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(scaled_feature)
distances, indices = nbrs.kneighbors(scaled_feature)

# k-distance plot
distances = np.sort(distances[:, -1], axis=0)
plt.plot(distances)
plt.ylabel('k-distance')
plt.xlabel('Points sorted by distance')
plt.title('k-distance Plot to Find eps')
plt.show()

# DBSCAN 모델을 최적화된 eps 값으로 학습
# 여기서는 k-distance plot에서 '엘보우' 포인트를 찾아 eps 값을 설정합니다.
# 예를 들어, 0.3과 같은 값으로 설정해볼 수 있습니다.
eps_value = 0.3  # k-distance plot에서 적당한 값으로 설정하세요.
model = DBSCAN(eps=eps_value, min_samples=5)
predict = pd.DataFrame(model.fit_predict(scaled_feature), columns=['predict'])

# Noise를 'Noise'로 라벨링
r = pd.concat([pd.DataFrame(scaled_feature, columns=feature.columns), predict], axis=1)
r['predict'] = r['predict'].map(lambda x: 'Noise' if x == -1 else f'Cluster {x}')

# Seaborn 스타일 설정
sns.set(style="whitegrid")

# Pairplot with DBSCAN results
sns.pairplot(r, hue='predict', palette='Set2')
plt.show()

# Pairplot with true labels (Wine 데이터셋의 실제 레이블)
sns.pairplot(data, hue='labels', palette='Set2')
plt.show()
