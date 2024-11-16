# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# # 데이터 스케일링
# # 평균 편차
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 엘보우 방법으로 최적의 k 찾기
# inertia = []
# silhouette_avgs = []

# for k in range(2, 10):
#     kmeans = KMeans(n_clusters=k, random_state=7, init='k-means++')
#     kmeans.fit(X_scaled)
    
#     inertia.append(kmeans.inertia_)
#     silhouette_avgs.append(silhouette_score(X_scaled, kmeans.labels_))
#     print(f'클러스터 수: {k} // 실루엣 점수: {silhouette_avgs[-1]}')

# optimal_k = silhouette_avgs.index(max(silhouette_avgs)) + 2  # k는 2부터 시작하므로 인덱스에 2를 더해줍니다.
# print(f'최적의 k 값: {optimal_k}')

# plt.rcParams['font.family'] = 'sans-serif'  # 기본 폰트 설정
# plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시를 위한 설정

# # 엘보우 방법 시각화
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(range(2, 10), inertia, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Cluster count')
# plt.ylabel('Inertia')

# # 실루엣 점수 시각화
# plt.subplot(1, 2, 2)
# plt.plot(range(2, 10), silhouette_avgs, marker='o', color='orange')
# plt.title('silhouette_score')
# plt.xlabel('Cluster count')
# plt.ylabel('silhouette_score')

# plt.tight_layout()
# plt.show()

# kmeans = KMeans(n_clusters=optimal_k, random_state=7, init='k-means++')
# kmeans.fit(X_scaled)

# # 최종 모델의 실루엣 점수
# final_silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
# print(f'최적 k={optimal_k}의 최종 실루엣 점수: {final_silhouette_avg}')


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Wine 데이터셋 로드
wine = datasets.load_wine()
X = wine.data
y = wine.target

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 엘보우 방법으로 최적의 k 찾기
inertia = []
silhouette_avgs = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=7, init='k-means++')
    kmeans.fit(X_scaled)
    
    inertia.append(kmeans.inertia_)
    silhouette_avgs.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f'클러스터 수: {k} // 실루엣 점수: {silhouette_avgs[-1]}')

# 최적의 k 값 찾기
optimal_k = silhouette_avgs.index(max(silhouette_avgs)) + 2  # k는 2부터 시작하므로 인덱스에 2를 더해줍니다.
print(f'최적의 k 값: {optimal_k}')

# 엘보우 방법 시각화
plt.rcParams['font.family'] = 'sans-serif'  # 기본 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시를 위한 설정

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 10), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Cluster count')
plt.ylabel('Inertia')

# 실루엣 점수 시각화
plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_avgs, marker='o', color='orange')
plt.title('Silhouette Score')
plt.xlabel('Cluster count')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# 최적의 k값을 사용하여 KMeans 클러스터링 학습
kmeans = KMeans(n_clusters=optimal_k, random_state=7, init='k-means++')
kmeans.fit(X_scaled)

# 최종 모델의 실루엣 점수 출력
final_silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f'최적 k={optimal_k}의 최종 실루엣 점수: {final_silhouette_avg}')

