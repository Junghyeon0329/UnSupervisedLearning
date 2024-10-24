import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# 데이터 스케일링
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


plt.rcParams['font.family'] = 'sans-serif'  # 기본 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시를 위한 설정

# 엘보우 방법 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 10), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Cluster count')
plt.ylabel('Inertia')

# 실루엣 점수 시각화
plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_avgs, marker='o', color='orange')
plt.title('silhouette_score')
plt.xlabel('Cluster count')
plt.ylabel('silhouette_score')

plt.tight_layout()
plt.show()

# 최적의 k로 KMeans 실행 (엘보우 그래프를 참고하여 선택)
optimal_k = 3  # 최적의 k 값을 선택
kmeans = KMeans(n_clusters=optimal_k, random_state=7, init='k-means++')
kmeans.fit(X_scaled)

# 최종 모델의 실루엣 점수
final_silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f'최적 k={optimal_k}의 최종 실루엣 점수: {final_silhouette_avg}')


