import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data  # 특성 데이터
y = iris.target  # 실제 레이블

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gaussian Mixture Model (EM) 적용
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

# 클러스터 레이블 예측
labels = gmm.predict(X_scaled)

# PCA로 2D로 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100, edgecolor='k')
plt.title('Gaussian Mixture Model (EM) Clustering on Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# GMM의 클러스터 중심 (각각의 평균 값)
centroids = gmm.means_

# PCA로 축소된 데이터의 클러스터 중심
centroids_pca = pca.transform(centroids)

# 클러스터링 결과 시각화 (중심 추가)
plt.figure(figsize=(8, 6))
s
