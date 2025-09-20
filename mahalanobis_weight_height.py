import numpy as np              #numpyは数値計算ライブラリ
import pandas as pd             #pandasはデータ分析ライブラリ
import matplotlib.pyplot as plt #matplotlibはグラフ描画ライブラリ
#from scipy.spatial import distance

# Davis Dataset
df = pd.read_csv('Davis.csv') #csv(text data)を構造化dataへ変換
print(df.head())              #データの先頭5行を表示
print(df.columns)             #データの列名を表示
df = df[['weight', 'height']] #weightとheightの列を抽出
data = df.to_numpy()          #データをnumpy配列へ変換
print('data',data.shape)      #行列サイズを表示
print('data.T',data.T.shape)  #転置行列のサイズを表示
print(data.T[0])              #転置行列の0列目を表示

# 統計情報の集計
## mean
mu_mat = np.mean(data, axis=0) #平均値を計算 axis=0:列 / axis=1:行 ごとの平均値を計算
print(mu_mat)                  #平均値を表示

## data - mean
data_m_mat = data - mu_mat #データから平均値を引く

## covariance matrix
cov_mat = np.cov(data.T)            #共分散行列を計算
cov_i_mat = np.linalg.pinv(cov_mat) #共分散行列の逆行列を計算
print(cov_mat)                      #共分散行列を表示
print(cov_i_mat)                    #共分散行列の逆行列を表示

## マハラノビス距離を2通りの方法で計算
mahala_result1 = np.sqrt(np.sum(np.dot(data_m_mat,cov_i_mat)*data_m_mat,axis = 1))    #マハラノビス距離を計算
print(mahala_result1) #マハラノビス距離を表示
mahala_result2 = np.sqrt(np.einsum('ij,jk,ik->i', data_m_mat, cov_i_mat, data_m_mat)) #マハラノビス距離を計算
print(mahala_result2) #マハラノビス距離を表示 


# --- 1) グリッド作成（weight: 30-200, height: 50-210）---
xs = np.linspace(30, 200, 200)   # x 軸（weight）
ys = np.linspace(50, 210, 200)   # y 軸（height）
X, Y = np.meshgrid(xs, ys)       # 形状 (ny, nx)


# --- 2) グリッド上でマハラノビス距離を計算（ベクトル化）---
# diff[y, x, :] = [X[y,x]-μx, Y[y,x]-μy]
diff = np.stack([X - mu_mat[0], Y - mu_mat[1]], axis=-1)     # (ny, nx, 2)
D2   = np.einsum('...i,ij,...j->...', diff, cov_i_mat, diff) # (ny, nx)
D    = np.sqrt(D2)                                     # (ny, nx)


# Plot
fig1 = plt.figure(figsize=(8,5))  #グラフのサイズを設定
ax1 = fig1.add_subplot()          #グラフの軸を設定
CS = ax1.contour(X, Y, D)         #等高線を表示
ax1.clabel(CS)                    #等高線のラベルを表示
ax1.scatter(data.T[0], data.T[1]) #データを散布図で表示
ax1.set_title('Davis Data')       #グラフのタイトルを設定
ax1.set_xlabel('weight')          #グラフの軸ラベルを設定
ax1.set_ylabel('height')          #グラフの軸ラベルを設定
ax1.set_aspect('equal')           #グラフのアスペクト比を設定
ax1.grid()                        #グラフのグリッドを表示 
plt.show()                        #グラフを表示