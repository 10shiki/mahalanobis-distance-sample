"""複数のグラフを並べて描画するプログラム"""
import numpy as np
import matplotlib.pyplot as plt

# figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
fig = plt.figure()

# add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

t = np.linspace(-10, 10, 1000)
y1 = np.sin(t)
y2 = np.cos(t) 
y3 = np.abs(np.sin(t))
y4 = np.sin(t)**2

c1,c2,c3,c4 = "blue","green","red","black"      # 各プロットの色
l1,l2,l3,l4 = "sin","cos","abs(sin)","sin**2"   # 各ラベル

ax1.plot(t, y1, color=c1, label=l1)
ax2.plot(t, y2, color=c2, label=l2)
ax3.plot(t, y3, color=c3, label=l3)
ax4.plot(t, y4, color=c4, label=l4)
ax1.legend(loc = 'upper right') #凡例
ax2.legend(loc = 'upper right') #凡例
ax3.legend(loc = 'upper right') #凡例
ax4.legend(loc = 'upper right') #凡例
fig.tight_layout()              #レイアウトの設定
plt.show()
