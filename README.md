# Mahalanobis Distance Demo

純粋な Python だけでマハラノビス距離を計算するチュートリアルです。numpy 等を使わずに平均ベクトル、共分散行列、逆行列、距離の算出までを手作業で追えるようにしています。`m_distance.py` が CLI エントリーポイントで、matplotlib が導入されている場合は散布図と楕円による可視化も行います。

## モジュールのポイント
- `mean_vector(samples)` — 入力サンプルの平均ベクトルを算出。次元不一致を検出して ValueError を送出。
- `covariance_matrix(samples)` — 偏差の外積から 2×2 の共分散行列を構築。サンプル不足を検出。
- `invert_2x2(matrix)` — 解析的に 2×2 行列の逆行列を計算し、特異行列は拒否。
- `mahalanobis_distance(x, mean, covariance)` — diff^T Σ^{-1} diff を Python のみで評価。
- `plot_data(...)` — matplotlib が利用可能なときにサンプル・候補点・距離楕円を描画（未導入ならメッセージを出してスキップ）。

メインの `main()` では身長・体重に相当する 5 サンプルを定義し、`typical`/`edge_case`/`outlier` の 3 点に対する距離を計算します。`ellipse_parameters` は楕円の幅・高さ・回転角を算出し、距離感を視覚化するために利用されています。

## セットアップ
1. Python 3.11 で作業することを推奨します。
2. 仮想環境（任意）:
   ```
   python -m venv .venv
   source .venv/bin/activate  # Windows では .venv\Scripts\activate
   ```
3. Matplotlib は可視化を使いたい場合のみインストール:
   ```
   pip install matplotlib
   ```

## 実行方法
`m_distance.py` を直接実行します。`python` と `python3` のどちらが利用可能かは環境に合わせて選択してください。
```
python3 m_distance.py
```

平均ベクトルと共分散行列、各候補点のマハラノビス距離が表示されます。matplotlib が無い場合は次のようなメッセージが出て描画がスキップされます。
```
Mean vector: [170.0, 65.8]
Covariance matrix:
   [14.5, 19.75]
   [19.75, 27.699999999999996]

   typical -> Mahalanobis distance: 1.326
 edge_case -> Mahalanobis distance: 31.232
   outlier -> Mahalanobis distance: 40.720
matplotlib を読み込めなかったためグラフ表示をスキップします: No module named 'matplotlib'
```

## テストと検証
- 構文チェック:
  ```
  python -m compileall m_distance.py
  ```
- 将来的にテストを追加する場合は `tests/` 配下に配置し、`pytest` で実行します。ルートをインポートする際は `PYTHONPATH=.` を設定してください。
- 数値系の関数にはサンプル不足、次元不一致、特異行列といったエッジケースを含むパラメータ化テストを用意することを想定しています。

## ファイル構成
```
.
├── AGENTS.md
├── Davis.csv                         # Davis の身長体重データ（探索用）
├── README.md
├── m_distance.py                     # 純粋 Python のマハラノビス距離デモ
├── mahalanobis_weight_height.py      # NumPy/Pandas を使った旧バージョン
├── mahalanobis_weight_height_20250910-A.py
├── mahalanobis_weight_height_68-95-99 copy.py
└── practice.py                       # 探索的なスクリプト
```
`m_distance.py` が最新の公式サンプルです。その他のスクリプトや CSV は実験用リファレンスとして残しています。

## コントリビュート
変更する際は `AGENTS.md` に記載されたガイドライン（コーディングスタイル、テスト基準、PR 手順）を確認してください。新しい数値ヘルパーを追加する場合は副作用を避け、可視化補助は必要に応じて `plot_data` にまとめてください。
