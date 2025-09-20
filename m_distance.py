"""Demonstrate how to compute the Mahalanobis distance without external libraries.

The example builds a small two-dimensional dataset, derives the mean vector and
covariance matrix, and then uses the Mahalanobis metric to quantify how unusual
some candidate points are relative to the sample distribution.
"""

from __future__ import annotations

from math import atan2, degrees, sqrt
from typing import Iterable, List, Sequence

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def mean_vector(samples: Iterable[Vector]) -> List[float]:
    """サンプル集合の平均ベクトルを算出し、各次元の中心値を返す。"""
    samples = list(samples)
    if not samples:
        raise ValueError("at least one sample is required")

    dimension = len(samples[0])
    # サンプルの各次元ごとに合計値を保持し、あとで平均へ正規化する
    totals = [0.0] * dimension
    for row in samples:
        if len(row) != dimension:
            raise ValueError("all samples must share the same dimension")
        for i, value in enumerate(row):
            totals[i] += float(value)

    count = float(len(samples))
    return [value / count for value in totals]


def covariance_matrix(samples: Iterable[Vector]) -> List[List[float]]:
    """平均との差分から共分散行列を構築し、相関と分散を表現する。"""
    samples = list(samples)
    if len(samples) < 2:
        raise ValueError("at least two samples are required to compute covariance")

    mean = mean_vector(samples)
    dimension = len(mean)
    cov = [[0.0 for _ in range(dimension)] for _ in range(dimension)]

    for row in samples:
        # 各ベクトルから平均を引いて偏差を算出し、外積で共分散を累積
        diff = [row[i] - mean[i] for i in range(dimension)]
        for i in range(dimension):
            for j in range(dimension):
                cov[i][j] += diff[i] * diff[j]

    scale = 1.0 / (len(samples) - 1)
    for i in range(dimension):
        for j in range(dimension):
            cov[i][j] *= scale

    return cov


def invert_2x2(matrix: Matrix) -> List[List[float]]:
    """2x2 行列の解析的な逆行列を計算し、マハラノビス距離計算に備える。"""
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        raise ValueError("this helper only inverts 2x2 matrices")

    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    if det == 0:
        raise ValueError("matrix is singular and cannot be inverted")

    # 余因子行列を用いた 2x2 の解析的逆行列計算
    inv_det = 1.0 / det
    return [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det],
    ]


def mahalanobis_distance(x: Vector, mean: Vector, covariance: Matrix) -> float:
    """ベクトルと分布の中心・共分散からマハラノビス距離を算出する。"""
    if len(x) != len(mean):
        raise ValueError("vector and mean must have the same dimension")

    inv_cov = invert_2x2(covariance)
    diff = [x[i] - mean[i] for i in range(len(mean))]

    # Perform diff^T * inv_cov * diff without using NumPy.
    # 2x2 行列とベクトルの積を段階的に実装し、平方距離を算出
    temp = [
        inv_cov[row][0] * diff[0] + inv_cov[row][1] * diff[1]
        for row in range(2)
    ]
    squared_distance = diff[0] * temp[0] + diff[1] * temp[1]
    return sqrt(squared_distance)


def ellipse_parameters(covariance: Matrix, scale: float = 1.0) -> tuple[float, float, float]:
    """共分散行列の固有値から楕円の幅・高さ・回転角を計算する。"""

    if len(covariance) != 2 or len(covariance[0]) != 2 or len(covariance[1]) != 2:
        raise ValueError("this helper expects a 2x2 covariance matrix")

    a, b = covariance[0]
    c, d = covariance[1]
    # Treat covariance as symmetric; average the off-diagonal terms if needed.
    off_diag = 0.5 * (b + c)

    trace = a + d
    diff = a - d
    # 2x2 共分散行列の固有値を解析的に求めて長軸・短軸を決定
    discriminant = sqrt(max(diff * diff + 4.0 * off_diag * off_diag, 0.0))

    lambda1 = max((trace + discriminant) / 2.0, 0.0)
    lambda2 = max((trace - discriminant) / 2.0, 0.0)

    angle = 0.5 * atan2(2.0 * off_diag, diff)

    width = 2.0 * scale * sqrt(lambda1)
    height = 2.0 * scale * sqrt(lambda2)
    return width, height, degrees(angle)


def plot_data(
    samples: Iterable[Vector],
    mean: Vector,
    covariance: Matrix,
    candidates: dict[str, Vector],
    distances: dict[str, float],
) -> None:
    """サンプルや候補点を散布図で描画し、距離情報を視覚化する。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except Exception as exc:  # pragma: no cover - visualization helper
        print("matplotlib を読み込めなかったためグラフ表示をスキップします:", exc)
        return

    sample_x = [row[0] for row in samples]
    sample_y = [row[1] for row in samples]

    fig, ax = plt.subplots()
    # 元データ群を青い散布図として描画
    ax.scatter(sample_x, sample_y, color="tab:blue", label="samples")

    for label, point in candidates.items():
        # 候補点を×印で追加し、距離のラベルを横に表示
        ax.scatter(point[0], point[1], marker="x", s=80, label=label)
        ax.text(
            point[0] + 0.5,
            point[1] + 0.5,
            f"{label}\nD={distances[label]:.2f}",
            fontsize=9,
        )

    ax.scatter(mean[0], mean[1], color="black", marker="o", s=60, label="mean")

    try:
        # 共分散行列から距離一定の楕円を計算して視覚的なスケールを提供
        width, height, angle = ellipse_parameters(covariance, scale=2.0)
        ellipse = Ellipse(
            (mean[0], mean[1]),
            width=width,
            height=height,
            angle=angle,
            edgecolor="tab:orange",
            facecolor="none",
            lw=2,
            label="Mahalanobis d=2",
        )
        ax.add_patch(ellipse)
    except ValueError as exc:  # pragma: no cover - defensive guard
        print("楕円描画に失敗しました:", exc)

    ax.set_title("Mahalanobis distance example")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """デモ用データを構築し、距離計算と可視化を実行する。"""
    # Construct a small correlated dataset (height, weight) in arbitrary units.
    samples = [
        (170.0, 65.0),
        (168.0, 63.0),
        (172.0, 70.0),
        (165.0, 59.0),
        (175.0, 72.0),
    ]

    mean = mean_vector(samples)
    cov = covariance_matrix(samples)

    candidates = {
        "typical": (171.0, 66.0),
        "edge_case": (160.0, 80.0),
        "outlier": (185.0, 50.0),
    }

    print("Mean vector:", mean)
    print("Covariance matrix:")
    for row in cov:
        print("  ", row)
    print()

    distances: dict[str, float] = {}
    for label, point in candidates.items():
        # 各候補点に対するマハラノビス距離を算出し、表示用に蓄積
        distance = mahalanobis_distance(point, mean, cov)
        distances[label] = distance
        print(f"{label:>10s} -> Mahalanobis distance: {distance:.3f}")

    plot_data(samples, mean, cov, candidates, distances)


if __name__ == "__main__":
    main()
