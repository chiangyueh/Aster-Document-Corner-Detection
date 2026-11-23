                   
import numpy as np


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def corners_inside(corners, margin=0.0):
    c = np.asarray(corners).reshape(4, 2)
    return np.all((c >= -margin) & (c <= 1.0 + margin), axis=1)


def quad_area(pts):
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex(pts):
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    cross = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    cross = np.array(cross)
    return bool(np.all(cross > 0) or np.all(cross < 0))
