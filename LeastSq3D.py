import numpy as np
import matplotlib.pyplot as plt

class LeastSq3D:
    def __init__(self):
        self.coeficient = None
    
    def fit(self, pts):
        m11 = np.sum(pts[:, 0] * pts[:, 0])
        m12 = np.sum(pts[:, 0] * pts[:, 1])
        m13 = np.sum(pts[:, 0])
        m21 = m12
        m22 = np.sum(pts[:, 1] * pts[:, 1])
        m23 = np.sum(pts[:, 1])
        m31 = m13
        m32 = m23
        m33 = pts.shape[0]
        A = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])

        b1 = np.sum(pts[:, 0] * pts[:, 2])
        b2 = np.sum(pts[:, 1] * pts[:, 2])
        b3 = np.sum(pts[:, 2])
        b = np.array([b1, b2, b3])
        # self.coeficient = np.dot(np.linalg.inv(A), b)
        self.coeficient = np.linalg.solve(A, b)
    
    def predict(self, pts):
        if self.coeficient is None:
            raise Exception('You need to fit the model first.')
        pts = np.array(pts)
        return np.dot(pts, self.coeficient)

if __name__ == '__main__':
    # 載入已經讀好的點雲
    pts = np.load(r'human_body_vertices.npy')
    
    # 畫出半透明的原始點雲
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect(aspect = (1, 1, 0.25))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.1, c='goldenrod')

    # 用最小二乘法求解
    l3d = LeastSq3D()
    l3d.fit(pts)

    # 對平面點雲取樣，並畫出
    x_range = np.arange(pts[:, 0].min(), pts[:, 0].max(), pts[:, 0].ptp()/10)
    y_range = np.arange(pts[:, 1].min(), pts[:, 1].max(), pts[:, 1].ptp()/10)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = l3d.predict(np.array([xx.flatten(), yy.flatten(), np.ones(shape=len(xx.flatten()))]).T).reshape(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')
    plt.show()