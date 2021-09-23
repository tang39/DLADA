# reference: Liu, Y. and Durlofsky, L.J., 2020. 3D CNN-PCA: A Deep-Learning-Based. Parameterization for Complex Geomodels. arXiv preprint arXiv:2007.08478
import numpy as np
import matplotlib.pyplot as plt


class PCA(object):
    def __init__(self, nc=1, nr=1, l=1):
        self.l = l
        self.nc = nc
        self.nr = nr
        self.xm = np.zeros((nc, 1))
        self.usig = np.zeros((nc, l))
        self.data_matrix = None
        self.sig = None
        self.u = None

    def construct_pca(self, x):
        assert x.shape == (self.nc, self.nr)
        self.data_matrix = x
        self.xm = np.mean(x, axis=1)[:, None]
        y = 1. / (np.sqrt(float(self.nr - 1.))) * (x - self.xm)
        self.u, self.sig, _ = np.linalg.svd(y, full_matrices=False)
        self.u = self.u[:, :self.l]
        self.sig = self.sig[:self.l, None]
        self.usig = np.dot(self.u, np.diag(self.sig[:, 0]))

    def generate_pca_realization(self, xi, dim=None):
        if dim is None:
            assert xi.shape[0] == self.l
            if xi.shape == (self.l, ):
                xi = xi[:, None]
            return self.usig.dot(xi) + self.xm
        else:
            assert xi.shape[0] == dim
            if xi.shape == (dim, ):
                xi = xi[:, None]
            return self.usig[:, :dim].dot(xi) + self.xm

    def get_xi(self, m, dim=None):
        assert self.u is not None, "Input or calculate U matrix to obtain reconstructed xi"
        assert m.shape[0] == self.nc

        if m.shape == (self.nc, ):
            m = m[:, None]
        if dim is None:
            xi = self.u.T.dot(m - self.xm) / self.sig
        else:
            xi = self.u[:, :dim].T.dot(m - self.xm) / self.sig[:dim]
        return xi

    def energy_plot(self, rel_energy, truncate_point):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(rel_energy)
        plt.ylabel('Relative Energy', fontsize=12)
        plt.xlabel('Number of principal components', fontsize=12)
        plt.subplot(1, 2, 2)
        plt.plot(rel_energy[:truncate_point])
        plt.ylabel('Relative Energy', fontsize=12)
        plt.xlabel('Number of principal components', fontsize=12)
        plt.tight_layout()
        plt.show()

    def princ_component(self, tol=0.9):
        cum_energy = np.cumsum(self.sig**2)
        rel_energy = cum_energy / cum_energy[-1]
        truncate_point = np.argmin(np.abs(rel_energy - tol)) + 1
        print('Principle components: ', truncate_point)
        return truncate_point, rel_energy
