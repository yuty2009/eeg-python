"""
Reference: 
    https://fbcsptoolbox.github.io/
"""
import numpy as np
import scipy.linalg as la


class CSP:
    def __init__(self, n_filters):
        self.n_filters = n_filters

    def fit(self, x_train, y_train):
        """fit the CSP filters
        inputs:
            x_train: (n_trials, n_samples, n_channels)
            y_train: (n_trials,)
        outputs:
            (n_channels, n_channels)
        """
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        labels = np.sort(np.unique(y_train))
        assert len(labels) == 2, 'CSP was designed for binary classification'
        n_trials, n_samples, n_channels = x_data.shape
        cov_x = np.zeros((2, n_channels, n_channels), dtype=np.float)
        for i in range(n_trials):
            x_trial = x_data[i, :, :]
            y_trial = y_labels[i]
            ci_trial = np.argwhere(labels==y_trial)
            cov_x_trial = np.matmul(np.transpose(x_trial), x_trial)
            cov_x_trial /= np.trace(cov_x_trial)
            cov_x[ci_trial, :, :] += cov_x_trial
        cov_1, cov_2 = [cov_x[ci]/np.sum(y_labels==cls) for ci, cls in enumerate(labels)]
        return self._fit_2(cov_1, cov_2)

    def _fit_1(self, R1, R2):
        R = R1 + R2
        V1, U1 = la.eig(R)
        P = np.dot(np.diag(V1**(-1/2)), U1.T)
        S1 = np.dot(np.dot(P, R1), P.T)
        # S2 = np.dot(np.dot(P, R2), P.T)
        V2, U2 = np.linalg.eig(S1)
        W = np.dot(P.T, U2)
        ind = np.argsort(abs(V2))[::-1]
        W = W[:, ind]
        return W

    def _fit_2(self, R1, R2):
        V, U = la.eig(R1, R1+R2)
        ind = np.argsort(abs(V))[::-1]
        V = V[ind]
        W = U[:, ind]
        return W

    def project(self, x_trial, wcsp):
        """Apply CSP filtering
        inputs:
            x_trial: (n_samples, n_channels)
            wcsp: (n_channels, n_channels)
        outputs:
            (n_samples, n_filters)
        """
        wcsp_selected = np.hstack((wcsp[:, :self.n_filters], wcsp[:, -self.n_filters:]))
        z_trial_selected = np.matmul(x_trial, wcsp_selected)
        return z_trial_selected

    def transform(self, x_trial, wcsp):
        """Extract CSP filtered covariance feature
        inputs:
            x_trial: (n_samples, n_channels)
            wcsp: (n_channels, n_channels)
        outputs:
            (n_filters,)
        """
        z_trial_selected = self.project(x_trial, wcsp)
        var_z = np.abs(np.diag(np.matmul(np.transpose(z_trial_selected), z_trial_selected)))
        sum_var_z = sum(var_z)
        return np.log(var_z/sum_var_z)


class FBCSP:
    def __init__(self, n_filters):
        self.n_filters = n_filters

    def fit(self, x_train_fb, y_train):
        """fit the FBCSP filters
        inputs:
            x_train_fb: (n_trials, n_fbanks, n_samples, n_channels)
            y_train: (n_trials,) multi-class label is allowed
        outputs:
            (n_classes, n_fbanks, n_channels, n_channels)
        """
        labels = np.sort(np.unique(y_train))
        n_classes = len(labels)
        self.csp = CSP(self.n_filters)
        fbcsp_filters_multi = []
        for i in range(n_classes):
            cls_of_interest = labels[i]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            fbcsp_filters = self.fit_ovr(x_train_fb, y_train_cls)
            fbcsp_filters_multi.append(fbcsp_filters)
        return np.asarray(fbcsp_filters_multi)

    def fit_ovr(self, x_train_fb, y_train_cls):
        """fit the one-vesus-the-rest (OVR) CSP filters
        inputs:
            x_train_fb: (n_trials, n_fbanks, n_samples, n_channels)
            y_train_cls: (n_trials,) 0 target class, 1 otherwise
        outputs:
            (n_fbanks, n_channels, n_channels)
        """
        n_fbanks = x_train_fb.shape[1]
        fbcsp_filters = []
        for k in range(n_fbanks):
            x_train = x_train_fb[:, k, :, :]
            csp_filters = self.csp.fit(x_train, y_train_cls)
            fbcsp_filters.append(csp_filters)
        return np.asarray(fbcsp_filters)

    def transform(self, x_data_fb, fbcsp_filters):
        """Apply FBCSP filtering
        inputs:
            x_data_fb: (n_trials, n_fbanks, n_samples, n_channels)
            fbcsp_filters: (n_classes, n_fbanks, n_channels, n_channels)
        outputs:
            (n_classes*n_fbanks*n_filters,)
        """
        n_trials, n_fbanks, n_samples, n_channels = x_data_fb.shape
        n_classes = fbcsp_filters.shape[0]
        x_features = np.zeros((n_trials, n_classes, n_fbanks, self.n_filters*2))
        for i in range(n_classes):
            for j in range(n_fbanks):
                csp_filters = fbcsp_filters[i, j]
                for k in range(n_trials):
                    x_trial = np.copy(x_data_fb[k, i, :, :])
                    x_features[k, i, j] = self.csp.transform(x_trial, csp_filters)
        return np.reshape(x_features, [n_trials, -1])