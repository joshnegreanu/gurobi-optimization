import numpy as np
import gurobipy as grb

'''
SoftMarginSVM
    Support vector machine implementation using Gurobi for optimization.
    Soft margin formulation with dual problem solved via quadratic programming,
    supporting linear, quadratic, and RBF kernels.
'''
class SoftMarginSVM:
    '''
    SoftMarginSVM.__init__
        Initializes soft margin SVM with given data and hyperparameters.

    Parameters:
        X : np.ndarray training data features of shape (num_samples, num_features)
        y : np.ndarray training data labels of shape (num_samples,), with values in {-1, 1}
        C : float regularization parameter
        kernel : str type of kernel to use ("linear", "quadratic", "rbf")
        gamma : float parameter for RBF kernel (ignored for other kernels)
    
    Returns:
        None
    '''
    def __init__(self, X, y, C, kernel = "rbf", gamma = 1.0, verbose = False):
        # hyperparameters
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        # data
        self.X = X
        assert kernel in ["linear", "quadratic", "rbf"], "Unsupported kernel type."
        self.K = self._compute_gram(X, X, kernel)

        self.y = y
        self.num_samples, self.dim = X.shape
        self.alpha = None

        # build model
        self.model = grb.Model("SoftSVM_Dual")
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        self.alpha = self.model.addMVar(self.num_samples, lb=0, ub=C, name="alpha")
        self.opt_alpha = None

        # compute quadratic term with numpy efficiently
        Q = (y[:, None] * y[None, :]) * self.K

        obj = self.alpha.sum() - 0.5 * self.alpha @ Q @ self.alpha
        self.model.setObjective(obj, grb.GRB.MAXIMIZE)

        # KKT equality constraints
        self.model.addConstr(self.alpha @ self.y == 0, name="KKT_eq")
    
    '''
    SoftMarginSVM.optimize
        Optimizes the dual problem to find optimal alpha values,
        stored internally for prediction.
    
    Parameters:
        None
    
    Returns:
        None
    '''
    def optimize(self):
        # optimize dual problem
        self.model.optimize()
        if self.model.status == grb.GRB.OPTIMAL:
            self.opt_alpha = self.alpha.X
        else:
            raise Exception("Optimization did not converge to an optimal solution.")
    
    '''
    SoftMarginSVM.predict
        Predicts labels for validation data using the optimized model.
    
    Parameters:
        X_val : np.ndarray validation data features of shape (num_val_samples, num_features)
    
    Returns:
        np.ndarray predicted labels of shape (num_val_samples,), with values in {-1, 1}
    '''
    def predict(self, X_val):
        # predict labels for validation data
        if self.opt_alpha is None:
            raise Exception("Model has not been optimized yet.")
        
        K_val = self._compute_gram(X_val, self.X, self.kernel)
        decision_values = (self.opt_alpha * self.y) @ K_val.T
        return np.sign(decision_values)

    '''
    SoftMarginSVM._compute_gram
        Computes the Gram matrix for given data and kernel type.
    
    Parameters:
        X1 : np.ndarray first data set of shape (num_samples_1, num_features)
        X2 : np.ndarray second data set of shape (num_samples_2, num_features)
        kernel : str type of kernel to use ("linear", "quadratic", "rbf")
    
    Returns:
        np.ndarray Gram matrix of shape (num_samples_1, num_samples_2)
    '''
    def _compute_gram(self, X1, X2, kernel):
        # compute gram matrix based on kernel type
        match kernel:
            case "linear":
                return X1 @ X2.T
            case "quadratic":
                return (X1 @ X2.T) ** 2
            case "rbf":
                sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
                return np.exp(-self.gamma * sq_dists)