def generate_data(n_samples=2000, noise_std=0.6):
    X = np.random.uniform(-4, 4, size=(n_samples, 2)).astype(np.float32)
    # Ground-truth rule: class 1 if x0 and x1 have different signs (XOR), else class 0
    y = (np.sign(X[:, 0]) != np.sign(X[:, 1])).astype(np.float32)
    X += np.random.normal(0, noise_std, size=X.shape).astype(np.float32)  # feature noise
    return torch.tensor(X), torch.tensor(y).view(-1, 1)

X, y = generate_data()

n_train = int(0.8 * len(X))
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:], y[n_train:]
