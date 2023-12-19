""" Implements a pipeline for preprocessing that's compatible with pytorch Tensor objects. """


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for step in self.steps:
            X = step.fit_transform(X)
        return X

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def inverse_transform(self, X):
        for step in reversed(self.steps):
            X = step.inverse_transform(X)
        return X
