class BaseModel:
    name = "BaseModel"

    def fit(self, data):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
