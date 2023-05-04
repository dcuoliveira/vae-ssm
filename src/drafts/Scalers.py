from sklearn.preprocessing import MinMaxScaler


class Scalers:
    def __init__(self, scaler_type, params=None) -> None:
        self.scaler_type = scaler_type
        self.scaler_metadata = {
            "min_max_scaler": MinMaxScaler
            }
        
        if params == None:
            self.scaler = self.scaler_metadata[scaler_type]()
        else:
            self.scaler = self.scaler_metadata[scaler_type](params)

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)