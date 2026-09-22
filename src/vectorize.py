import pandas as pd 
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def get_prepared_data():
    df = pd.read_csv("..\\data\\processed\\data_netflix")

    vectorized = TfidfVectorizer(max_features=1000)
    X_numpy = vectorized.fit_transform(df["description"]).toarray()

    label_encoder = LabelEncoder()
    y_numpy = label_encoder.fit_transform(df["type"])

    X_tensor = torch.tensor(X_numpy, dtype = torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype = torch.long)

    print("Обнаружено классов (тем): {len(label_encoder.classes)}")

    return X_tensor, y_tensor

if __name__ == "__main__":
    get_prepared_data()
