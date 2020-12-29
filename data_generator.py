# %%
from sklearn import datasets
import numpy as np
import pandas as pd
# %%
data, y0 = datasets.make_blobs(n_samples=200, n_features=20, centers=4)
# %%
probs = np.linspace(0, 1, 10)
labels = np.stack([np.random.choice([0, 1], data.shape[0], p=(prob, 1. - prob)) for prob in probs]).T
data.shape, y0.shape, labels.shape
data_conc = np.concatenate([data,np.expand_dims(y0, 1), labels], axis=1)

header = [f"x_{i}" for i in range(data.shape[1])] + [f"y_{i}" for i in range(len(probs) + 1)]


# %%
df = pd.DataFrame(data_conc, columns=header)
df.to_csv("sample_data.csv")
# %%
