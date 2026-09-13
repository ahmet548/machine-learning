import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.DataFrame()
df["sentences"] = ["ali bak", "ali ata bak", "bak ali bak", "ali güzel ata bak", "ışık ılık süt iç", "ışık süt iç", "iç ışık iç"]

cv = CountVectorizer(max_features=4)
arr = cv.fit_transform(df["sentences"])

print(cv.get_feature_names_out())
print(arr.toarray())