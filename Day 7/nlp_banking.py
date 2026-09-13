from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("banking_knowledge_base_1000.csv")

df = df.dropna(subset=['Question', 'Section'])
 
df['Question'] = df['Question'].str.lower()

cv = CountVectorizer(stop_words='english')
x = cv.fit_transform(df['Question']).toarray()
y = df['Section']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=6,train_size=0.77
)

rf = RandomForestClassifier()
model = rf.fit(x_train,y_train)
print(model.score(x_test, y_test))

message = ["Does this health insurance policy cover emergency hospitalization expenses abroad?"]

message_cv = cv.transform(message).toarray()

print(f"Result: {model.predict(message_cv)}")