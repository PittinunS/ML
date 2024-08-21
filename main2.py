import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

df = sns.load_dataset('iris')
df

x = df.iloc[:, :-1]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

st.sidebar.title('Classifiers')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'DT', 'RF', 'NN'))
k = st.sidebar.slider('K', 1, 120, 3)
if classifier == 'KNN':
  knn = KNeighborsRegressor(n_neighbors=5)
  knn.fit(x.reshape(-1, 1), y)
  y_pred = knn.predict(x.reshape(-1, 1))
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'SVM':
  svm = SVC()
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'DT':
  dt = DecisionTreeRegressor()
  dt.fit(x.reshape(-1, 1), y)
  y_pred = dt.predict(x.reshape(-1, 1))
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'RF':
  rf = RandomForestRegressor()
  rf.fit(x.reshape(-1, 1), y)
  y_pred = rf.predict(x.reshape(-1, 1))
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'NN':
  nn = MLPRegressor()
  nn.fit(x.reshape(-1, 1), y)
  y_pred = nn.predict(x.reshape(-1, 1))
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
