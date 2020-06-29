import streamlit as st

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

st.title("Machine Learning application")

st.write("""
    # Explore different Classifiers
    Which One is the Best?
    """)

dataset_name = st.sidebar.selectbox("Select the Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
classifier_name = st.sidebar.selectbox("Select the Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if(dataset_name == "Iris"):
        data = datasets.load_iris()
    elif(dataset_name == "Breast Cancer"):
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)

st.write("Shape of the Dataset: ", X.shape)
st.write("Number of Classes: ", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if(clf_name == "KNN"):
        K = st.sidebar.slider("K", 1, 15)     # Start = 1 and stop =15 for slider
        params["K"] = K
    elif(clf_name == "SVM"):
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)    # Depth for each tree
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)  # Number of Trees
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if(clf_name == "KNN"):
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif(clf_name == "SVM"):
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth = params["max_depth"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {accuracy}")


pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.xlabel("Principal Component 2")
plt.colorbar()
st.pyplot()
