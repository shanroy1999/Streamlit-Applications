import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use("Agg")

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
    """ Semi-Auto ML Application with Streamlit """

    st.title("Semi-Auto ML Application with Streamlit")
    activities = ["EDA", "Plot", "Model Building"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice=="EDA":
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload Your Dataset", type=["csv", "txt", ])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

            if st.checkbox("Show Shape"):
                st.write(df.shape)

            if st.checkbox("Show Columns"):
                all_col = df.columns.to_list()
                st.write(all_col)

            if st.checkbox("Select Columns to Show"):
                selected_columns = st.multiselect("Select Columns", all_col)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Show Summary"):
                st.write(df.describe())

            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:, -1].value_counts())

            if st.checkbox("Correlation with Seaborn"):
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot()

            if st.checkbox("Pie Chart"):
                all_col = df.columns.to_list()
                columns_to_plot = st.selectbox("Select Columns", all_col)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()

    elif choice=="Plot":
        st.subheader("Data Visualization")

        data = st.file_uploader("Upload Your Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

        all_columns_names = df.columns.to_list()
        plot_type = st.selectbox("Select the type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(plot_type, selected_columns_names))

            if plot_type=="area":
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

            elif plot_type=="bar":
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

            elif plot_type=="line":
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            elif plot_type:
                cust_plot = df[selected_columns_names].plot(kind=plot_type)
                st.write(cust_plot)
                st.pyplot()

    elif choice=="Model Building":
        st.subheader("Building ML Model")

        data = st.file_uploader("Upload Your Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

            X = df.iloc[:, 0:-1]
            y = df.iloc[:, -1]
            seed = 7

            models = []
            models.append(("LR", LogisticRegression()))
            models.append(("LDA", LinearDiscriminantAnalysis()))
            models.append(("KNN", KNeighborsClassifier()))
            models.append(("CART", DecisionTreeClassifier()))
            models.append(("NB", GaussianNB()))
            models.append(("SVM", SVC()))

            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = "accuracy"

            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())

                accuracy_results = {"model_name":name, "model_accuracy":cv_results.mean(), "standard_deviation":cv_results.std()}
                all_models.append(accuracy_results)

            if st.checkbox("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std), columns=["Model Names", "Model Accuracy", "Standard Deviation"]))

            if st.checkbox("Metrics as JSON"):
                st.json(all_models)

    elif choice=="About":
        st.subheader("About")

if __name__=="__main__":
    main()
