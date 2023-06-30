import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, classification_report
from sklearn.svm import SVC
import time
import base64
import plotly.express as px


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def main():
    st.set_page_config(page_title="Airline Analytics", page_icon=":airplane:")
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#4e54c8, #8f94fb);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image(
        "https://www.analyticslane.com/wp-content/uploads/2018/10/system.jpg",
        width=300,
    )
    st.sidebar.title("Airline Analytics")
    choice = st.sidebar.radio(
        "Navigation", ["Upload", "Profiling", "Modelling", "Download", "Data Visualization"])
    st.sidebar.info(
        "This project application helps you analyze and model your airline data.")

    if choice == "Upload":
        # Display the first image above the Upload section
        st.image("https://conocedores.com/wp-content/uploads/2019/04/nuevo-aeropuerto-ezeiza-zepellin-13042019in1.jpeg",
                 width=600)

        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset (max 600MB)",
                                type=["csv"], accept_multiple_files=False)
        if file:
            st.session_state.df = load_data(file)
            st.dataframe(st.session_state.df)

    elif choice == "Profiling":
        st.title("Data Profiling")
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first.")
            return

        columns = st.multiselect(
            'Choose Columns', options=st.session_state.df.columns)

        if st.button('Start Profiling'):
            # Start the timer
            start_time = time.time()

            # Generate the report for the selected columns
            pr = ProfileReport(st.session_state.df[columns], explorative=True)
            st_profile_report(pr)

            # Calculate the time it took to generate the report
            elapsed_time = time.time() - start_time

            # Display the elapsed time
            st.write(f"Elapsed time: {elapsed_time:.2f} seconds")

        if st.button('Restart Profiling'):
            # Clear the report from the page
            st.write("Profiling restarted.")

    elif choice == "Modelling":
        # Display the second image above the Modelling section
        st.image("https://www.mochilaycamara.com/wp-content/uploads/2020/01/IMG_1599-2048x1164.jpeg",
                 width=600)

        st.title("Modeling")
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first.")
            return

        chosen_target = st.selectbox(
            'Choose the Target Column', st.session_state.df.columns)
        features_to_drop = st.multiselect(
            'Choose Features to Drop', st.session_state.df.columns)
        model_choice = st.multiselect('Choose Models', [
            'Logistic Regression', 'Naive Bayes', 'SVC', 'Random Forest', 'Gradient Boosting'])

        if st.button('Run Modeling'):
            X = st.session_state.df.drop(
                columns=[chosen_target] + features_to_drop)
            y = st.session_state.df[chosen_target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            models = []
            if 'Logistic Regression' in model_choice:
                models.append(('Logistic Regression', LogisticRegression()))
            if 'Naive Bayes' in model_choice:
                models.append(('Naive Bayes', GaussianNB()))
            if 'Random Forest' in model_choice:
                models.append(('Random Forest', RandomForestClassifier()))
            if 'Gradient Boosting' in model_choice:
                models.append(('Gradient Boosting', GradientBoostingClassifier(
                    max_depth=6, min_samples_split=5, n_estimators=120)))
            if 'SVC' in model_choice:
                models.append(
                    ('SVC', SVC(C=1, kernel='linear', random_state=42)))

            for name, model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                recall = recall_score(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred, output_dict=True)
                st.write(f"Model: {name}")
                st.write("Recall:", recall)
                st.write("Classification Report:")
                st.json(report)

                # Save the classification report to session state for download
                st.session_state.report = report

    elif choice == "Download":
        # Display the third image above the Download section
        st.image("https://cloudfront-us-east-1.images.arcpublishing.com/infobae/NVAZ3FOMHFCN3JZE6AFWH5WKNM.jpg",
                 width=600)

        if 'report' not in st.session_state:
            st.warning("No model to download.")
            return

        # Convert the classification report to a DataFrame and download as CSV
        report_df = pd.DataFrame(st.session_state.report).transpose()
        csv = report_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="classification_report.csv">Download Classification Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif choice == "Data Visualization":
        # Display an image related to data science above the Data Visualization section
        st.image("https://conocedores.com/wp-content/uploads/2019/04/nuevo-aeropuerto-ezeiza-zepellin-13042019in1.jpeg", width=600)
        st.title("Data Visualization")
        file = st.file_uploader("Upload a CSV file to visualize", type=["csv"])
        if file:
            vis_df = pd.read_csv(file)
            st.session_state['df'] = vis_df
        if 'df' not in st.session_state:
            st.warning("Please upload a dataset first.")
            return

        features = st.multiselect(
            "Selecciona varias features y haz gropuby ", vis_df.columns)
        if len(features) > 1:
            fig = px.scatter_matrix(vis_df, dimensions=features)
            st.plotly_chart(fig)
        else:
            st.warning("Por favor, selecciona al menos dos características.")

        x_axis = st.selectbox("Choose x-axis column", vis_df.columns)
        y_axis = st.selectbox("Choose y-axis column", vis_df.columns)
        chart_type = st.selectbox("Select chart type", [
            "Scatter plot", "Bar plot", "Pie plot"])
        if chart_type == "Scatter plot":
            fig = px.scatter(vis_df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Bar plot":
            fig = px.bar(vis_df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Pie plot":
            fig = px.pie(vis_df, names=y_axis)
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()
