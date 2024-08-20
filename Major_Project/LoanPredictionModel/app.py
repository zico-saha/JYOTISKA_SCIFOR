import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import os

# Define the absolute path to the directory where your models and data files are located
base_dir = '/mount/src/jyotiska_scifor/Major_Project/LoanPredictionModel'

# Set up the Streamlit app configuration
st.set_page_config(
    page_title="Loan Eligibility Prediction",  # Title of the app in the browser
    page_icon=":bar_chart:",  # Emoji as an app icon
    layout="wide",  # Layout setting to make the app occupy the full width of the browser
    initial_sidebar_state="expanded"  # Sidebar will be expanded initially
)

# Load pre-trained machine learning models from files
logistic_model = joblib.load(os.path.join(base_dir, 'LogisticRegression_best_model.pkl'))
linear_svc_model = joblib.load(os.path.join(base_dir, 'LinearSVC_best_model.pkl'))
nb_model = joblib.load(os.path.join(base_dir, 'GaussianNB_best_model.pkl'))
decision_tree_model = joblib.load(os.path.join(base_dir, 'DecisionTreeClassifier_best_model.pkl'))
random_forest_model = joblib.load(os.path.join(base_dir, 'RandomForestClassifier_best_model.pkl'))
adaboost_model = joblib.load(os.path.join(base_dir, 'AdaBoostClassifier_best_model.pkl'))
gradient_boost_model = joblib.load(os.path.join(base_dir, 'GradientBoostingClassifier_best_model.pkl'))
knn_model = joblib.load(os.path.join(base_dir, 'KNeighborsClassifier_best_model.pkl'))

# Define mappings for categorical features to numerical values
categorical_mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    'Education': {'Graduate': 0, 'Not Graduate': 1},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Credit_History': {1: 1, 0: 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
}

# List of numeric features to be scaled
numeric_features = ['ApplicantIncome',
                    'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Create an instance of StandardScaler to scale numeric features
scaler = StandardScaler()

# Function to preprocess user input data (categorical feature encoding and numeric scaling)
def preprocess_input(data):
    # Convert categorical features to numeric values using mappings
    for feature, mapping in categorical_mappings.items():
        if feature in data.columns:
            data[feature] = data[feature].map(mapping)

    # Scale numeric features using StandardScaler
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    return data

# CSS for styling the navigation buttons
st.markdown(
    """
    <style>
    .btn-group {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .btn {
        background-color: #007BFF;
        color: white;
        padding: 10px 20px;
        margin: 0 5px;  /* Reduced margin to decrease gap between buttons */
        border-radius: 5px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
    }
    .btn.selected {
        background-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True
)

# Initialize the selected button in session state if not already initialized
if 'active_button' not in st.session_state:
    st.session_state.active_button = 'Loan Status Prediction'

# Display the navigation buttons for different app sections
col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Loan Prediction', key='loan_button'):
        st.session_state.active_button = 'Loan Prediction'
with col2:
    if st.button('Training Data Visualization', key='viz_button'):
        st.session_state.active_button = 'Training Data Visualization'
with col3:
    if st.button('Model Training Results', key='results_button'):
        st.session_state.active_button = 'Model Training Results'

# Display content based on the selected button

# Loan Prediction Section
if st.session_state.active_button == 'Loan Prediction':
    st.title("Loan Prediction")

    # Function to collect user input
    def get_user_input():
        # Get user input from the sidebar
        gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        married = st.sidebar.selectbox('Married', ['Yes', 'No'])
        dependents = st.sidebar.selectbox('Dependents', ['0', '1', '2', '3+'])
        education = st.sidebar.selectbox(
            'Education', ['Graduate', 'Not Graduate'])
        self_employed = st.sidebar.selectbox('Self Employed', ['No', 'Yes'])
        credit_history = st.sidebar.selectbox('Credit History', [1, 0])
        property_area = st.sidebar.selectbox(
            'Property Area', ['Rural', 'Semiurban', 'Urban'])

        # Numeric feature inputs from the sidebar
        applicant_income = st.sidebar.number_input(
            'Applicant Income', min_value=0)
        coapplicant_income = st.sidebar.number_input(
            'Coapplicant Income', min_value=0)
        loan_amount = st.sidebar.number_input('Loan Amount', min_value=0)
        loan_amount_term = st.sidebar.number_input(
            'Loan Amount Term', min_value=0)

        # Create a data dictionary from the inputs
        data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area,
        }

        # Return the data as a pandas DataFrame
        return pd.DataFrame([data])

    # Sidebar dropdown to allow the user to select a machine learning model
    model_choice = st.sidebar.selectbox("Select the classifier",
                                        ['Logistic Regression', 'LinearSVC', 'GaussianNB', 'DecisionTree',
                                         'RandomForest', 'AdaBoost', 'GradientBoost', 'KNeighbors'])

    # Function to load the chosen machine learning model
    def get_model(model_choice):
        if model_choice == 'Logistic Regression':
            return logistic_model
        elif model_choice == 'LinearSVC':
            return linear_svc_model
        elif model_choice == 'GaussianNB':
            return nb_model
        elif model_choice == 'DecisionTree':
            return decision_tree_model
        elif model_choice == 'RandomForest':
            return random_forest_model
        elif model_choice == 'AdaBoost':
            return adaboost_model
        elif model_choice == 'GradientBoost':
            return gradient_boost_model
        elif model_choice == 'KNeighbors':
            return knn_model

    # Display the selected model name to the user
    st.write(f"Using model: {model_choice}")

    # Get user input from the sidebar
    input_data = get_user_input()

    # Preprocess the user input
    input_data = preprocess_input(input_data)

    # Get the selected machine learning model
    model = get_model(model_choice)

    # Display the loan prediction result when the user clicks the 'Predict' button
    if st.button('Predict'):
        prediction = model.predict(input_data)
        # Display the result as "Loan Approved" or "Loan Not Approved" based on the model's prediction
        result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Not Approved ❌"
        st.markdown("### Prediction:")
        st.markdown(f"## {result}")

# Training Data Visualization Section
elif st.session_state.active_button == 'Training Data Visualization':
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Property_Area']
    num_cols = ['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term']

    st.title("Training Data Visualizations")

    # Add spacing and a subtle divider for better visual separation
    st.markdown("<br>"*2, unsafe_allow_html=True)
    st.markdown("---")

    # Load the data
    DF = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))

    categorical_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    imputer_cat = SimpleImputer(strategy='most_frequent')
    DF[categorical_columns] = imputer_cat.fit_transform(
        DF[categorical_columns])
    numerical_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    imputer_num = SimpleImputer(strategy='median')
    DF[numerical_columns] = imputer_num.fit_transform(DF[numerical_columns])

    # Define a custom color sequence
    color_sequence = ['#FF6347', '#4682B4']

    # Plot categorical bar plots
    st.subheader("Categorical Features")
    for col in categorical_columns:
        fig = px.bar(DF, x=col, title=f'{col} Distribution',
                     color=DF[col].astype(str), color_discrete_sequence=color_sequence)
        st.plotly_chart(fig)

    # Plot numerical histograms
    st.subheader("Numerical Features")
    for col in numerical_columns:
        fig = px.histogram(DF, x=col, title=f'{col} Histogram')
        st.plotly_chart(fig)

    # Plot numerical box plots
    st.subheader("Numerical Box Plots")
    for col in numerical_columns:
        fig = px.box(DF, y=col, title=f'{col} Box Plot')
        st.plotly_chart(fig)

    # Plot numerical swarm plots
    st.subheader("Numerical Swarm Plots")
    for col in numerical_columns:
        fig = px.strip(DF, y=col, title=f'{col} Swarm Plot')
        st.plotly_chart(fig)

# Model Training Results Section
elif st.session_state.active_button == 'Model Training Results':
    st.title("Model Training Results")

    # Define the accuracy scores for different models
    accuracy_scores = {
        'Logistic Regression': 0.85,
        'Linear SVC': 0.69,
        'Gaussian Naive Bayes': 0.85,
        'Decision Tree': 0.71,
        'Random Forest': 0.81,
        'AdaBoost': 0.80,
        'Gradient Boost': 0.82,
        'K-Neighbors': 0.64
    }

    # Create a bar chart to display the accuracy scores
    fig = go.Figure(data=[go.Bar(
        x=list(accuracy_scores.keys()),
        y=list(accuracy_scores.values()),
        marker_color='#1f77b4'
    )])
    fig.update_layout(title='Model Accuracy Comparison',
                      xaxis_title='Model',
                      yaxis_title='Accuracy',
                      xaxis_tickangle=-45)

    st.plotly_chart(fig)

    st.markdown("**Hyperparameters Used:**")
    st.markdown(
        """
        - **Logistic Regression:** {'C': 0.1, 'max_iter': 100, 'solver': 'lbfgs'}
        - **Linear SVC:** {'C': 0.1}
        - **Decision Tree Classifier:** {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2}
        - **Random Forest Classifier:** {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
        - **AdaBoost Classifier:** {'learning_rate': 0.01, 'n_estimators': 50}
        - **Gradient Boosting Classifier:** {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}
        - **K-Neighbors Classifier:** {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'uniform'}
        """
    )
