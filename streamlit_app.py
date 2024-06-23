import subprocess
import sys


# Function to check if library is installed
def check_library_installed(lib_name):
    try:
        __import__(lib_name)
        return True
    except ImportError:
        return False


# Function to install libraries if not already installed
def install_required_libraries():
    libraries = [
        "streamlit==1.36.0",
        "numpy==1.25.2",
        "joblib==1.4.2",
        "scikit-learn==1.2.2"
    ]

    for lib in libraries:
        if not check_library_installed(lib.split('==')[0]):
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])



#install required libraries
install_required_libraries()


# Then import the required libraries
import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor





# The function loads the saved model
def load_model(model_path):
    try:
        loaded_model = joblib.load(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict(model, input_data):
    # Predict using the loaded model
    prediction = model.predict(input_data)
    return prediction


def main():

    st.title('Player Performance Prediction App')

    # Sidebar for input features
    st.title('Input Player Features')
    value_eur = st.slider('Value of Player in Euros', 0.0, 194000000.0, 5000000.0)
    age = st.slider('Age of the Player', 16, 40, 25)
    potential = st.slider('Potential of the Player', 0, 100, 50)
    movement_reactions = st.slider('Movement Reaction of the Player', 0, 100, 50)
    wage_eur = st.slider("Player's Wage in Euros", 0.0, 600000.0, 50000.0)

    # Create a numpy array for prediction
    input_data = np.array([[value_eur, age, potential, movement_reactions, wage_eur]])

    # Load the model
    model_path = 'rf_model.pkl'
    model = load_model(model_path)

    if model:
        # Predict using the loaded model
        prediction = predict(model, input_data)

        # Display prediction in the sidebar
        predicted_performance = prediction[0].round(6)

        st.sidebar.subheader('Predicted Overall Performance:')
        st.sidebar.markdown(f'<div style="color: orange; font-size: 36px;">{predicted_performance}</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
