import streamlit as st
import numpy as np
import joblib


# Function to load the saved model
def load_model(model_path):
    loaded_model = joblib.load(model_path, mmap_mode='r')
    return loaded_model


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

    # Predict using the loaded model
    prediction = predict(model, input_data)

    with st.sidebar:
        # Display prediction in the main page
        predicted_performance = prediction[0].round(6)
        st.subheader('Predicted Overall Performance:')
        st.markdown(f'<div style="color: orange; font-size: 36px;">{predicted_performance}</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
