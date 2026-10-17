import numpy as np #Used for working with the arrays
import pickle #Used for downloading the saved model
import streamlit as st #Used for Deployment of webApp

#Load the trained model
scaler_model = pickle.load(open("scaler.sav", "rb"))
loaded_model = pickle.load(open("trained_model.sav", "rb"))

#Creating a function for prediction
def diabetesPrediction(input_data):

    #changing the input data to numpy array
    input_data_as_Numpy_array = np.asanyarray(input_data)


    # Reshape the array as we are predicting for one instace not 768 as the whole instance
    input_data_reshape =  input_data_as_Numpy_array.reshape(1,-1)

    # Standardize the test input data since we trained using a standardized dataset
    std_data = scaler_model.transform(input_data_reshape)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print("Prediction", prediction)

    if(prediction[0] == 0):
        return "The person is non-diabetic"
    elif(prediction[0] == 1):
        return "The person is diabetic"


def main():

    # Giving a Title for our webPage
    st.title('Diabetes Prediction Web APP')

    #Getting the input data from the user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Levels")
    Bmi = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age of the person")


    # code for prediction
    diagnosis = ""

    #Creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetesPrediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)




if __name__ == '__main__':
    main()

