import numpy as np
import pickle

#Load the trained model
scaler_model = pickle.load(open("scaler.sav", "rb"))
loaded_model = pickle.load(open("trained_model.sav", "rb"))

input_data = (5,166,72,19,175,25.8,0.587,51)

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
    print("The person is non-diabetic")
elif(prediction[0] == 1):
    print("The person is diabetic")