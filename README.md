 Heart Disease Prediction Web App

 Project Overview

This project is a Machine Learning based Heart Disease Prediction Web Application developed using Python, Flask, HTML, and Scikit-learn.

The application allows users to enter medical details through a frontend form and predicts whether the patient has chances of heart disease.

The trained Random Forest Machine Learning model is integrated with a Flask backend and deployed locally through a web interface.


 Features

* Heart disease prediction using Machine Learning
* Frontend form interface using HTML
* Flask backend integration
* Random Forest Classifier model
* Input preprocessing using MinMaxScaler
* Real-time prediction output


 Technologies Used

* Python
* Flask
* HTML
* Scikit-learn
* NumPy
* Pickle


 Project Structure
 
heart disease prediction model/
│
├── app.py
├── rf_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── templates/
│     └── home.html
│
└── analysis_heart_disease.ipynb



 How the Project Works

User Input Form
       ↓
Flask Backend
       ↓
Data Preprocessing using Scaler
       ↓
Random Forest ML Model
       ↓
Prediction Output


 Machine Learning Model

The project uses:

* Random Forest Classifier for prediction
* MinMaxScaler for preprocessing

Prediction Output:

* 0 → No Heart Disease
* 1 → Heart Disease



 Setup Instructions

 Step 1: Clone or Download the Project

Download the project folder and open it in VS Code.


 Step 2: Install Python

Make sure Python is installed.

Check version:


python --version



 Step 3: Install Required Libraries

Open terminal inside the project folder and run:


pip install -r requirements.txt


If requirements.txt is not available, install manually:


pip install flask numpy pandas scikit-learn



 Step 4: Run the Flask Application

Open terminal in the project folder and run:


python app.py


You will see:


Running on http://127.0.0.1:5000




 Step 5: Open in Browser

Open:


http://127.0.0.1:5000


Enter patient medical details in the form and click Predict.


 Input Parameters

The application accepts the following medical inputs:

* Gender
* Age
* Current Smoker
* Cigarettes Per Day
* BP Medicines
* Prevalent Stroke
* Prevalent Hypertension
* Diabetes
* Total Cholesterol
* Systolic BP
* Diastolic BP
* BMI
* Heart Rate
* Glucose Level



 Deployment Explanation

I deployed my Heart Disease Prediction Machine Learning model using Flask and created a frontend form interface where users can enter medical details and get predictions.

The frontend is developed using HTML forms and the backend is implemented using Flask.
The trained Random Forest model processes user inputs and returns prediction results in real time.



 Author

Sara Maryam
B.Tech Electrical Engineering
