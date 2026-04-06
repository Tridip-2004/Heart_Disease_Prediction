## 🧠 Tech Stack

- **Programming Language:** Python  
- **Libraries:**  
  - pandas  
  - numpy  
  - scikit-learn  
  - imbalanced-learn  
  - xgboost  
  - lightgbm  
  - joblib  
- **Web Framework:** Flask  
- **Deployment Platform:** Render


## Live Application

https://heart-disease-prediction-8-mko2.onrender.com/


## 📁 Project Structure

heart-disease-prediction/
│

├── app.py # Flask


├── model/


│ └── heart_model.joblib # Trained ML model

│


├── requirements.txt # Python dependencies


├── .python-version # Python version for Render


├── README.md # Project documentation


└── data.csv # Dataset (optional)

## 📊 Dataset


- Dataset sourced from **UCI Machine Learning Repository**

- Contains clinical features such as:
  - Age
  - Sex
  - Chest pain type
  - Blood pressure
  - Cholesterol
  - Fasting blood sugar
  - ECG results
  - Max heart rate
  - Exercise-induced angina

## ⚙️ Environment Setup (Local)

### 1️⃣ Create virtual environment
bash

python -m venv venv

2️⃣ Activate environment


💻 Windows 


venv\Scripts\activate

🧑‍💻 Linux / macOS


source venv/bin/activate


## 3️⃣ Install dependencies

pip install -r requirements.txt

📦 requirements.txt

gunicorn==25.3.0

imbalanced-learn==0.14.1

joblib==1.5.3

lightgbm==4.6.0

numpy==2.0.2

packaging==26.0

pandas==2.2.2

python-dateutil==2.9.0.post0

pytz==2026.1.post1

scikit-learn==1.6.1

scipy==1.17.1

six==1.17.0

sklearn-compat==0.1.5

threadpoolctl==3.6.0

tzdata==2026.1

xgboost==3.2.0

click==8.3.2 

flask==3.1.3 

jinja2==3.1.6


## 🐍 add .python-version (IMPORTANT)

Render must NOT use Python 3.13 / 3.14 for ML apps.

python-3.12.9

This prevents pandas build errors during deployment.


## ▶️ Run the App Locally


Streamlit

--Streamlit run app.py

HTML,CSS

Flask

--python app.py


## ☁️ Deploy on Render

📄 Build Command

pip install -r requirements.txt

🤖 Start Command

Streamlit

streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Flask

gunicorn app:app

❗ Common Deployment Issue (Solved)

Error

metadata-generation-failed

pandas build error

Cause

Render auto-selects Python 3.14

Pandas does not support Python 3.14

Solution

✔️ Added .python-version(3.12.9 is stable)

✔️ Pinned stable package versions


## 📈 Model Performance

Accuracy: 86%+

Evaluation metrics:

Accuracy

ROC-AUC

Classification Report


## 🔒 Notes

Do NOT commit the venv/ folder

Always pin dependency versions

Use Python 3.11 / 3.12 for ML projects


## 🙌 Future Improvements

Add REST API endpoint

Add patient health tips

Add authentication

Improve UI


## 👨‍💻 Author

Tridip Panja

Machine Learning & AI Enthusiast
