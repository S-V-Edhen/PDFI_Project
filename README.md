# Punjab Farmer Distress Prediction and Decision Support System

This project predicts a Punjab Farmer Distress Index (PFDI) using machine learning and generates farmer-level and policy-level recommendations.

## Features
- Synthetic Punjab farmer dataset generation
- PFDI prediction using XGBoost, SVR, and Random Forest
- Decision support system for intervention recommendations
- Streamlit-based user interface

## Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit

## Project Structure
- `app.py` - Streamlit UI
- `src/decision_support.py` - recommendation engine
- `models/` - trained model
- `data/` - dataset
- `notebooks/` - experimentation notebooks

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
