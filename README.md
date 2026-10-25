# Customer Behaviour Analytics & Future Spending Prediction

A machine learning project that analyzes retail customer behaviour, segments customers using K-Means clustering, and predicts future spending using a Random Forest Regressor. Built with Python and Streamlit.

## Features
- Customer behaviour analysis
- Customer segmentation using K-Means
- Future spending prediction
- Interactive dashboard and visualizations
- Feature importance analysis

## Technologies
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit · Joblib

## Dataset
Uses the Online Retail dataset from the UCI Machine Learning Repository.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo
https://customer-behaviour-analytics-and-customer-lifetime-value-predi.streamlit.app/

## Note
The model predicts spending over a defined future period, not a customer's entire lifetime value. Prediction accuracy is limited, so results should be interpreted cautiously.

## License
MIT License
