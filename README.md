# 🚗 Car Price Predictor

A machine learning web app that predicts the market price of a used car based on its brand, body type, mileage, engine details, registration status, and year — built with **scikit-learn** and deployed with **Streamlit**.

🔗 **Live demo:** _add your Streamlit Cloud link here after deploying_

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## Overview

This project trains and compares **Linear, Ridge, and Lasso Regression** models on a real-world used car sales dataset, then serves the best-performing model (Linear Regression) through an interactive Streamlit UI where anyone can enter a car's details and get an instant price estimate.

## Features

- 🧹 Cleaned real-world data (missing values dropped, engine-size outliers removed via IQR)
- 📈 Log-transformed target variable to correct for right-skewed prices
- 🔢 Label encoding for categorical features (Brand, Body, Engine Type, Registration)
- 🤖 Three regression models trained and compared on R² and RMSE
- 🎨 Clean, responsive Streamlit UI with instant predictions
- 📊 Sidebar showing model performance metrics

## Model Performance

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 0.819 | 0.386 |
| Ridge Regression | 0.819 | 0.386 |
| Lasso Regression | 0.537 | 0.618 |

*(scored on log-price; Linear Regression is the model shipped in the app)*

## Tech Stack

- **Python**, **pandas**, **NumPy** — data cleaning & processing
- **scikit-learn** — model training (Linear, Ridge, Lasso Regression)
- **Streamlit** — web app / UI
- **joblib** — model persistence

## Project Structure

```
├── app.py                          # Streamlit app
├── train.py                        # Training script (recreates the .pkl files)
├── requirements.txt                # Python dependencies
├── linear_regression_model.pkl     # Trained model
├── label_encoders.pkl              # Saved LabelEncoders for each categorical column
├── feature_columns.pkl             # Exact feature column order the model expects
└── metrics.pkl                     # R² / RMSE for all three models
```

## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/car-price-app.git
cd car-price-app

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Retraining the Model

If you want to retrain on updated data, place your CSV as `car_data.csv` in the project folder and run:

```bash
python train.py
```

This regenerates `linear_regression_model.pkl`, `label_encoders.pkl`, `feature_columns.pkl`, and `metrics.pkl`.

## Deployment

This app is designed for one-click deployment on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repo to GitHub
2. Go to share.streamlit.io → **New app**
3. Select this repo and set the main file to `app.py`
4. Deploy 🚀

## Dataset

Trained on the ["Real-life example" used car sales dataset](https://www.kaggle.com/datasets/smritisingh1997/car-salescsv) (Kaggle), containing ~4,300 car listings with Brand, Body type, Mileage, Engine Volume, Engine Type, Registration status, Year, and Price.

## Disclaimer

Predictions are statistical estimates based on historical data and should not be treated as a guaranteed valuation. Actual prices vary with condition, location, and market demand.

## License

MIT — feel free to use, modify, and build on this project.
