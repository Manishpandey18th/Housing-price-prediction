# Housing Prices Prediction

This repository contains a Jupyter notebook and a Streamlit app to analyze and predict housing prices based on a real estate dataset.

## Files

- `house.ipynb`: Jupyter notebook performing exploratory data analysis (EDA), preprocessing, feature engineering, model training (Linear Regression, Random Forest), evaluation, and saving the best model.
- `app.py`: Streamlit app to load the trained model and predict house prices based on user inputs.
- `requirements.txt`: Python dependencies required to run the project.
- `housing_prices.csv`: Dataset (not included in repository; place in project root).

## Dataset

The dataset (`housing_prices.csv`) contains the following features:

- `price` (target): House price in ₹
- `area`: Area in square feet
- `bedrooms`, `bathrooms`, `stories`, `parking`: Numeric features
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`: Binary (yes/no)
- `furnishingstatus`: Categorical (furnished, semi-furnished, unfurnished)
- `Location`: Categorical (various locations)

## Setup

### 1. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

### 2. Install Dependencies
```bash
pip install -r requirements.txt

### 3. Place Dataset

- Save `housing_prices.csv` in the project root directory.
- Alternatively, update the DATA_PATH variable in house.ipynb to point to your dataset location.

## 4. Run the Notebook

- Open `housing_price_predicts.ipynb` in Jupyter (e.g., jupyter notebook).
- Run the cells in order to perform EDA, preprocess data, train models, and save the best model to `./output/best_model.joblib`.

### 5. Run the Streamlit App

- After running the notebook, launch the app:
```bash
streamlit run app.py

- Input feature values in the app to get price predictions. 
- Ensure inputs match the dataset's ranges and categories.

### Analysis and Modeling

- `EDA`: Visualizes price distribution, price per square foot, and correlations.
- `Preprocessing`: Handles missing values (imputation), encodes categorical features, scales numeric features, and removes outliers (1st/99th percentiles).
- `Feature Engineering`: Adds price_per_sqft for analysis.
- `Models`: Trains Linear Regression and Random Forest, evaluates using MAE, RMSE, and R².
- `Output`: Saves predictions, model, and evaluation report to ./output.

### Notes & Next Steps

- `Improvements`: Consider hyperparameter tuning (e.g., GridSearchCV), adding interaction terms, or incorporating more data.
- `Deployment`: Containerize with Docker or deploy to Streamlit Community Cloud for sharing.
- `Limitations`: Model performance depends on dataset quality; some locations may have limited samples.