# House Price Prediction Model

A machine learning project that predicts house prices using various regression algorithms, achieving up to 74% accuracy with Linear Regression.

## 📊 Project Overview

This project implements multiple machine learning models to predict house prices based on various property features. The dataset contains 4,600 house records with detailed property information including location, size, condition, and historical data.

## 🏠 Dataset Features

The dataset includes the following key features:
- **Property Details**: Bedrooms, bathrooms, living area (sqft), lot size
- **Structural Features**: Number of floors, waterfront access, view rating, condition
- **Location Data**: Street address, city, state/zip, country
- **Historical Information**: Year built, year renovated, sale date
- **Target Variable**: Sale price

### Dataset Statistics
- **Total Records**: 4,600 houses
- **Average Price**: $551,963
- **Price Range**: $0 - $26,590,000
- **Average Living Area**: 2,139 sqft
- **Average Bedrooms**: 3.0
- **Average Bathrooms**: 2.5

## 🔧 Data Preprocessing

The following preprocessing steps were applied:
- **Feature Engineering**: Created `year_sold`, `house_age`, and `has_been_renovated` features
- **Data Cleaning**: Handled missing values and outliers
- **Categorical Encoding**: Processed location-based features
- **Feature Selection**: Selected relevant numerical and categorical features

## 🤖 Machine Learning Models

Four different regression models were implemented and evaluated:

### Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** | 66,894.08 | 104,615.67 | **0.74** |
| Random Forest | 77,455.14 | 120,017.94 | 0.65 |
| Gradient Boosting | 81,955.89 | 120,997.40 | 0.65 |
| Decision Tree | 100,565.50 | 149,387.33 | 0.46 |

### Key Findings
- **Linear Regression** achieved the best performance with 74% R² score
- **Mean Absolute Error** of $66,894 for the best model
- **Root Mean Square Error** of $104,616 for linear regression
- Decision Tree showed the highest error rates, indicating potential overfitting

## 📈 Model Evaluation Metrics

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices
- **Root Mean Square Error (RMSE)**: Square root of average squared differences
- **R² Score**: Coefficient of determination indicating model's explanatory power

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## 📁 Project Structure

```
house-price-prediction/
│
├── house_price_prediction.ipynb    # Main Jupyter notebook
├── house data.csv                  # Dataset file                       
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Project
1. Clone the repository
2. Ensure `house data.csv` is in the project directory
3. Open `house_price_prediction.ipynb` in Jupyter Notebook
4. Run all cells to reproduce the analysis

### Usage
```python
# Load and explore the dataset
df = pd.read_csv("house data.csv")
df.head()

# Train models and evaluate performance
# (See notebook for detailed implementation)
```

## 📊 Results Summary

The **Linear Regression model** emerged as the top performer with:
- ✅ **74% accuracy** (R² score)
- ✅ **Lowest prediction error** ($66,894 MAE)
- ✅ **Best generalization** across the test set

This suggests that house prices in this dataset follow relatively linear relationships with the selected features, making simpler models more effective than complex ensemble methods.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements. Areas for contribution:
- Enhanced feature engineering
- Additional model implementations
- Improved data visualization
- Performance optimization
 
*This project demonstrates practical application of machine learning for real estate price prediction, showcasing data preprocessing, model comparison, and performance evaluation techniques.*

##  Output: 

<img width="1908" height="995" alt="Image" src="https://github.com/user-attachments/assets/de8c4a19-4b5c-4708-a7f4-11fdb665bab6" />

