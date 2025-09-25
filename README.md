# Rain Prediction in Australia using Machine Learning

This project predicts whether it will rain tomorrow in Australia using historical weather data and different machine learning algorithms.

##  Project Overview
- Dataset: [Rain in Australia dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- Goal: Predict the binary target **RainTomorrow** (`Yes` or `No`)
- Techniques: Data preprocessing, feature engineering, model training, evaluation
- Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

##  Workflow
1. **Data Loading & Exploration**  
   - Handle missing values  
   - Summary statistics & distributions  
   - Visualizations of weather patterns  

2. **Data Preprocessing**  
   - Encode categorical variables  
   - Feature scaling (MinMax/Standard Scaler)  
   - Train-test split  

3. **Modeling**  
   Algorithms used:
   - Logistic Regression  
   - K- Nearest Neighbor 
   - Support Vector Machine
   - Decision Tree

4. **Evaluation Metrics**  (Decision Tree)
   - Accuracy  = 89.04 %
   - Precision, Recall, F1-score  
   - ROC-AUC Curve  = 0.8908

5. **Results & Insights**  
   - Best performing model and its metrics  
   - Feature importance analysis  

##  Results
- Logistic Regression: (add results)  
- Random Forest: (add results)  
- XGBoost: (add results)

## Future Improvements
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Try deep learning models (LSTM for time-based weather data)
- Deploy with Streamlit/Flask   

##  How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Rain-Prediction-Australia-using-ML-algorithms.git
   cd Rain-Prediction-Australia-using-ML-algorithms
