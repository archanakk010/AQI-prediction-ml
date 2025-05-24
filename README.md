# Air Quality Analysis and AQI Prediction Using Machine Learning

This project aims to predict Air Quality Index (AQI) levels using real-time monitoring data and advanced machine learning models. By analyzing pollutants such as PM2.5, PM10, and Ozone, the model provides accurate insights into environmental health conditions, aiding decision-makers, researchers, and the public.

---


##  Key Highlights

-  Real-time air pollution data analysis
-  Visualizations and EDA for actionable insights
-  Multiple ML models including XGBoost, Random Forest, and Gradient Boosting
-  Performance tuning via feature selection & hyperparameter optimization
-  Deployment-ready with a Flask API

---

##  Dataset Information

- **Source**: U.S. EPA / AirNow API (Public)
- **File**:  https://drive.google.com/file/d/1qEhnxnsgnBe4i6M9CNcGev0LaFsyu3kf/view?usp=sharing
- **Features**:
 `PM2.5`, `PM10`, `OZONE`,`PM2.5AQI`, `PM10AQI`, `OzoneAQI`,`Latitude`, `Longitude`, `StateName`, `ReportingArea`etc
    

---

##  Project Objectives

- Clean and preprocess air quality data
- Explore and visualize pollutant distributions and correlations
- Build predictive models for AQI values
- Optimize model performance using feature selection & tuning
- Deploy a prediction pipeline using Flask

---

##  Machine Learning Workflow

### 1. Data Preprocessing
- Missing value imputation
- Outlier handling using boxplots
- Label encoding & feature scaling

### 2. Exploratory Data Analysis
- Distribution plots
- Correlation heatmaps
- AQI category interpretation

### 3. Feature Selection
- `SelectKBest`
- `Recursive Feature Elimination (RFE)`
- `Lasso Regression`

### 4. Model Building
- `XGBRegressor` (best performance)
- Gradient Boosting, Random Forest, Linear Regression, etc.

### 5. Evaluation Metrics
| Metric | Score |
|--------|-------|
| R²     | 0.974 |
| MSE    | 4.97 |
| RMSE   | 2.23 |
| MAE    | 1.19 |

---
##  Conclusion

This project demonstrates that machine learning can effectively predict AQI from monitored pollution data. Accurate AQI predictions can assist citizens, policymakers, and health authorities in taking timely action to reduce exposure and improve air quality awareness.



##  Model Deployment

The final model is deployed using a **Flask API** with support for JSON input. Use tools like **Postman**

