# 🌱 Dry Bean Classification using Machine Learning

An end-to-end Machine Learning project that classifies dry bean seeds into different bean varieties based on their physical and geometric characteristics.

This project focuses on analyzing bean morphology and building classification models capable of accurately identifying bean types using supervised machine learning techniques.

---

# 📌 Project Overview

Bean classification is an important task in agriculture and food processing industries. Different bean varieties possess unique physical properties that can be measured and analyzed.

The primary objective of this project is to:

✅ Analyze bean characteristics

✅ Explore feature relationships

✅ Train multiple classification models

✅ Compare model performance

✅ Accurately classify bean varieties

---

# 🧠 Machine Learning Problem Type

This is a **Supervised Machine Learning Classification Problem**.

### Target Variable:

```text id="bean01"
Bean Class
```

The target variable represents the bean variety/category.

---

# 📂 Dataset Information

The dataset contains measurements extracted from images of dry beans.

### Features Included:

* Area
* Perimeter
* Major Axis Length
* Minor Axis Length
* Aspect Ratio
* Eccentricity
* Convex Area
* Equivalent Diameter
* Extent
* Solidity
* Roundness
* Compactness
* Shape Factors

### Target Classes:

* BARBUNYA
* BOMBAY
* CALI
* DERMASON
* HOROZ
* SEKER
* SIRA

---

# 🔍 Exploratory Data Analysis

Several important insights were discovered during EDA:

✅ Different bean varieties show distinct geometric characteristics

✅ Area and perimeter vary significantly across bean classes

✅ Shape-related features contribute heavily to classification

✅ Certain bean varieties share similar patterns, making classification challenging

✅ Feature distributions reveal clear separation among some classes

### Data Preprocessing Included:

* Missing value analysis
* Duplicate checking
* Feature scaling
* Label encoding
* Correlation analysis
* Train-test splitting

---

# ⚙️ Models Used

The following Machine Learning classification models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes
* Gradient Boosting Classifier
* AdaBoost Classifier
* XGBoost Classifier

### Evaluation Metrics

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix
* Classification Report

---

# 📊 Model Performance

Multiple classification algorithms were compared to identify the best-performing model.

The final selected model achieved excellent classification accuracy and successfully distinguished between different bean varieties.

Model selection was based on:

* Highest Test Accuracy
* Better Generalization
* Stable Cross-Validation Performance
* Reduced Overfitting

---

# 🚀 Features

✨ Dry bean variety prediction

✨ Multi-class classification

✨ Feature importance analysis

✨ Confusion matrix visualization

✨ Classification report generation

✨ Model comparison dashboard

✨ End-to-end ML workflow

---

# 📈 Visualizations Included

### 📌 Bean Class Distribution

Shows the number of samples available for each bean variety.

### 📌 Feature Distribution Analysis

Visualizes numerical feature patterns across classes.

### 📌 Correlation Heatmap

Displays relationships between geometric characteristics.

### 📌 Pairwise Feature Analysis

Examines class separation among important features.

### 📌 Feature Importance

Highlights the most influential features affecting classification.

### 📌 Confusion Matrix

Evaluates classification performance for each bean category.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook
* Joblib

---

# 🔮 Future Improvements

Possible future enhancements:

* Hyperparameter tuning
* Deep learning-based image classification
* Real-time bean classification system
* Deployment using Streamlit
* Integration with agricultural quality-control systems
* Larger and more diverse bean datasets

---

# 🎯 Conclusion

This project demonstrates how Machine Learning classification techniques can accurately identify dry bean varieties using geometric and morphological features.

The project combines:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Classification Modeling
* Model Evaluation
* Performance Comparison

into a complete agricultural machine learning solution.
