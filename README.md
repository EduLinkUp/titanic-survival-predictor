# 🚢 Titanic Survival Predictor

### Machine Learning Classification Project for Passenger Survival Prediction

**Binary Classification** · **Feature Engineering** · **Model Comparison** · **Hyperparameter Tuning** · **Streamlit Deployment**

Built using Python & Scikit-learn | End-to-End ML Pipeline

---

> **"Not all passengers had equal chances of survival."**  
> This project analyzes historical passenger data and builds a machine learning model to predict survival probability on the Titanic.

---

## 📋 Table of Contents

- [The Challenge](#-the-challenge)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering-approach)
- [Models Implemented](#-models-implemented)
- [Model Evaluation](#-model-evaluation--monitoring)
- [Feature Importance](#-feature-importance-insights)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-repository-structure)
- [Run Locally](#-how-to-run-locally)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

## 🔴 The Challenge

The sinking of the Titanic remains one of the most famous maritime disasters in history.

Given passenger details such as:

- Age  
- Gender  
- Passenger Class  
- Fare  
- Family Members Aboard  
- Port of Embarkation  

The objective is to build a **supervised machine learning model** that predicts:

Did the passenger survive?  
0 → Did Not Survive  
1 → Survived  

This is a **binary classification problem**.

---

## 📊 Exploratory Data Analysis

Before modeling, extensive **Exploratory Data Analysis (EDA)** was performed to understand patterns and relationships.

### Analysis Performed

- Survival distribution analysis
- Gender vs survival comparison
- Passenger class survival trends
- Missing value inspection
- Feature correlation visualization

### 🔎 Key Observations

- Female passengers had significantly higher survival rates.
- First-class passengers had higher survival probability.
- Higher ticket fares showed positive correlation with survival.
- Age and family size influenced survival probability.

---

## ⚙ Feature Engineering Approach

Feature engineering significantly improved model performance.

### ✅ Handling Missing Values

| Feature | Strategy |
|---|---|
| Age | Filled using median grouped by passenger class |
| Embarked | Filled using mode |
| Cabin | Dropped due to excessive missing values |

---

### ✅ Feature Creation

#### 1️⃣ FamilySize
FamilySize = SibSp + Parch + 1


- Captures family presence impact
- Improves prediction performance

#### 2️⃣ Title Extraction

- Extracted titles (Mr, Mrs, Miss, etc.) from passenger names
- Rare titles grouped into **"Rare"**
- Improved demographic representation

---

### ✅ Encoding

- Label Encoding → Sex
- One-Hot Encoding → Embarked, Title

---

### ✅ Feature Scaling

StandardScaler applied to:

- Age  
- Fare  
- FamilySize  

Improves performance for linear and distance-based models.

---

## 🤖 Models Implemented

The following machine learning models were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

### Hyperparameter Optimization

- GridSearchCV
- 5-Fold Cross Validation

---

## 📈 Model Evaluation & Monitoring

### Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross-Validation Score

---

## 🏆 Final Selected Model: Random Forest

The tuned Random Forest model achieved strong and balanced performance.

### Performance Characteristics

- High overall accuracy
- Balanced precision and recall
- Reduced overfitting through cross-validation
- Clear feature importance ranking

---

## 📊 Feature Importance Insights

Top influential features:

- Sex
- Passenger Class (Pclass)
- Fare
- Age
- FamilySize

These findings align with historical survival patterns.

---

## 🚀 Deployment

The trained model was serialized using:

- **pickle**

A web application was built using **Streamlit** to:

- Accept passenger input data
- Apply preprocessing pipeline
- Scale features
- Generate survival prediction
- Display user-friendly output

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deployment | Streamlit |
| Model Storage | Pickle |

---

## 📂 Repository Structure
titanic-survival-predictor/
│
├── data/
├── notebook/
│ └── titanic_survival_analysis.ipynb
├── app.py
├── titanic_model.pkl
├── requirements.txt
├── README.md
└── LICENSE

---

## ▶ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/EduLinkUp/titanic-survival-predictor.git

# Navigate to project folder
cd titanic-survival-predictor

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

---

### 🎯 Key Learnings

- Feature engineering significantly improves model performance.
- Cross-validation helps prevent overfitting.
- Comparing multiple models improves decision confidence.
- Clean project structure enhances readability and usability.
- End-to-end ML pipelines include preprocessing → training → deployment.

---

### 👩‍💻 Author

**Malleswarapu Sriya**  
Machine Learning Enthusiast | Data Science Student

---