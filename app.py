import pickle
import pandas as pd
import streamlit as st

# ======================================
# SAFE MODEL LOADING
# ======================================
try:
    with open("titanic_model.pkl", "rb") as f:
        package = pickle.load(f)

    model = package["model"]
    le = package["label_encoder"]
    scaler = package["scaler"]
    model_columns = package["model_columns"]

except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'titanic_model.pkl' exists.")
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Feature integrity check
if model.n_features_in_ != len(model_columns):
    st.error("❌ Model feature mismatch detected.")
    st.stop()

# ======================================
# STREAMLIT UI
# ======================================
st.title("🚢 Titanic Survival Predictor")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
Fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=50.0)

SibSp = st.number_input("Siblings / Spouses Aboard", 0, 20, 0)
Parch = st.number_input("Parents / Children Aboard", 0, 20, 0)

Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
Title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

# ======================================
# INPUT VALIDATION FUNCTION
# ======================================
def validate_inputs(age, fare, sibsp, parch):

    if age < 0 or age > 120:
        return "Age must be between 0 and 120."

    if fare < 0:
        return "Fare cannot be negative."

    if sibsp < 0 or sibsp > 20:
        return "Siblings/Spouses must be between 0 and 20."

    if parch < 0 or parch > 20:
        return "Parents/Children must be between 0 and 20."

    if sibsp + parch > 15:
        return "Total family members seems unrealistic."

    return None


# ======================================
# PREDICTION FUNCTION
# ======================================
def predict_survival(age, sex, pclass, fare, sibsp, parch, embarked, title):

    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    input_data["Pclass"] = pclass
    input_data["Sex"] = le.transform([sex])[0]
    input_data["Age"] = age
    input_data["SibSp"] = sibsp
    input_data["Parch"] = parch
    input_data["Fare"] = fare
    input_data["FamilySize"] = sibsp + parch + 1

    if f"Embarked_{embarked}" in input_data.columns:
        input_data[f"Embarked_{embarked}"] = 1

    if f"Title_{title}" in input_data.columns:
        input_data[f"Title_{title}"] = 1

    # Scale numerical features
    input_data[["Age", "Fare", "FamilySize"]] = scaler.transform(
        input_data[["Age", "Fare", "FamilySize"]]
    )

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    return prediction[0], probability


# ======================================
# PREDICT BUTTON
# ======================================
if st.button("Predict Survival"):

    validation_error = validate_inputs(Age, Fare, SibSp, Parch)

    if validation_error:
        st.error(f"⚠ {validation_error}")

    else:
        try:
            result, probability = predict_survival(
                age=Age,
                sex=Sex,
                pclass=Pclass,
                fare=Fare,
                sibsp=SibSp,
                parch=Parch,
                embarked=Embarked,
                title=Title
            )

            if result == 1:
                st.success("🎉 The passenger would likely SURVIVE.")
            else:
                st.error("❌ The passenger would likely NOT survive.")

            st.info(f"📊 Survival Probability: {probability:.2%}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")