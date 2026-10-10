
import joblib
import pandas as pd
import streamlit as st

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# ----------------------------------
# 2. LOAD TRAINED ML MODEL
# ----------------------------------

saved_data = joblib.load("model/student_model.pkl")

model = saved_data["model"]
features = saved_data["features"]

# ----------------------------------
# 3. DARK THEME CSS
# ----------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #111827;
    color: #F9FAFB;
}

/* Main content width */
.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
h1 {
    color: #60A5FA !important;
    text-align: center;
    font-weight: 800;
}

/* Subheadings */
h2, h3 {
    color: #93C5FD !important;
}

/* Normal text */
.stApp p,
.stApp label,
.stMarkdown {
    color: #E5E7EB;
}

/* Form container */
div[data-testid="stForm"] {
    background-color: #1F2937;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #374151;
}

/* Input fields */
div[data-baseweb="input"] input {
    background-color: #111827;
    color: #FFFFFF;
    border: 1px solid #4B5563;
    border-radius: 8px;
}

/* Number input controls */
div[data-testid="stNumberInput"] button {
    background-color: #374151;
    color: #FFFFFF;
    border: 1px solid #4B5563;
}

/* Prediction button */
div[data-testid="stFormSubmitButton"] button {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    width: 100%;
    padding: 12px;
    font-weight: bold;
}

/* Button hover */
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF;
    border: 1px solid #60A5FA;
}

/* Result metric */
div[data-testid="stMetric"] {
    background-color: #1F2937;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #374151;
}

/* Divider */
hr {
    border-color: #374151;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# 4. PAGE HEADING
# ----------------------------------

st.title("🎓 Student Pass Prediction")

st.markdown(
    """
    <p style="
        text-align:center;
        color:#D1D5DB;
        font-size:17px;
    ">
        Machine Learning Based Academic Performance Prediction
    </p>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter the student's academic details below "
    "to predict the likely examination result."
)

st.info(
    "This application uses a trained Machine Learning model "
    "to predict Pass or Fail."
)

# ----------------------------------
# 5. STUDENT INPUT FORM
# ----------------------------------

with st.form("student_prediction_form"):

    st.subheader("Enter Student Details")

    attendance = st.number_input(
        "Attendance Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    study_hours = st.number_input(
        "Average Study Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.5
    )

    previous_marks = st.number_input(
        "Previous Examination Marks (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    assignments_completed = st.number_input(
        "Assignments Completed (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    submitted = st.form_submit_button(
        "🔍 Predict Student Result",
        use_container_width=True
    )

# ----------------------------------
# 6. PREDICTION
# ----------------------------------

if submitted:

    # Prepare input in the same format used during training
    student_data = pd.DataFrame(
        [[
            attendance,
            study_hours,
            previous_marks,
            assignments_completed
        ]],
        columns=features
    )

    # Predict Pass or Fail
    prediction = model.predict(student_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == "Pass":

        st.success(
            "🎉 Predicted Result: PASS"
        )

    else:

        st.error(
            "Predicted Result: FAIL"
        )

    # Calculate model probability for predicted class
    probabilities = model.predict_proba(student_data)[0]

    classes = list(model.classes_)

    predicted_index = classes.index(prediction)

    confidence = probabilities[predicted_index] * 100

    st.metric(
        label="Model Probability for Predicted Class",
        value=f"{confidence:.1f}%"
    )

    st.caption(
        "This probability is not a guarantee of the actual result. "
        "The model's reliability depends on the quality and quantity "
        "of its training data."
    )

    # Display entered details
    st.subheader("Student Details")

    st.dataframe(
        student_data,
        hide_index=True,
        use_container_width=True
    )

# ----------------------------------
# 7. FOOTER
# ----------------------------------

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9CA3AF;
        font-size:13px;
        margin-top:35px;
    ">
        Student Pass Prediction | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
