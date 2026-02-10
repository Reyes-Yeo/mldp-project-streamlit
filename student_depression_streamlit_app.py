# import joblib
# import streamlit as st
# import numpy as np
# import pandas as pd

# ## Load trained model
# model = joblib.load("testing_model.pkl")

# ## Streamlit app
# st.title("Student Depression Prediction")

# ## Define the input options
# Gender = ['Male', 'Female']
# City = ['Visakhapatnam','Bangalore','Srinagar','Varanasi','Jaipur','Pune','Thane',
#  'Chennai','Nagpur','Nashik','Vadodara','Kalyan','Rajkot','Ahmedabad',
#  'Kolkata','Mumbai','Lucknow','Indore','Surat','Ludhiana','Bhopal',
#  'Meerut','Agra','Ghaziabad','Hyderabad','Vasai-Virar','Kanpur', 'Patna',
#  'Faridabad', 'Delhi', 'Khaziabad']
# storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09']


# ## User inputs
# town_selected = st.selectbox("Select Town", towns)
# flat_type_selected = st.selectbox("Select Flat Type", flat_types)
# storey_range_selected = st.selectbox("Select Storey", storey_ranges)
# floor_area_selected = st.slider("Select Floor Area (sqm)", 
#                                 min_value=30, 
#                                 max_value=200, 
#                                 value=70)

# ## Predict button
# if st.button("Predict HDB price"):

#     ## Create dict for input features
#     input_data = {
#         'town': town_selected,
#         'flat_type': flat_type_selected,
#         'storey_range': storey_range_selected,
#         'floor_area': floor_area_selected
#     }

#     ## Convert input data to a DataFrame
#     df_input = pd.DataFrame({
#         'town': [town_selected],
#         'flat_type': [flat_type_selected],
#         'storey_range': [storey_range_selected],
#         'floor_area': [floor_area_selected]
#     })

#     ## One-hot encoding
#     df_input = pd.get_dummies(df_input, 
#                               columns = ['town', 'flat_type', 'storey_range']
#                              )
    
#     # df_input = df_input.to_numpy()

#     df_input = df_input.reindex(columns = model.feature_names_in_,
#                                 fill_value=0)



#     ## Predict
#     y_unseen_pred = model.predict(df_input)[0]
#     st.success(f"Predicted Resale Price: ${y_unseen_pred:,.2f}")

# ## Page design
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: url("https://www.shutterstock.com/shutterstock/videos/1025418011/thumb/1.jpg");
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# import joblib
# import streamlit as st
# import numpy as np
# import pandas as pd

# ## Load trained model
# model = joblib.load("final_model.pkl")
# scaler = joblib.load("scaler.pkl")
# feature_cols = joblib.load("feature_cols.pkl")


# ## Streamlit app
# st.title("Student Depression Prediction")

# st.write(
#     "This application predicts whether a student may be at risk of depression "
#     "based on academic, financial, and lifestyle factors."
# )

# ## Define the input options
# Gender = ['Male', 'Female']

# ## User inputs
# st.header("Enter Student Information")

# st.caption(
#     "Please answer honestly. The values are self-reported and used only for prediction."
# )

# gender_selected = st.selectbox("Gender", Gender)

# academic_pressure = st.slider(
#     "Academic Pressure (1 = Low, 5 = High)",
#     min_value=1,
#     max_value=5,
#     value=3
# )

# financial_stress = st.slider(
#     "Financial Stress (1 = Low, 5 = High)",
#     min_value=1,
#     max_value=5,
#     value=3
# )

# sleep_duration = st.slider(
#     "Sleep Duration (hours)",
#     min_value=3,
#     max_value=10,
#     value=7
# )

# study_satisfaction = st.slider(
#     "Study Satisfaction (1 = Low, 5 = High)",
#     min_value=1,
#     max_value=5,
#     value=3
# )

# ## Predict button
# if st.button("Predict Depression Risk"):

#     ## Convert input data to DataFrame
#     df_input = pd.DataFrame({
#         'Gender': [gender_selected],
#         'Academic Pressure': [academic_pressure],
#         'Financial Stress': [financial_stress],
#         'Sleep Duration': [sleep_duration],
#         'Study Satisfaction': [study_satisfaction]
#     })

#     ## One-hot encoding
#     df_input = pd.get_dummies(df_input)

#     ## Align input features with trained model
#     df_input = df_input.reindex(
#         # columns=model.feature_names_in_,
#         columns=feature_cols,
#         fill_value=0
#     )

#     ## Predict
#     y_unseen_pred = model.predict(df_input)[0]

#     ## Output result
#     if y_unseen_pred == 1:
#         st.error(
#             "⚠️ High risk of depression detected.\n\n"
#             "Consider seeking academic or counselling support."
#         )
#     else:
#         st.success(
#             "✅ Low risk of depression detected."
#         )

# ## Page design
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f9f9f9;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


import joblib
import streamlit as st
import pandas as pd

# =========================
# Load trained artifacts
# =========================
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# Decision threshold chosen from evaluation
THRESHOLD = 0.40

# =========================
# Streamlit app
# =========================
st.title("Student Depression Prediction")

st.write(
    "This application predicts whether a student may be at risk of depression "
    "based on academic, financial, and lifestyle factors."
)

Gender = ["Male", "Female"]

st.header("Enter Student Information")
st.caption("Please answer honestly. The values are self-reported and used only for prediction.")

gender_selected = st.selectbox("Gender", Gender)

academic_pressure = st.slider(
    "Academic Pressure (1 = Low, 5 = High)", 1, 5, 3
)

financial_stress = st.slider(
    "Financial Stress (1 = Low, 5 = High)", 1, 5, 3
)

sleep_duration = st.slider(
    "Sleep Duration (hours)", 3, 10, 7
)

study_satisfaction = st.slider(
    "Study Satisfaction (1 = Low, 5 = High)", 1, 5, 3
)

# =========================
# Predict
# =========================
if st.button("Predict Depression Risk"):

    # 1) Convert inputs to DataFrame
    df_input = pd.DataFrame({
        "Gender": [gender_selected],
        "Academic Pressure": [academic_pressure],
        "Financial Stress": [financial_stress],
        "Sleep Duration": [sleep_duration],
        "Study Satisfaction": [study_satisfaction]
    })

    # 2) One-hot encoding (same idea as training)
    df_input = pd.get_dummies(df_input)

    # 3) Align input columns to training feature order
    df_input = df_input.reindex(columns=feature_cols, fill_value=0)

    # 4) Scale input (CRUCIAL - model trained on scaled features)
    df_input_scaled = scaler.transform(df_input)

    # 5) Predict probability of Depressed=1
    prob_depressed = model.predict_proba(df_input_scaled)[0][1]

    # 6) Apply threshold
    y_unseen_pred = 1 if prob_depressed >= THRESHOLD else 0

    st.write(f"Prediction probability (Depressed=1): **{prob_depressed:.3f}**")
    st.write(f"Decision threshold: **{THRESHOLD}**")

    # Output result
    if y_unseen_pred == 1:
        st.error(
            "⚠️ High risk of depression detected.\n\n"
            "Consider seeking academic or counselling support."
        )
    else:
        st.success("✅ Low risk of depression detected.")

# Page design
st.markdown(
    """
    <style>
    .stApp { background-color: #f9f9f9; }
    </style>
    """,
    unsafe_allow_html=True
)

