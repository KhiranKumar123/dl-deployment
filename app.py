import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONSTANTS & GLOBALS
# ==========================================
OPTION_COLS = ["A", "B", "C", "D", "E"]
NEGATION_WORDS = {"no", "not", "never", "none", "neither", "nor", "cannot", "without", "n't", "false", "unlike"}
ABSOLUTE_WORDS = {"always", "all", "every", "entirely", "absolutely", "completely", "only", "must", "true"}
UNITS_WORDS = {"kg", "cm", "mm", "m", "km", "hz", "mol", "nm", "v", "w", "ev", "kj", "pa", "atm", "sec", "min", "hr"}

# ==========================================
# 2. HELPER FUNCTIONS (Paste from your Notebook)
# ==========================================
# Copy and paste these exact functions from your Jupyter notebook here:
# - clean_text()
# - strip_boilerplate()
# - get_longest_common_prefix()
# - prepare_text_data()
# - basic_features()
# - similarity_features()
# - advanced_features()
# - extract_features()

# ==========================================
# 3. LOAD MODELS
# ==========================================
@st.cache_resource 
def load_models():
    # Make sure the paths match exactly where you uploaded them in GitHub
    tfidf = joblib.load("tfidf.pkl")
    joint_tfidf = joblib.load("joint_tfidf.pkl")
    svd = joblib.load("svd.pkl")
    lgb_models = joblib.load("lightgbm_models.pkl")
    return tfidf, joint_tfidf, svd, lgb_models

tfidf, joint_tfidf, svd, lgb_models = load_models()

# ==========================================
# 4. USER INTERFACE
# ==========================================
st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")
st.title("🧠 Smart MCQ Solver")
st.markdown("Enter your question and 5 options to get the most probable answer.")

with st.form("prediction_form"):
    prompt = st.text_area("Question Prompt", placeholder="Type the question here...")
    
    col1, col2 = st.columns(2)
    with col1:
        option_a = st.text_input("Option A")
        option_b = st.text_input("Option B")
        option_c = st.text_input("Option C")
    with col2:
        option_d = st.text_input("Option D")
        option_e = st.text_input("Option E")
        
    submit_button = st.form_submit_button("Predict Answer")

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if submit_button:
    if prompt and option_a and option_b and option_c and option_d and option_e:
        with st.spinner("Calculating features and running models..."):
            
            # Package the user input into a DataFrame
            input_df = pd.DataFrame([{
                "prompt": prompt,
                "A": option_a, "B": option_b, "C": option_c, "D": option_d, "E": option_e
            }])

            # Run feature extraction
            # (Note: make sure your prepare_text_data uses the loaded tfidf and svd variables)
            text_data = prepare_text_data(input_df)
            X_dense_df, _, _ = extract_features(input_df, text_data)

            # Create Joint TF-IDF features
            joint_texts = []
            clean_p = strip_boilerplate(clean_text(prompt))
            for col in OPTION_COLS:
                joint_texts.append(clean_p + " " + clean_text(input_df[col].iloc[0]))
            
            X_joint = joint_tfidf.transform(joint_texts)

            # Combine into final sparse matrix
            X_dense = csr_matrix(X_dense_df.values)
            X_final = hstack([X_dense, X_joint]).tocsr()

            # Average the predictions across your LightGBM models
            preds = np.zeros(5)
            for model in lgb_models:
                preds += model.predict_proba(X_final)[:, 1] / len(lgb_models)

            # Get the highest probability
            best_idx = np.argmax(preds)
            best_prediction = OPTION_COLS[best_idx]

            st.success("Prediction complete!")
            st.metric(label="Predicted Best Option", value=best_prediction)
    else:
        st.error("Please enter a question and all 5 options.")
