import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ==========================================
# 1. CONSTANTS & GLOBALS
# ==========================================
OPTION_COLS = ["A", "B", "C", "D", "E"]
NEGATION_WORDS = {"no", "not", "never", "none", "neither", "nor", "cannot", "without", "n't", "false", "unlike"}
ABSOLUTE_WORDS = {"always", "all", "every", "entirely", "absolutely", "completely", "only", "must", "true"}
UNITS_WORDS = {"kg", "cm", "mm", "m", "km", "hz", "mol", "nm", "v", "w", "ev", "kj", "pa", "atm", "sec", "min", "hr"}
STOPWORDS = set(ENGLISH_STOP_WORDS)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def strip_boilerplate(text):
    text = clean_text(text)
    patterns = [
        r"pick the best possible answer:?", r"choose the correct answer:?",
        r"which of the following is correct:?", r"which of the following statements is true:?",
        r"select the best option:?", r"what is the correct answer:?"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()

def get_longest_common_prefix(strings):
    if len(strings) == 0: return ""
    s1, s2 = min(strings), max(strings)
    for i, c in enumerate(s1):
        if c != s2[i]: return s1[:i]
    return s1

def prepare_text_data(df):
    data = {}
    data["prompt_clean"] = df["prompt"].fillna("").apply(clean_text)
    data["prompt_core"] = df["prompt"].fillna("").apply(strip_boilerplate)
    data["prompt_tfidf"] = tfidf.transform(data["prompt_core"])
    data["prompt_svd"] = svd.transform(data["prompt_tfidf"])
    data["options_clean"] = {}
    data["options_tfidf"] = {}
    data["options_svd"] = {}
    
    for col in OPTION_COLS:
        texts = df[col].fillna("").astype(str).apply(clean_text)
        data["options_clean"][col] = texts
        tfidf_vec = tfidf.transform(texts)
        data["options_tfidf"][col] = tfidf_vec
        data["options_svd"][col] = svd.transform(tfidf_vec)
        
    data["option_set"] = df[OPTION_COLS].apply(lambda x: " | ".join(str(v).strip() for v in x), axis=1).values
    return data

def basic_features(row_idx, option_idx, option_label, text_data):
    feat = {}
    prompt_text = text_data["prompt_core"].iloc[row_idx]
    prompt_words = prompt_text.lower().split()
    prompt_word_cnt = max(len(prompt_words), 1)
    prompt_char_cnt = max(len(prompt_text), 1)
    
    option_text = text_data["options_clean"][option_label].iloc[row_idx]
    option_words = option_text.lower().split()
    option_word_cnt = max(len(option_words), 1)
    option_char_cnt = len(option_text)
    
    row_options = [text_data["options_clean"][c].iloc[row_idx] for c in OPTION_COLS]
    char_lengths = [len(x) for x in row_options]
    word_lengths = [len(x.split()) for x in row_options]
    
    mean_char = np.mean(char_lengths)
    std_char = np.std(char_lengths) + 1e-6
    mean_word = np.mean(word_lengths)
    std_word = np.std(word_lengths) + 1e-6
    ranks = pd.Series(char_lengths).rank(ascending=False, method="min").values
    
    feat["char_len"] = option_char_cnt
    feat["word_len"] = option_word_cnt
    feat["len_rank"] = ranks[option_idx]
    feat["is_longest"] = int(ranks[option_idx] == 1)
    feat["is_shortest"] = int(ranks[option_idx] == 5)
    feat["char_zscore"] = (option_char_cnt - mean_char) / std_char
    feat["word_zscore"] = (option_word_cnt - mean_word) / std_word
    feat["char_ratio"] = option_char_cnt / (mean_char + 1e-6)
    feat["word_ratio"] = option_word_cnt / (mean_word + 1e-6)
    feat["prompt_char_ratio"] = option_char_cnt / prompt_char_cnt
    feat["prompt_word_ratio"] = option_word_cnt / prompt_word_cnt
    feat["lcp_length"] = len(get_longest_common_prefix(row_options))
    return feat

def similarity_features(row_idx, option_idx, option_label, text_data):
    feat = {}
    prompt_words = set(text_data["prompt_core"].iloc[row_idx].lower().split())
    prompt_vec = text_data["prompt_tfidf"][row_idx]
    prompt_svd = text_data["prompt_svd"][row_idx]
    
    option_text = text_data["options_clean"][option_label].iloc[row_idx]
    option_words = set(option_text.lower().split())
    option_vec = text_data["options_tfidf"][option_label][row_idx]
    option_svd = text_data["options_svd"][option_label][row_idx]
    
    intersection = len(prompt_words & option_words)
    union = len(prompt_words | option_words)
    
    feat["word_overlap"] = intersection
    feat["overlap_ratio"] = intersection / (min(len(prompt_words), len(option_words)) + 1e-6)
    feat["jaccard"] = intersection / (union + 1e-6)
    feat["tfidf_cosine"] = cosine_similarity(prompt_vec, option_vec)[0, 0]
    feat["svd_cosine"] = np.dot(prompt_svd, option_svd) / (np.linalg.norm(prompt_svd) * np.linalg.norm(option_svd) + 1e-6)
    
    from scipy.sparse import vstack
    option_vectors = vstack([text_data["options_tfidf"][c][row_idx] for c in OPTION_COLS])
    sim_matrix = cosine_similarity(option_vectors)
    sims = np.delete(sim_matrix[option_idx], option_idx)
    
    feat["opt_sim_mean"] = np.mean(sims)
    feat["opt_sim_max"] = np.max(sims)
    feat["opt_sim_min"] = np.min(sims)
    feat["opt_sim_std"] = np.std(sims)
    
    row_options = [text_data["options_clean"][c].iloc[row_idx] for c in OPTION_COLS]
    prefix = get_longest_common_prefix(row_options)
    feat["common_prefix_len"] = len(prefix)
    feat["option_prefix_ratio"] = len(prefix) / (len(option_text) + 1e-6)
    return feat

def advanced_features(row_idx, option_idx, option_label, text_data):
    feat = {}
    option_text = text_data["options_clean"][option_label].iloc[row_idx]
    words = option_text.lower().split()
    word_count = max(len(words), 1)
    char_count = max(len(option_text), 1)
    
    neg_count = sum(w in NEGATION_WORDS for w in words)
    feat["negation_count"] = neg_count
    feat["negation_ratio"] = neg_count / word_count
    
    abs_count = sum(w in ABSOLUTE_WORDS for w in words)
    feat["absolute_count"] = abs_count
    feat["absolute_ratio"] = abs_count / word_count
    
    stop_count = sum(w in STOPWORDS for w in words)
    feat["stopword_ratio"] = stop_count / word_count
    
    digit_count = sum(c.isdigit() for c in option_text)
    feat["digit_ratio"] = digit_count / char_count
    feat["contains_digit"] = int(digit_count > 0)
    feat["contains_unit"] = int(any(w in UNITS_WORDS for w in words))
    
    alpha_count = sum(c.isalpha() for c in option_text)
    upper_count = sum(c.isupper() for c in option_text)
    feat["uppercase_ratio"] = upper_count / (alpha_count + 1e-6)
    
    feat["question_marks"] = option_text.count("?")
    feat["comma_count"] = option_text.count(",")
    feat["semicolon_count"] = option_text.count(";")
    feat["colon_count"] = option_text.count(":")
    feat["parenthesis_count"] = option_text.count("(") + option_text.count(")")
    
    unique_words = len(set(words))
    feat["lexical_diversity"] = unique_words / word_count
    feat["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    
    svd_vec = text_data["options_svd"][option_label][row_idx]
    feat["svd_mean"] = np.mean(svd_vec)
    feat["svd_std"] = np.std(svd_vec)
    feat["svd_max"] = np.max(svd_vec)
    feat["svd_min"] = np.min(svd_vec)
    feat["svd_sum"] = np.sum(svd_vec)
    feat["svd_norm"] = np.linalg.norm(svd_vec)
    return feat

def extract_features(df, text_data):
    feature_rows = []
    for row_idx in range(len(df)):
        for option_idx, option_label in enumerate(OPTION_COLS):
            feat = {}
            feat.update(basic_features(row_idx, option_idx, option_label, text_data))
            feat.update(similarity_features(row_idx, option_idx, option_label, text_data))
            feat.update(advanced_features(row_idx, option_idx, option_label, text_data))
            feature_rows.append(feat)
    return pd.DataFrame(feature_rows), None, None


# ==========================================
# 3. LOAD MODELS
# ==========================================
@st.cache_resource 
def load_models():
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
