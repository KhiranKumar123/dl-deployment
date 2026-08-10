import os
import re
import random
import hashlib
import warnings
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & SEEDING
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "/kaggle/input/competitions/smart-mcq-solver-challenge/"
OUTPUT_DIR = "/kaggle/working/"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

OPTION_COLS = ["A", "B", "C", "D", "E"]
LABEL2IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

NEGATION_WORDS = {"no", "not", "never", "none", "neither", "nor", "cannot", "without", "n't", "false", "unlike"}
ABSOLUTE_WORDS = {"always", "all", "every", "entirely", "absolutely", "completely", "only", "must", "true"}
UNITS_WORDS = {"kg", "cm", "mm", "m", "km", "hz", "mol", "nm", "v", "w", "ev", "kj", "pa", "atm", "sec", "min", "hr"}
STOPWORDS = set(ENGLISH_STOP_WORDS)

# ==========================================
# 2. HELPER & CLEANING FUNCTIONS
# ==========================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def strip_boilerplate(text):
    text = clean_text(text)
    patterns = [
        r"pick the best possible answer:?",
        r"choose the correct answer:?",
        r"which of the following is correct:?",
        r"which of the following statements is true:?",
        r"select the best option:?",
        r"what is the correct answer:?"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()

def option_hash(row):
    key = "|".join(str(row[c]).strip().lower() for c in OPTION_COLS)
    return hashlib.md5(key.encode()).hexdigest()

def get_longest_common_prefix(strings):
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

# ==========================================
# 3. DATA LOADING & EDA
# ==========================================
print("Loading Datasets...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Train Shape: {train.shape} | Test Shape: {test.shape}")

train["Question_hash"] = train.apply(option_hash, axis=1)
test["Question_hash"] = test.apply(option_hash, axis=1)

print(f"Unique Question Hashes in Train: {train['Question_hash'].nunique()} / {len(train)}")
print(f"Unique Question Hashes in Test:  {test['Question_hash'].nunique()} / {len(test)}")

# Lookup Map for Option Re-use
train["option_set"] = train[OPTION_COLS].apply(lambda x: " | ".join([str(i).strip() for i in x]), axis=1)
test["option_set"] = test[OPTION_COLS].apply(lambda x: " | ".join([str(i).strip() for i in x]), axis=1)

lookup_map = {row["option_set"]: row["answer"] for _, row in train.iterrows()}
print(f"Exact Option Lookup Map Size: {len(lookup_map)}")

# ==========================================
# 4. TF-IDF & SVD VECTORIZER FITTING
# ==========================================
print("Fitting TF-IDF Vectorizers and SVD...")

all_prompts = pd.concat([train["prompt"], test["prompt"]]).apply(clean_text)
all_options = []
for col in OPTION_COLS:
    all_options.extend(train[col].apply(clean_text))
    all_options.extend(test[col].apply(clean_text))

corpus = list(all_prompts) + all_options

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
tfidf.fit(corpus)

def build_joint_text(df):
    pairs = []
    for _, row in df.iterrows():
        prompt = strip_boilerplate(clean_text(str(row["prompt"])))
        for col in OPTION_COLS:
            option = clean_text(str(row[col]))
            pairs.append(f"{prompt} {option}")
    return pairs

train_joint = build_joint_text(train)
test_joint = build_joint_text(test)

joint_tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
joint_tfidf.fit(train_joint)

svd = TruncatedSVD(n_components=16, random_state=SEED)
svd.fit(tfidf.transform(corpus))

# Save Vectorizers
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.pkl"))
joblib.dump(joint_tfidf, os.path.join(MODEL_DIR, "joint_tfidf.pkl"))
joblib.dump(svd, os.path.join(MODEL_DIR, "svd.pkl"))
print("Vectorizers saved successfully to /models/.")

# ==========================================
# 5. FEATURE EXTRACTION PIPELINE
# ==========================================
def prepare_text_data(df):
    data = {
        "prompt_clean": df["prompt"].fillna("").apply(clean_text),
        "prompt_core": df["prompt"].fillna("").apply(strip_boilerplate),
        "options_clean": {},
        "options_tfidf": {},
        "options_svd": {}
    }
    data["prompt_tfidf"] = tfidf.transform(data["prompt_core"])
    data["prompt_svd"] = svd.transform(data["prompt_tfidf"])

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

    mean_char, std_char = np.mean(char_lengths), np.std(char_lengths) + 1e-6
    mean_word, std_word = np.mean(word_lengths), np.std(word_lengths) + 1e-6
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
    prompt_text = text_data["prompt_core"].iloc[row_idx]
    prompt_words = set(prompt_text.lower().split())
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
    feature_rows, labels, groups = [], [], []
    for row_idx in tqdm(range(len(df)), desc="Extracting Features"):
        for option_idx, option_label in enumerate(OPTION_COLS):
            feat = {}
            feat.update(basic_features(row_idx, option_idx, option_label, text_data))
            feat.update(similarity_features(row_idx, option_idx, option_label, text_data))
            feat.update(advanced_features(row_idx, option_idx, option_label, text_data))
            feature_rows.append(feat)

            if "answer" in df.columns:
                labels.append(int(option_label == df.iloc[row_idx]["answer"]))
        groups.append(row_idx)

    return pd.DataFrame(feature_rows), labels, groups

# Build Matrix
print("Building Train/Test Feature Matrices...")
train_text = prepare_text_data(train)
X_train_df, y, _ = extract_features(train, train_text)

test_text = prepare_text_data(test)
X_test_df, _, _ = extract_features(test, test_text)

X_joint_train = joint_tfidf.transform(train_joint)
X_joint_test = joint_tfidf.transform(test_joint)

X_train = hstack([csr_matrix(X_train_df.values), X_joint_train]).tocsr()
X_test = hstack([csr_matrix(X_test_df.values), X_joint_test]).tocsr()

y = np.array(y, dtype=np.int8)
question_groups = pd.factorize(train["Question_hash"])[0]
groups = np.repeat(question_groups, 5)

dense_feature_names = X_train_df.columns.tolist()
feature_names = dense_feature_names + [f"tfidf_{i}" for i in range(X_joint_train.shape[1])]

# Save Processed Arrays
joblib.dump({
    "X_train": X_train,
    "X_test": X_test,
    "y": y,
    "groups": groups,
    "feature_names": feature_names,
    "lookup_map": lookup_map
}, os.path.join(OUTPUT_DIR, "processed_features.pkl"))

print(f"Features saved successfully! X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")