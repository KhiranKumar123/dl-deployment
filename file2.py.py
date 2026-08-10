import os
import re
import copy
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, log_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & PATHS
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/kaggle/input/competitions/smart-mcq-solver-challenge/"
OUTPUT_DIR = "/kaggle/working/"
OPTION_COLS = ["A", "B", "C", "D", "E"]
LABEL2IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

# Load Data
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

features_dict = joblib.load(os.path.join(OUTPUT_DIR, "processed_features.pkl"))
X_train = features_dict["X_train"]
X_test = features_dict["X_test"]
y = features_dict["y"]
groups = features_dict["groups"]
feature_names = features_dict["feature_names"]

# ==========================================
# 2. EVALUATION METRICS
# ==========================================
def apk(actual, predicted, k=3):
    predicted = predicted[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score

def mapk(actuals, predicteds, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)])

def prediction_to_labels(probabilities):
    top = np.argsort(-probabilities, axis=1)
    return [[IDX2LABEL[i] for i in row[:3]] for row in top]

def evaluate(probabilities, answers):
    preds = prediction_to_labels(probabilities)
    map3 = mapk(answers, preds)
    top1 = np.argmax(probabilities, axis=1)
    acc = accuracy_score([LABEL2IDX[a] for a in answers], top1)
    f1 = f1_score([LABEL2IDX[a] for a in answers], top1, average="macro")
    return acc, f1, map3

# ==========================================
# 3. MODEL 1: LIGHTGBM CLASSIFIER
# ==========================================
print("\n--- Training LightGBM Classifier ---")
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "n_estimators": 5000,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1
}

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(X_test.shape[0])
lgb_models = []

for fold, (train_idx, valid_idx) in enumerate(gkf.split(X_train, y, groups), start=1):
    X_tr, X_va = X_train[train_idx], X_train[valid_idx]
    y_tr, y_va = y[train_idx], y[valid_idx]

    model = LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(200), log_evaluation(0)]
    )

    oof_lgb[valid_idx] = model.predict_proba(X_va)[:, 1]
    test_lgb += model.predict_proba(X_test)[:, 1] / N_FOLDS
    lgb_models.append(model)

oof_lgb_matrix = oof_lgb.reshape(len(train), 5)
lgb_acc, lgb_f1, lgb_map3 = evaluate(oof_lgb_matrix, train["answer"].values)
print(f"LightGBM Validation | Accuracy: {lgb_acc:.5f} | Macro F1: {lgb_f1:.5f} | MAP@3: {lgb_map3:.5f}")

# ==========================================
# 4. MODEL 2: PYTORCH MLP FROM SCRATCH
# ==========================================
print("\n--- Training Scratch PyTorch MLP ---")
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text_simple(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()

train_pair_texts = []
for _, row in train.iterrows():
    p = clean_text_simple(row["prompt"])
    for opt in OPTION_COLS:
        train_pair_texts.append(f"question: {p} option: {clean_text_simple(row[opt])}")

vectorizer_mlp = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True, dtype=np.float32)
X_mlp_all = vectorizer_mlp.fit_transform(train_pair_texts).toarray().astype(np.float32)
X_mlp_reshaped = X_mlp_all.reshape(len(train), 5, -1)
y_mlp_labels = train["answer"].map(LABEL2IDX).values

class ScratchMCQMLP(nn.Module):
    def __init__(self, input_size, hidden_size=512, dropout=0.30):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        b, n, f = x.shape
        scores = self.network(x.reshape(b * n, f))
        return scores.reshape(b, n)

mlp_model = ScratchMCQMLP(input_size=X_mlp_reshaped.shape[-1]).to(DEVICE)
optimizer_mlp = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3)
criterion_mlp = nn.CrossEntropyLoss()

X_tensor = torch.tensor(X_mlp_reshaped)
y_tensor = torch.tensor(y_mlp_labels, dtype=torch.long)

for epoch in range(3):
    mlp_model.train()
    perm = torch.randperm(len(X_tensor))
    for start in range(0, len(X_tensor), 32):
        idx = perm[start:start+32]
        optimizer_mlp.zero_grad()
        loss = criterion_mlp(mlp_model(X_tensor[idx].to(DEVICE)), y_tensor[idx].to(DEVICE))
        loss.backward()
        optimizer_mlp.step()

print("Scratch MLP Training complete.")

# ==========================================
# 5. MODEL 3: DEBERTA-V3 FINE-TUNING
# ==========================================
print("\n--- Training DeBERTa-v3 Multiple Choice Model ---")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MCQDataset(Dataset):
    def __init__(self, df, labels=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompts = [str(row["prompt"])] * 5
        options = [str(row[col]) for col in OPTION_COLS]

        encoded = tokenizer(
            prompts, options,
            truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        item = {k: v for k, v in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

question_labels = train["answer"].map(LABEL2IDX).values
train_dataset = MCQDataset(train, question_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

deberta_model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(DEVICE)
optimizer_deb = torch.optim.AdamW(deberta_model.parameters(), lr=1e-5, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
scheduler_deb = get_linear_schedule_with_warmup(optimizer_deb, num_warmup_steps=int(0.10 * total_steps), num_training_steps=total_steps)

for epoch in range(EPOCHS):
    deberta_model.train()
    for batch in train_loader:
        labels_b = batch.pop("labels").to(DEVICE)
        batch_b = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer_deb.zero_grad()
        loss = deberta_model(**batch_b, labels=labels_b).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deberta_model.parameters(), 1.0)
        optimizer_deb.step()
        scheduler_deb.step()
    print(f"DeBERTa Epoch {epoch+1}/{EPOCHS} Finished.")

# Inference DeBERTa Test
test_dataset = MCQDataset(test, labels=None)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

deberta_model.eval()
deb_probs_list = []
with torch.no_grad():
    for batch in test_loader:
        batch_b = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = deberta_model(**batch_b).logits
        deb_probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())

deberta_test_probs = np.concatenate(deb_probs_list)

# ==========================================
# 6. ENSEMBLE & SAFE LOOKUP POST-PROCESSING
# ==========================================
print("\n--- Ensembling & Creating Final Submission ---")

lgb_test_matrix = test_lgb.reshape(len(test), 5)
lgb_test_probs = lgb_test_matrix / (lgb_test_matrix.sum(axis=1, keepdims=True) + 1e-12)

# Weighted Blending (30% DeBERTa, 70% LightGBM)
ensemble_test_probs = 0.30 * deberta_test_probs + 0.70 * lgb_test_probs

# Option Lookup Override
def option_set_key(row):
    return " || ".join(str(row[col]).strip().lower() for col in OPTION_COLS)

train_lookup = (
    train.assign(option_key=train.apply(option_set_key, axis=1))
         .groupby("option_key")["answer"]
         .first()
         .to_dict()
)

final_predictions = []
lookup_matches = 0

for row_idx, (_, row) in enumerate(test.iterrows()):
    ranked_indices = np.argsort(-ensemble_test_probs[row_idx])
    ranking = [OPTION_COLS[i] for i in ranked_indices]

    key = option_set_key(row)
    if key in train_lookup:
        known_answer = train_lookup[key]
        ranking = [known_answer] + [label for label in ranking if label != known_answer]
        lookup_matches += 1

    final_predictions.append(" ".join(ranking[:3]))

print(f"Exact Option Set Matches Applied: {lookup_matches} / {len(test)}")

# Output Submission
submission = sample_sub[[sample_sub.columns[0]]].copy()
submission[sample_sub.columns[1]] = final_predictions

sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)

print(f"Final submission saved successfully to '{sub_path}'.")
print(submission.head(10))