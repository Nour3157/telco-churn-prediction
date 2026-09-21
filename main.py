
import os
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


# Part 1 — Dataset loading & summarization

def load_and_summarize(csv_path_or_df):
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing_per_column": df.isna().sum().to_dict(),
    }


def identify_task(problem_statement: str) -> str:
    text = problem_statement.lower()

    if any(kw in text for kw in ["sequence", "order of", "before a customer",
                                  "sequential"]):
        return "sequential patterns"
    if any(kw in text for kw in ["bought together", "purchased together",
                                  "association", "market basket",
                                  "tend to be purchased"]):
        return "association rules"
    if any(kw in text for kw in ["unusual", "different from", "rare",
                                  "anomaly", "fraud", "outlier", "flag"]):
        return "anomaly detection"
    if any(kw in text for kw in ["group", "segment", "cluster",
                                  "without knowing in advance"]):
        return "clustering"
    if any(kw in text for kw in ["price", "predict a", "how much", "sale price",
                                  "continuous", "estimate the"]) and \
       any(kw in text for kw in ["price", "amount", "value", "sale"]):
        return "regression"
    if any(kw in text for kw in ["predict whether", "classify", "category",
                                  "predict if", "class"]):
        return "classification"
    return "classification"


# Part 2 — Preprocessing

def classify_attribute_types(df: pd.DataFrame, ordinal_orders: dict = None,
                              id_columns: tuple = ()) -> dict:
    ordinal_orders = ordinal_orders or {}
    result = {}
    for col in df.columns:
        if col in ordinal_orders:
            result[col] = "ordinal"
        elif col in id_columns:
            result[col] = "nominal"
        elif not pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "nominal"
        else:
            if (df[col].dropna() >= 0).all():
                result[col] = "ratio"
            else:
                result[col] = "interval"
    return result


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return df


def remove_duplicates(df: pd.DataFrame):
    before = len(df)
    deduped = df.drop_duplicates()
    return deduped, before - len(deduped)


def detect_outliers_iqr(df: pd.DataFrame, column: str):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return df.index[mask].tolist()


def reduce_dimensionality_pca(df: pd.DataFrame, n_components: int):
    numeric = df.select_dtypes(include=[np.number]).fillna(
        df.select_dtypes(include=[np.number]).mean())
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(numeric)
    return transformed, pca.explained_variance_ratio_


# Part 3 — Class imbalance

def class_distribution(y) -> dict:
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    counts_dict = dict(zip(labels.tolist(), counts.tolist()))
    imbalance_ratio = max(counts) / min(counts)
    return {"counts": counts_dict, "imbalance_ratio": imbalance_ratio}


def random_oversample(X, y, seed=0):
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    X_parts, y_parts = [X], [y]
    for label, count in zip(labels, counts):
        if count < majority_count:
            n_needed = majority_count - count
            class_idx = np.where(y == label)[0]
            extra_idx = rng.choice(class_idx, size=n_needed, replace=True)
            X_parts.append(X[extra_idx])
            y_parts.append(y[extra_idx])

    return np.vstack(X_parts), np.concatenate(y_parts)


# Part 4 — Similarity / distance measures

def euclidean_distance(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def k_nearest_neighbors(query, dataset, k, distance_fn):
    query = np.asarray(query, dtype=float)
    dataset = np.asarray(dataset, dtype=float)
    distances = [distance_fn(query, row) for row in dataset]
    order = np.argsort(distances)
    return order[:k].tolist()


# Part 5 — Exploratory Data Analysis

def summary_statistics(series: pd.Series) -> dict:
    return {
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0],
        "range": series.max() - series.min(),
        "variance": series.var(),
        "std": series.std(),
        "p25": series.quantile(0.25),
        "p50": series.quantile(0.50),
        "p75": series.quantile(0.75),
    }


def plot_histogram(series: pd.Series, out_path: str, bins: int = 20, title: str = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series.dropna(), bins=bins, color="steelblue", edgecolor="white")
    ax.set_title(title or f"Histogram of {series.name}")
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, columns: list, out_path: str, title: str = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    df[columns].boxplot(ax=ax)
    ax.set_title(title or "Box plot")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter_matrix(df: pd.DataFrame, columns: list, out_path: str):
    axes = scatter_matrix(df[columns], figsize=(8, 8), diagonal="hist")
    fig = axes[0][0].get_figure()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def eda_report(df: pd.DataFrame, numeric_columns: list, out_dir: str = "eda_output"):
    os.makedirs(out_dir, exist_ok=True)

    for col in numeric_columns:
        stats = summary_statistics(df[col])
        print(f"\n{col}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")

    plot_histogram(df[numeric_columns[0]],
                    os.path.join(out_dir, "histogram.png"))
    plot_boxplot(df, numeric_columns,
                 os.path.join(out_dir, "boxplot.png"))
    plot_scatter_matrix(df, numeric_columns,
                         os.path.join(out_dir, "scatter_matrix.png"))


# Part 6 — Impurity measures & best-split search

def gini_index(class_counts) -> float:
    total = sum(class_counts)
    if total == 0:
        return 0.0
    ps = [c / total for c in class_counts]
    return 1 - sum(p ** 2 for p in ps)


def entropy(class_counts) -> float:
    total = sum(class_counts)
    if total == 0:
        return 0.0
    ps = [c / total for c in class_counts if c > 0]
    return -sum(p * math.log2(p) for p in ps)


def classification_error(class_counts) -> float:
    total = sum(class_counts)
    if total == 0:
        return 0.0
    ps = [c / total for c in class_counts]
    return 1 - max(ps)


def weighted_impurity(splits, impurity_fn) -> float:
    total = sum(sum(split) for split in splits)
    if total == 0:
        return 0.0
    weighted = 0.0
    for split in splits:
        n_i = sum(split)
        weighted += (n_i / total) * impurity_fn(split)
    return weighted


def information_gain(parent_counts, splits, impurity_fn=entropy) -> float:
    return impurity_fn(parent_counts) - weighted_impurity(splits, impurity_fn)


def _class_counts_for_column_value(df, target_col, column, value):
    subset = df[df[column] == value]
    return subset[target_col].value_counts().tolist()


def best_split(df: pd.DataFrame, target_col: str, candidate_columns: list,
               impurity_fn=gini_index):
    parent_counts = df[target_col].value_counts().tolist()

    best_col, best_gain = None, -float("inf")
    for col in candidate_columns:
        splits = []
        for value in df[col].unique():
            counts = _class_counts_for_column_value(df, target_col, col, value)
            splits.append(counts)
        gain = information_gain(parent_counts, splits, impurity_fn)
        if gain > best_gain:
            best_col, best_gain = col, gain

    return best_col, best_gain


# Parts 7 & 8 — From-scratch decision tree (nominal multi-way splits +
# continuous threshold splits)

class TreeNode:
    def __init__(self, is_leaf, prediction=None, attribute=None,
                 children=None, majority_class=None, threshold=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.attribute = attribute
        self.children = children or {}
        self.majority_class = majority_class
        self.threshold = threshold

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.prediction})"
        if self.threshold is not None:
            return f"Node({self.attribute} <= {self.threshold:.2f}?)"
        return f"Node({self.attribute} -> {list(self.children.keys())})"


def _majority_class(targets):
    return targets.value_counts().idxmax()


def best_threshold_split(df: pd.DataFrame, target_col: str,
                          continuous_col: str, impurity_fn=entropy):
    sub = df[[continuous_col, target_col]].sort_values(continuous_col)
    values = sub[continuous_col].to_numpy()
    labels = sub[target_col].to_numpy()

    classes = sorted(set(labels))
    total_counts = {c: int((labels == c).sum()) for c in classes}
    parent_counts = list(total_counts.values())
    left_counts = {c: 0 for c in classes}

    best_threshold, best_gain = None, -float("inf")
    for i in range(len(values) - 1):
        left_counts[labels[i]] += 1
        if values[i] == values[i + 1]:
            continue

        t = (values[i] + values[i + 1]) / 2.0
        left_split = [left_counts[c] for c in classes]
        right_split = [total_counts[c] - left_counts[c] for c in classes]
        gain = information_gain(parent_counts, [left_split, right_split], impurity_fn)
        if gain > best_gain:
            best_threshold, best_gain = t, gain

    return best_threshold, best_gain


def build_tree(df: pd.DataFrame, target_col: str, feature_cols: list,
               continuous_cols=(), impurity_fn=gini_index,
               max_depth=None, depth=0) -> TreeNode:
    targets = df[target_col]

    if targets.nunique() == 1:
        return TreeNode(is_leaf=True, prediction=targets.iloc[0])

    if not feature_cols or (max_depth is not None and depth >= max_depth):
        return TreeNode(is_leaf=True, prediction=_majority_class(targets))

    parent_counts = targets.value_counts().tolist()
    best_col, best_gain, best_threshold = None, -float("inf"), None
    for col in feature_cols:
        if col in continuous_cols:
            threshold, gain = best_threshold_split(df, target_col, col,
                                                     impurity_fn=impurity_fn)
            if threshold is None:
                continue
        else:
            splits = []
            for value in df[col].unique():
                counts = _class_counts_for_column_value(df, target_col, col, value)
                splits.append(counts)
            gain = information_gain(parent_counts, splits, impurity_fn)
            threshold = None

        if gain > best_gain:
            best_col, best_gain, best_threshold = col, gain, threshold

    if best_col is None or best_gain <= 0:
        return TreeNode(is_leaf=True, prediction=_majority_class(targets))

    majority = _majority_class(targets)

    if best_col in continuous_cols:
        left_df = df[df[best_col] <= best_threshold]
        right_df = df[df[best_col] > best_threshold]
        if len(left_df) == 0 or len(right_df) == 0:
            return TreeNode(is_leaf=True, prediction=majority)

        children = {
            "le": build_tree(left_df, target_col, feature_cols, continuous_cols,
                              impurity_fn, max_depth, depth + 1),
            "gt": build_tree(right_df, target_col, feature_cols, continuous_cols,
                              impurity_fn, max_depth, depth + 1),
        }
        return TreeNode(is_leaf=False, attribute=best_col, children=children,
                         majority_class=majority, threshold=best_threshold)

    remaining_features = [c for c in feature_cols if c != best_col]
    children = {}
    for value in df[best_col].unique():
        subset = df[df[best_col] == value]
        children[value] = build_tree(subset, target_col, remaining_features,
                                      continuous_cols, impurity_fn,
                                      max_depth, depth + 1)

    return TreeNode(is_leaf=False, attribute=best_col, children=children,
                     majority_class=majority)


def predict_one(node: TreeNode, row) -> str:
    if node.is_leaf:
        return node.prediction

    value = row[node.attribute]

    if node.threshold is not None:
        branch = "le" if value <= node.threshold else "gt"
        child = node.children.get(branch)
    else:
        child = node.children.get(value)

    if child is None:
        return node.majority_class
    return predict_one(child, row)


def predict_all(node: TreeNode, df: pd.DataFrame):
    return [predict_one(node, row) for _, row in df.iterrows()]


# Part 9 — Evaluation toolkit

def confusion_matrix(y_true, y_pred, labels):
    y_true = list(y_true)
    y_pred = list(y_pred)
    label_index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[label_index[t]][label_index[p]] += 1
    return matrix


def accuracy(y_true, y_pred) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def precision_recall_f1(y_true, y_pred, positive_label) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)

    tp = sum(t == positive_label and p == positive_label
             for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_label and p == positive_label
             for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_label and p != positive_label
             for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


def cost_sensitive_score(y_true, y_pred, cost_matrix: dict, labels) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    total_cost = 0.0
    for t, p in zip(y_true, y_pred):
        total_cost += cost_matrix.get((t, p), 0.0)
    return total_cost


def k_fold_cross_validate(X, y, model_fn, k=5, seed=0):
    X = np.asarray(X, dtype=object)
    y = np.asarray(y, dtype=object)
    n = len(X)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]

    scores = []
    start = 0
    for fold_size in fold_sizes:
        test_idx = indices[start:start + fold_size]
        train_idx = np.concatenate([indices[:start], indices[start + fold_size:]])
        start += fold_size

        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(accuracy(y[test_idx], preds))

    return scores, float(np.mean(scores))


def plot_learning_curve(X, y, model_fn, out_path,
                         train_sizes=(0.1, 0.25, 0.5, 0.75, 1.0), seed=0):
    X = np.asarray(X, dtype=object)
    y = np.asarray(y, dtype=object)
    n = len(X)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_point = int(n * 0.8)
    train_idx, val_idx = indices[:split_point], indices[split_point:]

    train_accs, val_accs, sizes = [], [], []
    for frac in train_sizes:
        n_sub = max(2, int(len(train_idx) * frac))
        sub_idx = train_idx[:n_sub]

        model = model_fn()
        model.fit(X[sub_idx], y[sub_idx])

        train_preds = model.predict(X[sub_idx])
        val_preds = model.predict(X[val_idx])

        train_accs.append(accuracy(y[sub_idx], train_preds))
        val_accs.append(accuracy(y[val_idx], val_preds))
        sizes.append(n_sub)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, train_accs, marker="o", label="Training accuracy")
    ax.plot(sizes, val_accs, marker="o", label="Validation accuracy")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return sizes, train_accs, val_accs


def make_knn_model():
    return KNeighborsClassifier(n_neighbors=5, metric="euclidean")


def tree_cross_validate(feature_df, y, target_col, feature_cols, continuous_cols,
                         impurity_fn, max_depth, k=5, seed=0):
    y = np.asarray(y, dtype=object)
    n = len(feature_df)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]

    scores = []
    start = 0
    for fold_size in fold_sizes:
        test_idx = indices[start:start + fold_size]
        train_idx = np.concatenate([indices[:start], indices[start + fold_size:]])
        start += fold_size

        fold_train_df = feature_df.iloc[train_idx].copy()
        fold_train_df[target_col] = y[train_idx]
        fold_test_df = feature_df.iloc[test_idx]

        fold_root = build_tree(fold_train_df, target_col, feature_cols,
                                continuous_cols=continuous_cols,
                                impurity_fn=impurity_fn, max_depth=max_depth)
        fold_preds = predict_all(fold_root, fold_test_df)
        scores.append(accuracy(y[test_idx], fold_preds))

    return scores, float(np.mean(scores))


def tree_learning_curve(feature_df, y, target_col, feature_cols, continuous_cols,
                         impurity_fn, max_depth, out_path,
                         train_sizes=(0.1, 0.25, 0.5, 0.75, 1.0), seed=0):
    y = np.asarray(y, dtype=object)
    n = len(feature_df)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_point = int(n * 0.8)
    train_idx, val_idx = indices[:split_point], indices[split_point:]
    val_df = feature_df.iloc[val_idx]

    train_accs, val_accs, sizes = [], [], []
    for frac in train_sizes:
        n_sub = max(2, int(len(train_idx) * frac))
        sub_idx = train_idx[:n_sub]

        sub_train_df = feature_df.iloc[sub_idx].copy()
        sub_train_df[target_col] = y[sub_idx]
        sub_test_df = feature_df.iloc[sub_idx].reset_index(drop=True)

        root = build_tree(sub_train_df, target_col, feature_cols,
                           continuous_cols=continuous_cols,
                           impurity_fn=impurity_fn, max_depth=max_depth)
        train_preds = predict_all(root, sub_test_df)
        val_preds = predict_all(root, val_df)

        train_accs.append(accuracy(y[sub_idx], train_preds))
        val_accs.append(accuracy(y[val_idx], val_preds))
        sizes.append(n_sub)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, train_accs, marker="o", label="Training accuracy")
    ax.plot(sizes, val_accs, marker="o", label="Validation accuracy")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Configuration

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "telco_churn.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
MAX_DEPTH = 6

CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
ORDINAL_ORDERS = {"Contract": ["Month-to-month", "One year", "Two year"]}
ID_COLS = ("customerID",)
TARGET_COL = "Churn"
POSITIVE_LABEL = "Yes"


# Pipeline

if __name__ == "__main__":

    # Part 1: load & summarize
    raw_df = pd.read_csv(DATA_PATH)
    summary = load_and_summarize(raw_df)
    print(f"Rows: {summary['n_rows']}, Columns: {summary['n_cols']}")
    print("Columns:", summary["columns"])

    problem_statement = ("Predict whether a telecom customer will churn based "
                          "on their account, billing, and service usage info.")
    print(f"Task type: {identify_task(problem_statement)}")

    # Part 2: preprocessing
    df = raw_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing_total_charges = df["TotalCharges"].isna().sum()
    print(f"Missing TotalCharges values: {n_missing_total_charges}")

    attr_types = classify_attribute_types(df, ordinal_orders=ORDINAL_ORDERS,
                                           id_columns=ID_COLS)
    print("\nAttribute types:")
    for col, t in attr_types.items():
        print(f"  {col}: {t}")

    deduped, n_dupes = remove_duplicates(df)
    print(f"\nDuplicate rows removed: {n_dupes}")
    df = deduped

    # outliers and PCA are checked before imputation, so continuous
    # columns still have their missing values dropped just for this check
    for col in CONTINUOUS_COLS:
        idx = detect_outliers_iqr(df.dropna(subset=[col]), col)
        print(f"Outliers in {col}: {len(idx)}")

    pca_input = df[CONTINUOUS_COLS].dropna()
    _, var_ratio = reduce_dimensionality_pca(pca_input, n_components=2)
    print(f"\nPCA explained variance ratio: {var_ratio}")

    # Part 3: class imbalance
    dist = class_distribution(df[TARGET_COL].values)
    print(f"\nClass distribution: {dist}")
    needs_resampling = dist["imbalance_ratio"] > (70 / 30)
    print(f"Needs resampling: {needs_resampling}")

    categorical_cols = []
    for c, t in attr_types.items():
        if t in ("nominal", "ordinal") and c not in ID_COLS and c != TARGET_COL:
            categorical_cols.append(c)

    feature_df = df.drop(columns=list(ID_COLS) + [TARGET_COL])
    y = df[TARGET_COL].values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    X_test_df = X_test_df.copy()

    # fill missing values using training data only, then apply the same
    # value to the test set (no leakage)
    train_mean_total_charges = X_train_df["TotalCharges"].mean()
    X_train_df = handle_missing_values(X_train_df, strategy="mean")
    X_test_df["TotalCharges"] = X_test_df["TotalCharges"].fillna(train_mean_total_charges)
    print(f"\nFilled missing TotalCharges with training mean: {train_mean_total_charges:.2f}")

    # Part 5: EDA (training data only)
    eda_dir = os.path.join(OUT_DIR, "eda")
    eda_report(X_train_df, CONTINUOUS_COLS, out_dir=eda_dir)

    # Part 3 resample training data only
    if needs_resampling:
        row_ids = np.arange(len(X_train_df)).reshape(-1, 1)
        resampled_ids, y_train_bal = random_oversample(row_ids, y_train,
                                                         seed=RANDOM_STATE)
        resampled_ids = resampled_ids.reshape(-1).astype(int)
        X_train_bal_df = X_train_df.iloc[resampled_ids]
    else:
        X_train_bal_df = X_train_df.copy()
        y_train_bal = y_train

    print(f"\nTraining rows before/after oversampling: "
          f"{len(X_train_df)} / {len(X_train_bal_df)}")

    # Feature table for the from-scratch tree
    tree_feature_cols = categorical_cols + CONTINUOUS_COLS

    X_train_tree = X_train_bal_df[categorical_cols].astype(str).copy()
    for col in CONTINUOUS_COLS:
        X_train_tree[col] = X_train_bal_df[col].astype(float)
    X_train_tree = X_train_tree[tree_feature_cols]

    X_test_tree = X_test_df[categorical_cols].astype(str).copy()
    for col in CONTINUOUS_COLS:
        X_test_tree[col] = X_test_df[col].astype(float)
    X_test_tree = X_test_tree[tree_feature_cols]

    # Feature table for k-NN: one-hot categoricals + scaled numerics
    knn_preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), CONTINUOUS_COLS),
    ])
    knn_preprocessor.fit(X_train_df)

    X_train_knn = knn_preprocessor.transform(X_train_bal_df)
    X_test_knn = knn_preprocessor.transform(X_test_df)
    if hasattr(X_train_knn, "toarray"):
        X_train_knn = X_train_knn.toarray()
        X_test_knn = X_test_knn.toarray()

    # Part 6: sanity-check impurity functions against Play Tennis
    PLAY_TENNIS = pd.DataFrame([
        ["Sunny", "Hot", "High", "Weak", "No"], ["Sunny", "Hot", "High", "Strong", "No"],
        ["Overcast", "Hot", "High", "Weak", "Yes"], ["Rain", "Mild", "High", "Weak", "Yes"],
        ["Rain", "Cool", "Normal", "Weak", "Yes"], ["Rain", "Cool", "Normal", "Strong", "No"],
        ["Overcast", "Cool", "Normal", "Strong", "Yes"], ["Sunny", "Mild", "High", "Weak", "No"],
        ["Sunny", "Cool", "Normal", "Weak", "Yes"], ["Rain", "Mild", "Normal", "Weak", "Yes"],
        ["Sunny", "Mild", "Normal", "Strong", "Yes"], ["Overcast", "Mild", "High", "Strong", "Yes"],
        ["Overcast", "Hot", "Normal", "Weak", "Yes"], ["Rain", "Mild", "High", "Strong", "No"],
    ], columns=["Outlook", "Temperature", "Humidity", "Wind", "Decision"])
    parent_counts = PLAY_TENNIS["Decision"].value_counts().tolist()
    print(f"\nGini(parent) = {gini_index(parent_counts):.4f} (expected ~0.459)")
    print(f"Entropy(parent) = {entropy(parent_counts):.4f} (expected ~0.940)")

    # Parts 7 & 8: train the from-scratch trees
    train_tree_df = X_train_tree.copy()
    train_tree_df[TARGET_COL] = y_train_bal

    gini_tree_root = build_tree(train_tree_df, TARGET_COL, tree_feature_cols,
                                 continuous_cols=CONTINUOUS_COLS,
                                 impurity_fn=gini_index, max_depth=MAX_DEPTH)
    gini_preds = predict_all(gini_tree_root, X_test_tree)

    entropy_tree_root = build_tree(train_tree_df, TARGET_COL, tree_feature_cols,
                                    continuous_cols=CONTINUOUS_COLS,
                                    impurity_fn=entropy, max_depth=MAX_DEPTH)
    entropy_preds = predict_all(entropy_tree_root, X_test_tree)

    print(f"\nGini tree root: {gini_tree_root}")
    print(f"Entropy tree root: {entropy_tree_root}")

    demo_df = X_train_bal_df[["tenure"]].copy()
    demo_df[TARGET_COL] = y_train_bal
    best_t, best_gain = best_threshold_split(demo_df, TARGET_COL, "tenure")
    print(f"\nBest tenure threshold: {best_t:.1f} months, info gain = {best_gain:.4f}")

    # Additional model: k-NN
    knn = make_knn_model()
    knn.fit(X_train_knn, y_train_bal)
    knn_preds = knn.predict(X_test_knn)

    # Part 9: evaluation
    LABELS = ["Yes", "No"]
    COST_MATRIX = {("Yes", "Yes"): 0, ("No", "No"): 0,
                    ("Yes", "No"): 5, ("No", "Yes"): 1}

    results = {}
    for name, preds in [("Gini tree", gini_preds), ("Entropy tree", entropy_preds),
                         ("k-NN", knn_preds)]:
        cm = confusion_matrix(y_test, preds, LABELS)
        acc = accuracy(y_test, preds)
        prf = precision_recall_f1(y_test, preds, POSITIVE_LABEL)
        cost = cost_sensitive_score(y_test, preds, COST_MATRIX, LABELS)
        results[name] = {"confusion_matrix": cm, "accuracy": acc, **prf, "cost": cost}

        print(f"\n{name}")
        print(f"  Confusion matrix (labels={LABELS}): {cm}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prf['precision']:.4f}")
        print(f"  Recall: {prf['recall']:.4f}")
        print(f"  F1: {prf['f1']:.4f}")
        print(f"  Cost score: {cost:.1f}")

    print()

    knn_scores, knn_mean = k_fold_cross_validate(
        X_train_knn, y_train_bal, make_knn_model, k=5, seed=RANDOM_STATE)
    print(f"k-NN 5-fold accuracy: {[round(s, 4) for s in knn_scores]}, mean = {knn_mean:.4f}")

    gini_scores, gini_mean = tree_cross_validate(
        X_train_tree, y_train_bal, TARGET_COL, tree_feature_cols, CONTINUOUS_COLS,
        gini_index, MAX_DEPTH, k=5, seed=RANDOM_STATE)
    print(f"Gini tree 5-fold accuracy: {[round(s, 4) for s in gini_scores]}, mean = {gini_mean:.4f}")

    entropy_scores, entropy_mean = tree_cross_validate(
        X_train_tree, y_train_bal, TARGET_COL, tree_feature_cols, CONTINUOUS_COLS,
        entropy, MAX_DEPTH, k=5, seed=RANDOM_STATE)
    print(f"Entropy tree 5-fold accuracy: {[round(s, 4) for s in entropy_scores]}, mean = {entropy_mean:.4f}")

    plot_learning_curve(X_train_knn, y_train_bal, make_knn_model,
                         out_path=os.path.join(OUT_DIR, "learning_curve_knn.png"),
                         seed=RANDOM_STATE)

    tree_learning_curve(X_train_tree, y_train_bal, TARGET_COL, tree_feature_cols,
                         CONTINUOUS_COLS, entropy, MAX_DEPTH,
                         out_path=os.path.join(OUT_DIR, "learning_curve_entropy_tree.png"),
                         seed=RANDOM_STATE)

    print(f"\nSaved plots to {OUT_DIR}")

    best_by_f1 = max(results, key=lambda k: results[k]["f1"])
    best_by_acc = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest accuracy: {best_by_acc} ({results[best_by_acc]['accuracy']:.4f})")
    print(f"Best F1: {best_by_f1} ({results[best_by_f1]['f1']:.4f})")
