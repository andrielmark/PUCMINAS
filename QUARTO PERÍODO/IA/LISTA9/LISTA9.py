import os
import sys
import math
import warnings
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, completeness_score, homogeneity_score
from sklearn.utils import shuffle

try:
    from minisom import MiniSom
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "minisom"])
    from minisom import MiniSom

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

DATA_PATH = "creditcard.csv"
PDF_REFERENCE = "/mnt/data/Lista 9 - IA[1].pdf"
OUTPUT_DIR = "results_lista9"
RANDOM_STATE = 42


def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    df = pd.read_csv(path)
    return df


def remove_duplicates(df):
    df2 = df.drop_duplicates()
    return df2


def remove_outliers_z_iqr(df, z_thresh=4.0, iqr_factor=1.5):
    numeric = df.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0))
    z_mask = (z_scores > z_thresh)

    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    iqr_mask = ((numeric < (Q1 - iqr_factor * IQR)) | (numeric > (Q3 + iqr_factor * IQR)))

    combined_mask = (z_mask & iqr_mask).any(axis=1)
    df_clean = df.loc[~combined_mask].reset_index(drop=True)
    return df_clean, combined_mask.sum()


def scale_features(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def apply_pca(Xs, n_components=10):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    return Xp, pca


def stratified_split(X, y, test_size=0.3):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    return Xtr, Xte, ytr, yte


def run_kmeans(X, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_safe(X, labels)
    return model, labels, score


def run_dbscan(X, eps=1.5, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    valid_mask = labels != -1
    score = None
    if valid_mask.sum() > 1:
        try:
            score = silhouette_score(X[valid_mask], labels[valid_mask])
        except Exception:
            score = None
    return model, labels, score


def train_som(X_train, x=10, y=10, input_len=None, sigma=1.0, learning_rate=0.5, iters=500):
    if input_len is None:
        input_len = X_train.shape[1]
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate, random_seed=RANDOM_STATE)
    som.random_weights_init(X_train)
    som.train_random(X_train, iters)
    return som


def som_labels(som, X, grid_x=10, grid_y=10):
    win_map = defaultdict(list)
    for i, xx in enumerate(X):
        w = som.winner(xx)
        win_map[w].append(i)
    label_arr = np.full(X.shape[0], -1, dtype=int)
    cluster_id = 0
    for coord, indices in win_map.items():
        for idx in indices:
            label_arr[idx] = cluster_id
        cluster_id += 1
    return label_arr


def silhouette_safe(X, labels):
    try:
        if len(set(labels)) <= 1:
            return None
        return silhouette_score(X, labels)
    except Exception:
        return None


def map_clusters_to_class(labels, y_true):
    mapping = {}
    df = pd.DataFrame({"lbl": labels, "y": y_true})
    groups = df.groupby("lbl")["y"].agg(lambda s: Counter(s).most_common(1)[0][0] if len(s) else -1)
    return groups.to_dict()


def evaluate_clustering(labels, y_true):
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    sil = silhouette_safe(X_test_global, labels)
    mapped = map_clusters_to_class(labels, y_true)
    y_pred_mapped = pd.Series(labels).map(lambda x: mapped.get(x, 0)).values
    cm = confusion_matrix(y_true, y_pred_mapped)
    report = classification_report(y_true, y_pred_mapped, zero_division=0)
    homo = homogeneity_score(y_true, labels)
    compl = completeness_score(y_true, labels)
    return {
        "n_clusters": n_clusters,
        "silhouette": sil,
        "confusion_matrix": cm,
        "report": report,
        "homogeneity": homo,
        "completeness": compl,
        "mapping": mapped,
    }


def save_plot_pca(Xp, y, path):
    plt.figure(figsize=(8, 6))
    if Xp.shape[1] >= 2:
        plt.scatter(Xp[:, 0], Xp[:, 1], c=y, s=8, alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA (PC1 x PC2)")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    plt.close()


def save_cluster_plot(Xp, labels, title, path):
    plt.figure(figsize=(8, 6))
    if Xp.shape[1] >= 2:
        unique = np.unique(labels)
        palette = sns.color_palette("tab10", n_colors=max(2, len(unique)))
        for i, u in enumerate(unique):
            mask = labels == u
            plt.scatter(Xp[mask, 0], Xp[mask, 1], label=str(u), s=8, alpha=0.7)
        plt.legend(markerscale=3, fontsize=8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    plt.close()


def save_som_u_matrix(som, path):
    from matplotlib import cm
    umatrix = som.distance_map().T
    plt.figure(figsize=(6, 6))
    plt.imshow(umatrix, cmap=cm.viridis)
    plt.colorbar()
    plt.title("SOM U-Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def results_to_file(outpath, results):
    with open(outpath, "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}:\n{v}\n\n")


def robust_parameter_search_kmeans(X, k_values=(2, 3, 4), repeats=5):
    best = {"k": None, "score": -1, "labels": None, "model": None}
    for k in k_values:
        scores = []
        for r in range(repeats):
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE + r)
            labels = km.fit_predict(X)
            s = silhouette_safe(X, labels)
            if s is not None:
                scores.append((s, labels, km))
        if not scores:
            continue
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[0]
        if top[0] > best["score"]:
            best = {"k": k, "score": top[0], "labels": top[1], "model": top[2]}
    return best


def robust_parameter_search_dbscan(X, eps_values=(0.8, 1.0, 1.2, 1.5, 2.0), min_samples_values=(5, 8, 10, 15)):
    best = {"eps": None, "min_samples": None, "score": -1, "labels": None, "model": None}
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            valid = labels != -1
            if valid.sum() <= 1:
                continue
            s = silhouette_safe(X[valid], labels[valid])
            if s is not None and s > best["score"]:
                best = {"eps": eps, "min_samples": ms, "score": s, "labels": labels, "model": db}
    return best


def main(args):
    ensure_outdir(OUTPUT_DIR)
    df = load_data(args.data)
    df = remove_duplicates(df)

    df_clean, n_removed = remove_outliers_z_iqr(df, z_thresh=args.z_thresh, iqr_factor=args.iqr_factor)

    X = df_clean.drop("Class", axis=1)
    y = df_clean["Class"].astype(int).values

    Xs, scaler = scale_features(X)
    Xp, pca = apply_pca(Xs, n_components=args.pca_components)

    save_plot_pca(Xp, y, os.path.join(OUTPUT_DIR, "pca_scatter.png"))

    global X_test_global
    X_train, X_test, y_train, y_test = stratified_split(Xp, y, test_size=args.test_size)
    X_test_global = X_test

    kmeans_best = robust_parameter_search_kmeans(X_test, k_values=tuple(range(2, args.kmax + 1)), repeats=5)
    km_model = kmeans_best["model"]
    km_labels = kmeans_best["labels"]
    km_score = kmeans_best["score"]

    save_cluster_plot(X_test, km_labels, f"KMeans k={kmeans_best['k']} silhouette={km_score}", os.path.join(OUTPUT_DIR, "kmeans_clusters.png"))

    db_best = robust_parameter_search_dbscan(X_test, eps_values=args.dbscan_eps_values, min_samples_values=args.dbscan_min_samples_values)
    db_model = db_best.get("model")
    db_labels = db_best.get("labels")
    db_score = db_best.get("score")

    if db_labels is not None:
        save_cluster_plot(X_test, db_labels, f"DBSCAN eps={db_best['eps']} ms={db_best['min_samples']} silhouette={db_score}", os.path.join(OUTPUT_DIR, "dbscan_clusters.png"))

    som = train_som(X_train, x=args.som_x, y=args.som_y, input_len=X_train.shape[1], sigma=args.som_sigma, learning_rate=args.som_lr, iters=args.som_iters)
    som_lbls_train = som_labels(som, X_train, grid_x=args.som_x, grid_y=args.som_y)
    som_lbls_test = som_labels(som, X_test, grid_x=args.som_x, grid_y=args.som_y)

    save_som_u_matrix(som, os.path.join(OUTPUT_DIR, "som_u_matrix.png"))
    save_cluster_plot(X_test, som_lbls_test, "SOM clusters (mapped units)", os.path.join(OUTPUT_DIR, "som_clusters.png"))

    km_eval = evaluate_clustering(km_labels, y_test)
    db_eval = evaluate_clustering(db_labels if db_labels is not None else np.array([-1] * len(y_test)), y_test)
    som_eval = evaluate_clustering(som_lbls_test, y_test)

    results = {
        "n_rows_original": len(df),
        "n_rows_after_outlier_removal": len(df_clean),
        "n_outliers_removed": int(n_removed),
        "kmeans_best_k": int(kmeans_best["k"]) if kmeans_best["k"] is not None else None,
        "kmeans_silhouette": float(km_score) if km_score is not None else None,
        "dbscan_best_eps": float(db_best.get("eps")) if db_best.get("eps") is not None else None,
        "dbscan_best_min_samples": int(db_best.get("min_samples")) if db_best.get("min_samples") is not None else None,
        "dbscan_silhouette": float(db_score) if db_score is not None else None,
        "som_grid": f"{args.som_x}x{args.som_y}",
        "som_iterations": args.som_iters,
        "som_homogeneity": som_eval["homogeneity"],
        "som_completeness": som_eval["completeness"],
    }

    results_to_file(os.path.join(OUTPUT_DIR, "summary_results.txt"), results)
    results_to_file(os.path.join(OUTPUT_DIR, "kmeans_evaluation.txt"), km_eval)
    results_to_file(os.path.join(OUTPUT_DIR, "dbscan_evaluation.txt"), db_eval)
    results_to_file(os.path.join(OUTPUT_DIR, "som_evaluation.txt"), som_eval)

    df_clean.to_csv(os.path.join(OUTPUT_DIR, "data_after_preprocessing.csv"), index=False)

    print("RESULTS")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nKMeans silhouette:", km_score)
    print("DBSCAN silhouette:", db_score)
    print("SOM homogeneity/completeness:", som_eval["homogeneity"], som_eval["completeness"])
    print("\nFiles saved to:", OUTPUT_DIR)
    print("Reference PDF (uploaded):", PDF_REFERENCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--dbscan_eps_values", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--dbscan_min_samples_values", nargs="+", type=int, default=[5, 8, 10, 15])
    parser.add_argument("--som_x", type=int, default=10)
    parser.add_argument("--som_y", type=int, default=10)
    parser.add_argument("--som_sigma", type=float, default=1.0)
    parser.add_argument("--som_lr", type=float, default=0.5)
    parser.add_argument("--som_iters", type=int, default=500)
    parser.add_argument("--z_thresh", type=float, default=4.0)
    parser.add_argument("--iqr_factor", type=float, default=1.5)

    args = parser.parse_args()
    main(args)







