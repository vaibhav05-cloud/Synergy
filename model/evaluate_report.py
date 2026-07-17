"""
evaluate_report.py
-------------------
Everything related to SEEING how good the model is: every chart and every
score, all saved as image files, plus one combined HTML report that puts
them all on a single page you can open in your browser.

train_model.py imports this file and calls generate_full_report(...) once,
right after training finishes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed — just saves PNG files
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
import shap


PLOT_DIR_NAME = "plots"


# ---------------------------------------------------------------------------
# Individual charts — each function makes ONE chart and saves it as a PNG
# ---------------------------------------------------------------------------

def plot_class_distribution(y_train_raw, y_val_raw, y_test_raw, classes, out_dir):
    """Bar chart: how many Low/Medium/High examples in each split."""
    fig, ax = plt.subplots(figsize=(6, 4))
    splits = {"Train": y_train_raw, "Validation": y_val_raw, "Test": y_test_raw}
    x = np.arange(len(classes))
    width = 0.25
    for i, (split_name, y) in enumerate(splits.items()):
        counts = [sum(np.asarray(y) == c) for c in classes]
        ax.bar(x + i * width, counts, width, label=split_name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Number of deployments")
    ax.set_title("Class distribution across train / validation / test")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "01_class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(rf_f1, xgb_f1, out_dir):
    """Bar chart comparing the two candidate models on validation macro-F1."""
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["Random Forest", "XGBoost"]
    scores = [rf_f1, xgb_f1]
    colors = ["#7fa8c9", "#0F6E56"]
    bars = ax.bar(names, scores, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Model comparison (which one did we pick, and why)")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}",
                 ha="center", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "02_model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true, y_pred, classes, out_dir):
    """Heatmap: actual risk level vs what the model predicted."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / "03_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_class_metrics(y_true, y_pred, classes, out_dir):
    """Bar chart: precision / recall / F1 for each risk level separately."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_title("Precision / Recall / F1 per risk level (test set)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "04_per_class_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(y_true, y_proba, classes, out_dir):
    """
    ROC curve for each class (one-vs-rest) with AUC score.
    AUC close to 1.0 = model separates that class well from the rest.
    """
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — how well each risk class is separated")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "05_roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, out_dir, top_n=15):
    """Bar chart: which raw features the model relies on most overall."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#0F6E56")
    ax.set_title(f"Top {top_n} features driving the model")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = out_dir / "06_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _shap_values_as_list(shap_values):
    """Normalises SHAP's output (different shap versions return different shapes) into a plain list of one 2D array per class."""
    if isinstance(shap_values, list):
        return shap_values
    if shap_values.ndim == 3:
        return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    return [shap_values]


def plot_shap_overview(shap_values, X_transformed, feature_names, classes, out_dir):
    """One grouped bar chart: average impact of each feature, split by class."""
    shap_list = _shap_values_as_list(shap_values)
    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_list, X_transformed, feature_names=feature_names,
        plot_type="bar", class_names=list(classes), show=False, max_display=12,
    )
    plt.title("SHAP feature importance — overall, split by risk class")
    plt.tight_layout()
    path = out_dir / "07_shap_overview.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_shap_per_class(shap_values, X_transformed, feature_names, classes, out_dir):
    """One detailed beeswarm plot PER class — shows direction, not just magnitude."""
    shap_list = _shap_values_as_list(shap_values)
    paths = []
    for i, cls in enumerate(classes):
        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_list[i], X_transformed, feature_names=feature_names,
            show=False, max_display=10,
        )
        plt.title(f"What pushes a prediction toward '{cls}' risk")
        plt.tight_layout()
        path = out_dir / f"08_shap_{cls.lower()}.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# The one function train_model.py actually calls
# ---------------------------------------------------------------------------

def generate_full_report(
    *, output_dir: Path, classes,
    y_train_raw, y_val_raw, y_test_raw,
    rf_val_f1, xgb_val_f1, best_name,
    y_test, test_pred, test_proba,
    trained_classifier, feature_names,
    shap_values, X_test_transformed,
    test_accuracy, test_macro_f1,
):
    plot_dir = output_dir / PLOT_DIR_NAME
    plot_dir.mkdir(exist_ok=True)

    print("\nGenerating charts...")
    p1 = plot_class_distribution(y_train_raw, y_val_raw, y_test_raw, classes, plot_dir)
    p2 = plot_model_comparison(rf_val_f1, xgb_val_f1, plot_dir)
    p3 = plot_confusion_matrix(y_test, test_pred, classes, plot_dir)
    p4 = plot_per_class_metrics(y_test, test_pred, classes, plot_dir)
    p5 = plot_roc_curves(y_test, test_proba, classes, plot_dir)
    p6 = plot_feature_importance(trained_classifier, feature_names, plot_dir)
    p7 = plot_shap_overview(shap_values, X_test_transformed, feature_names, classes, plot_dir)
    p8_list = plot_shap_per_class(shap_values, X_test_transformed, feature_names, classes, plot_dir)
    print(f"Saved {7 + len(p8_list)} charts to {plot_dir}/")

    # Per-class score table for the HTML report
    precision, recall, f1, support = precision_recall_fscore_support(y_test, test_pred, labels=range(len(classes)))
    rows = "".join(
        f"<tr><td>{c}</td><td>{precision[i]:.2f}</td><td>{recall[i]:.2f}</td>"
        f"<td>{f1[i]:.2f}</td><td>{support[i]}</td></tr>"
        for i, c in enumerate(classes)
    )

    all_images = [p1, p2, p3, p4, p5, p6, p7] + p8_list
    img_tags = "\n".join(
        f'<div class="card"><img src="{PLOT_DIR_NAME}/{img.name}"></div>' for img in all_images
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DeploySense AI — Training Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#fafafa; color:#222; max-width:1100px; margin:30px auto; padding:0 20px; }}
h1 {{ color:#0F6E56; }}
.summary {{ background:#fff; border:1px solid #ddd; border-radius:10px; padding:20px 24px; margin-bottom:24px; }}
.summary b {{ color:#0F6E56; }}
table {{ border-collapse: collapse; width:100%; margin-top:10px; }}
th, td {{ border:1px solid #ddd; padding:6px 12px; text-align:center; }}
th {{ background:#E1F5EE; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:20px; }}
.card {{ background:#fff; border:1px solid #ddd; border-radius:10px; padding:10px; text-align:center; }}
.card img {{ max-width:100%; }}
</style></head>
<body>
<h1>DeploySense AI — Training Report</h1>
<div class="summary">
  <p><b>Best model:</b> {best_name}</p>
  <p><b>Test accuracy:</b> {test_accuracy:.1%} &nbsp;&nbsp; <b>Test macro-F1:</b> {test_macro_f1:.3f}</p>
  <table>
    <tr><th>Risk level</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support (rows)</th></tr>
    {rows}
  </table>
</div>
<div class="grid">
{img_tags}
</div>
</body></html>"""

    report_path = output_dir / "training_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nOpen this in your browser to see EVERYTHING in one place:\n  {report_path}")
    return report_path