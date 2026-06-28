import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# لیبل‌ها (CIFAR-10)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def full_classification_report(model, X, y):
    """
    Comprehensive report:
    - Accuracy
    - Confusion matrix
    - Per-class table
    - Sklearn classification report
    """
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Confusion Matrix:\n")
    display(cm_df)

    # Per-Class Table
    per_class = []
    for i, class_name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        incorrect = total - correct
        percent = correct / total * 100
        per_class.append({
            "Class": class_name,
            "Total": total,
            "Correct": correct,
            "Incorrect": incorrect,
            "Accuracy (%)": round(percent, 2)
        })
    per_class_df = pd.DataFrame(per_class)
    print("\nPer-Class Performance:\n")
    display(per_class_df)

    # Classification Report (precision, recall, f1-score, support)
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    clf_report_df = pd.DataFrame(clf_report).transpose()
    print("\nClassification Report:\n")
    display(clf_report_df)

    return cm_df, per_class_df, clf_report_df


def show_predictions_grid(model, X, y, n=4):
    """
    Show n x n grid of random test images with true & predicted labels
    """
    idxs = np.random.choice(len(X), n * n, replace=False)

    X_sel = X[idxs]
    y_true = np.argmax(y[idxs], axis=1)
    y_pred = np.argmax(model.predict(X_sel, verbose=0), axis=1)

    plt.figure(figsize=(n, n))

    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(X_sel[i])
        plt.axis("off")

        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]

        color = "green" if y_true[i] == y_pred[i] else "red"

        plt.title(
            f"T: {true_label}\nP: {pred_label}",
            color=color,
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


def show_misclassified_grid(model, X, y, n=4):
    """
    Show n x n grid of misclassified images
    """
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    wrong_idxs = np.where(y_true != y_pred)[0]

    if len(wrong_idxs) < n * n:
        print("Not enough misclassified samples.")
        return

    selected = np.random.choice(wrong_idxs, n * n, replace=False)

    plt.figure(figsize=(n, n))

    for i, idx in enumerate(selected):
        plt.subplot(n, n, i + 1)
        plt.imshow(X[idx])
        plt.axis("off")

        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]

        plt.title(
            f"T: {true_label}\nP: {pred_label}",
            color="red",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


def full_classification_report_sklearn(model, X, y):
    """
    Comprehensive report for sklearn models:
    - Accuracy
    - Confusion matrix
    - Per-class table
    - Classification report (precision, recall, f1, support)
    """
    y_pred = model.predict(X)

    # Accuracy
    acc = accuracy_score(y, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Confusion Matrix:\n")
    display(cm_df)

    # Per-class table
    per_class = []
    for i, class_name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i,i]
        incorrect = total - correct
        percent = correct / total * 100
        per_class.append({
            "Class": class_name,
            "Total": total,
            "Correct": correct,
            "Incorrect": incorrect,
            "Accuracy (%)": round(percent,2)
        })
    per_class_df = pd.DataFrame(per_class)
    print("\nPer-Class Performance:\n")
    display(per_class_df)

    # Classification Report
    clf_report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    clf_report_df = pd.DataFrame(clf_report).transpose()
    print("\nClassification Report:\n")
    display(clf_report_df)

    return cm_df, per_class_df, clf_report_df


def evaluate_sklearn_models(models_list, X_test, y_test):
    """
    Evaluate multiple sklearn models on test data
    models_list: list of tuples (model_name:str, model_object)
    X_test: test data, shape (num_samples, 32,32,3) or flattened
    y_test: true labels as integers
    show_grids: if True, show predictions and misclassified images
    n: grid size
    """
    for name, model in models_list:
        print("="*40)
        print(f"Evaluating model: {name}")
        print("="*40)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}\n")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print("Confusion Matrix:\n")
        display(cm_df)

        # Per-Class Performance
        per_class = []
        for i, class_name in enumerate(class_names):
            total = cm[i].sum()
            correct = cm[i, i]
            incorrect = total - correct
            percent = correct / total * 100
            per_class.append({
                "Class": class_name,
                "Total": total,
                "Correct": correct,
                "Incorrect": incorrect,
                "Accuracy (%)": round(percent,2)
            })
        per_class_df = pd.DataFrame(per_class)
        print("\nPer-Class Performance:\n")
        display(per_class_df)

        # Classification Report
        clf_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        clf_report_df = pd.DataFrame(clf_report).transpose()
        print("\nClassification Report:\n")
        display(clf_report_df)

        print("\n\n")
