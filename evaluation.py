from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def evaluate(self, y_true, y_pred, regularized=False):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- Precision: {precision:.4f}")
        print(f"- Recall: {recall:.4f}")
        print(f"- F1 Score: {f1:.4f}")
