from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def get_models():
    return {
        "NaiveBayes": GaussianNB(),

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ),

        "KNN": KNeighborsClassifier(n_neighbors=5),

        # LinearSVC — RBF-এর চেয়ে ১০x দ্রুত, ROC curve ও কাজ করবে
        "SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42)
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        ),

        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42
        )
    }