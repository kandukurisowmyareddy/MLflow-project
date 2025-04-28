import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(n_estimators=100, max_depth=5):
    
    with mlflow.start_run():
        
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        
        y_pred = model.predict(X_test)
        
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Logged run with Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    
    train_model(n_estimators=100, max_depth=5)
    train_model(n_estimators=200, max_depth=10)
    train_model(n_estimators=150, max_depth=7)
