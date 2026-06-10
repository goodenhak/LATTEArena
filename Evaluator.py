from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

def safe_scale_large_values(train_data: pd.DataFrame, test_data: pd.DataFrame, threshold=1e30):
    """
    Check if train_data/test_data contains excessively large values (exceeding threshold) or inf,
    replace inf with threshold, then scale the column in both train and test.

    Args:
        train_data, test_data : pandas.DataFrame
        threshold : float, columns with absolute values exceeding this will be scaled, inf replaced with this value
    Returns:
        train_scaled, test_scaled : Processed DataFrames
    """
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()

    # Replace inf with threshold
    train_scaled = train_scaled.replace([float('inf'), -float('inf')], threshold)
    test_scaled = test_scaled.replace([float('inf'), -float('inf')], threshold)

    # Check maximum absolute value for each column
    max_abs = pd.concat([train_scaled.abs().max(), test_scaled.abs().max()], axis=1).max(axis=1)

    # Find columns exceeding threshold
    bad_cols = max_abs[max_abs > threshold].index.tolist()
    if bad_cols:
        print(f"Warning: Detected {len(bad_cols)} column(s) with excessively large values: {bad_cols}")

        for col in bad_cols:
            # Scale using column's maximum absolute value, keeping train/test consistent
            max_val = max(train_scaled[col].abs().max(), test_scaled[col].abs().max())
            if max_val == 0:
                continue
            scale_factor = max_val / threshold
            train_scaled[col] = train_scaled[col] / scale_factor
            test_scaled[col] = test_scaled[col] / scale_factor
            print(f"Scaled column '{col}', scale factor = {scale_factor:.3e}")
    else:
        print("No excessively large values or inf detected, no scaling needed.")

    return train_scaled, test_scaled

def train_classifier(X_train, y_train, model_name='DecisionTree', model_params=None):
    """
    Trains a classifier on the provided training data based on the specified model name.

    Parameters:
    X_train (DataFrame): Feature matrix for training.
    y_train (Series): Labels for training.
    model_name (str): Name of the model to train ('DecisionTree' or 'NeuralNetwork').

    Returns:
    A trained model (either DecisionTreeClassifier or a PyTorch model).
    """
    if model_name.lower() == 'decision_tree':
        # Initialize and train the Decision Tree Classifier
        clf = DecisionTreeClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name.lower() == 'random_forest':
        # Initialize and train the Random Forest Classifier
        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name.lower() == 'knn':
        # Initialize and train the K-Nearest Neighbors Classifier
        clf = KNeighborsClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name.lower() == 'mlp':
        # Initialize and train the Multi-Layer Perceptron Classifier
        clf = MLPClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name == 'NeuralNetwork':
        # Convert DataFrame to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Define a simple neural network model
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, len(y_train.unique()))
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return model
    else:
        raise ValueError("Unsupported model name provided.")

def test_classifier(clf, X_test, y_test):
    """
    Tests a trained classifier on the provided test data and calculates accuracy, precision, recall, and F1 score.

    Parameters:
    clf (classifier): A trained classifier which can be DecisionTreeClassifier, RandomForest, KNN, MLP, or a PyTorch model.
    X_test (DataFrame): Feature matrix for testing.
    y_test (Series): Actual labels for testing.

    Returns:
    tuple: A tuple containing the predictions and the scores (accuracy, precision, recall, F1).
    """
    # Check if the classifier is a PyTorch model
    if isinstance(clf, torch.nn.Module):
        # Convert DataFrame to PyTorch tensors
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

        # Ensure model is in evaluation mode
        clf.eval()

        # Predict on the test data
        with torch.no_grad():
            outputs = clf(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.numpy()  # Convert predictions to numpy array for compatibility with sklearn metrics

    else:
        # Predict using a scikit-learn model
        y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # Macro averaging for multi-class classification
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Return all calculated metrics
    return y_pred, (accuracy, precision, recall, f1)


def load_dataset(data_name):
    train_data = TabularDataset(f'tmp/{data_name}/train.csv')
    val_data = TabularDataset(f'tmp/{data_name}/val.csv')
    test_data = TabularDataset(f'tmp/{data_name}/test.csv')
    return train_data, val_data, test_data

def best_dataset(data_name):
    train_data = TabularDataset(f"./tmp/{data_name}/best_train.csv")
    test_data = TabularDataset(f"./tmp/{data_name}/best_test.csv")
    return train_data, test_data

def executor(code, data_name, train_data, val_data):
    # Create namespace for exec and execute the code
    namespace = {}
    exec(code, globals(), namespace)
    
    # Get the feature_generation function from the namespace
    feature_generation_func = namespace.get('feature_generation')
        
    new_train = feature_generation_func(train_data)
    new_val = feature_generation_func(val_data)
    
    new_train.to_csv(f"./tmp/{data_name}/new_train.csv", index=False)
    new_val.to_csv(f"./tmp/{data_name}/new_val.csv", index=False)
    return new_train, new_val

def train_and_evaluate(train_data, test_data, target, task_type):
    excluded_model_types = ['NN_TORCH']
    if task_type == 1:
        predictor = TabularPredictor(label=target)
    elif task_type == 0:
        predictor = TabularPredictor(label=target,problem_type='regression')
    predictor.fit(train_data, presets='medium_quality', num_bag_folds=3, num_bag_sets=1, num_stack_levels=1, excluded_model_types=excluded_model_types, time_limit=1500)
    predictor.delete_models(models_to_keep='best',dry_run = False)
    if task_type == 1:
        test_acc = predictor.evaluate(test_data)['accuracy']
    elif task_type == 0:
        test_acc = predictor.evaluate(test_data)['root_mean_squared_error']+1
    return predictor, test_acc

def train_and_evaluate_rf(train_data, test_data, target, task_type):
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    # Preprocessing
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            # Fill missing values
            X_train[col] = X_train[col].fillna("missing")
            X_test[col] = X_test[col].fillna("missing")
            # Label Encoding (unified encoding for train and test)
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col], X_test[col]], axis=0))
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

        elif X_train[col].dtype == "category":
            # Fill missing values
            X_train[col] = X_train[col].cat.add_categories(["missing"]).fillna("missing")
            X_test[col] = X_test[col].cat.add_categories(["missing"]).fillna("missing")
            # Label Encoding (unified encoding for train and test)
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

        else:
            # Fill missing values for numerical columns (median is more robust)
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    X_train, X_test = safe_scale_large_values(X_train, X_test)

    # Train model
    if task_type ==1:
        model = RandomForestClassifier(random_state=42)
    elif task_type == 0:
        model = RandomForestRegressor(random_state=42)
    else:
        return None, None
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    if task_type ==1:
        metrics = {
            "score": accuracy_score(y_test, y_pred)
        }
    elif task_type == 0:
        metrics = {
            "score": 1-root_mean_squared_error(y_test, y_pred)
        }

    return model, metrics['score']

def train_and_evaluate_all(train_data, test_data, target):
    predictor = TabularPredictor(label=target)
    predictor.fit(train_data, presets='medium_quality', num_bag_folds=3, num_bag_sets=1, num_stack_levels=1)
    predictor.delete_models(models_to_keep='best',dry_run = False)
    val_metrics = predictor.evaluate(test_data)
    accuracy = val_metrics['accuracy']
    precision = None
    recall = None
    f1 = None

    return predictor, (accuracy,precision,recall,f1)

def train_and_evaluate_all_rf(train_data, test_data, target, task_type):
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    # Preprocessing
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            # Fill missing values
            X_train[col] = X_train[col].fillna("missing")
            X_test[col] = X_test[col].fillna("missing")
            # Label Encoding (unified encoding for train and test)
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col], X_test[col]], axis=0))
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

        elif X_train[col].dtype == "category":
            # Fill missing values
            X_train[col] = X_train[col].cat.add_categories(["missing"]).fillna("missing")
            X_test[col] = X_test[col].cat.add_categories(["missing"]).fillna("missing")
            # Label Encoding (unified encoding for train and test)
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

        else:
            # Fill missing values for numerical columns (median is more robust)
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    X_train, X_test = safe_scale_large_values(X_train, X_test)

    # Train model
    if task_type ==1:
        model = RandomForestClassifier(random_state=42)
    elif task_type == 0:
        model = RandomForestRegressor(random_state=42)
    else:
        return None, None
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    if task_type ==1:
    # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    elif task_type == 0:
        accuracy = 1-root_mean_squared_error(y_test, y_pred)
        precision = 0
        recall = 0
        f1 = 0

    return model, (accuracy,precision,recall,f1)

def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error

def downstream_task(data, task_type, metric_type, state_num=1):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if task_type == 'cls':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
            random_state=state_num, stratify=y)
    elif task_type == 'reg':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
            random_state=state_num)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        if metric_type == 'acc':
            return accuracy_score(y_test, y_predict)
        elif metric_type == 'pre':
            return precision_score(y_test, y_predict)
        elif metric_type == 'rec':
            return recall_score(y_test, y_predict)
        elif metric_type == 'f1':
            return f1_score(y_test, y_predict, average='weighted')
    if task_type == 'reg':
        reg = RandomForestRegressor(random_state=42).fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        if metric_type == 'mae':
            return mean_absolute_error(y_test, y_predict)
        elif metric_type == 'mse':
            return mean_squared_error(y_test, y_predict)
        elif metric_type == 'rae':
            return 1 - relative_absolute_error(y_test, y_predict)
        elif metric_type == 'rmse':
            return 1 - root_mean_squared_error(y_test, y_predict)