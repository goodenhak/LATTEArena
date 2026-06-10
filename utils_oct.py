"""
OCTree utilities for LLM4FE framework
Adapted from OCTree/ours/utils_xg.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.tree import _tree, DecisionTreeClassifier, DecisionTreeRegressor
import utils


def preprocess_dataframe(train_data, val_data, test_data, target):
    """
    Convert pandas DataFrame to normalized numpy arrays for OCTree processing.

    Returns:
        xtrain, ytrain, xval, yval, xtest, ytest: numpy arrays
        feature_names: list of feature names
        scalers: dict of scalers for each column
    """
    X_train = train_data.drop(columns=[target]).copy()
    y_train = train_data[target].values
    X_val = val_data.drop(columns=[target]).copy()
    y_val = val_data[target].values
    X_test = test_data.drop(columns=[target]).copy()
    y_test = test_data[target].values

    feature_names = list(X_train.columns)

    # Encode categorical columns and normalize numerical columns
    scalers = {}
    label_encoders = {}

    for col in X_train.columns:
        if X_train[col].dtype == "object" or X_train[col].dtype.name == "category":
            # Label encoding for categorical
            le = LabelEncoder()
            X_train[col] = X_train[col].fillna("missing")
            X_val[col] = X_val[col].fillna("missing")
            X_test[col] = X_test[col].fillna("missing")

            all_values = pd.concat([X_train[col], X_val[col], X_test[col]])
            le.fit(all_values)
            X_train[col] = le.transform(X_train[col])
            X_val[col] = le.transform(X_val[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le
        else:
            # Fill missing with median
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    # Convert to numpy and normalize to [0, 1]
    xtrain = X_train.values.astype(np.float64)
    xval = X_val.values.astype(np.float64)
    xtest = X_test.values.astype(np.float64)

    # MinMax scaling for each column
    for i in range(xtrain.shape[1]):
        scaler = MinMaxScaler()
        xtrain[:, i] = scaler.fit_transform(xtrain[:, i].reshape(-1, 1)).flatten()
        xval[:, i] = scaler.transform(xval[:, i].reshape(-1, 1)).flatten()
        xtest[:, i] = scaler.transform(xtest[:, i].reshape(-1, 1)).flatten()
        scalers[i] = scaler

    return xtrain, y_train, xval, y_val, xtest, y_test, feature_names, scalers


def evaluate_init(xtrain, ytrain, xval, yval, xtest, ytest, task_type=1):
    """
    Evaluate initial model performance.
    task_type: 1 for classification, 0 for regression
    """
    if task_type == 1:
        model = RandomForestClassifier(random_state=42)
        metric = accuracy_score
    else:
        model = RandomForestRegressor(random_state=42)
        metric = lambda true, pred: 1 - root_mean_squared_error(true, pred)

    model.fit(xtrain, ytrain)
    xtrain_pred = model.predict(xtrain)
    xval_pred = model.predict(xval)
    xtest_pred = model.predict(xtest)

    train_score = metric(ytrain, xtrain_pred)
    val_score = metric(yval, xval_pred)
    test_score = metric(ytest, xtest_pred)

    return train_score, val_score, test_score


def add_column(xtrain, xval, xtest, gen_c):
    """Add a new generated column to the datasets with MinMax scaling."""
    gen_c = np.array(gen_c).reshape(-1, 1)
    len_train, len_val = len(xtrain), len(xval)
    gen_c_train = gen_c[:len_train]
    gen_c_val = gen_c[len_train:len_train + len_val]
    gen_c_test = gen_c[len_train + len_val:]

    enc = MinMaxScaler()
    enc.fit(gen_c_train)
    gen_c_train = enc.transform(gen_c_train)
    gen_c_val = enc.transform(gen_c_val)
    gen_c_test = enc.transform(gen_c_test)

    new_train = np.concatenate([xtrain, gen_c_train], axis=-1)
    new_val = np.concatenate([xval, gen_c_val], axis=-1)
    new_test = np.concatenate([xtest, gen_c_test], axis=-1) if xtest is not None else None

    return new_train, new_val, new_test


def evaluate(new_train, ytrain, new_val, yval, new_test, ytest, task_type=1):
    """
    Evaluate feature performance with RandomForest.
    task_type: 1 for classification, 0 for regression
    """
    if task_type == 1:
        model = RandomForestClassifier(random_state=42)
        metric = accuracy_score
    else:
        model = RandomForestRegressor(random_state=42)
        metric = lambda true, pred: 1 - root_mean_squared_error(true, pred)

    model.fit(new_train, ytrain)
    xtrain_pred = model.predict(new_train)
    xval_pred = model.predict(new_val)
    xtest_pred = model.predict(new_test)

    train_score = metric(ytrain, xtrain_pred)
    val_score = metric(yval, xval_pred)
    test_score = metric(ytest, xtest_pred)

    return model, train_score, val_score, test_score


def get_cart(new_train, ytrain, new_val, yval, seed, task_type=1):
    """Train CART decision tree for prompt generation."""
    best_val_score = -np.inf
    best_CART = None

    if task_type == 1:
        metric = accuracy_score
        TreeModel = DecisionTreeClassifier
    else:
        metric = lambda true, pred: 1 - root_mean_squared_error(true, pred)
        TreeModel = DecisionTreeRegressor

    for j in range(1, 4):
        clf_CART = TreeModel(max_depth=j, random_state=seed)
        clf_CART.fit(new_train, ytrain)
        xval_pred = clf_CART.predict(new_val)
        val_score = metric(yval, xval_pred)
        if val_score > best_val_score:
            best_val_score = val_score
            best_CART = clf_CART

    return best_CART


def tree_to_code(tree, feature_names):
    """Convert decision tree to text description for prompt."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    is_classifier = isinstance(tree, DecisionTreeClassifier)

    def recurse(node, depth):
        indent = "  " * depth
        result = ""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            result += f"{indent}if {name} > {threshold:.2f}:\n"
            result += recurse(tree_.children_right[node], depth + 1)
            result += f"{indent}else:\n"
            result += recurse(tree_.children_left[node], depth + 1)
        else:
            if is_classifier:
                class_index = np.argmax(tree_.value[node][0])
                result += f"{indent}y = {float(class_index)}\n"
            else:
                value = tree_.value[node][0][0]
                result += f"{indent}y = {value:.4f}\n"
        return result

    return recurse(0, 0)


def gen_prompt(r_list, dt_list, score_list, idx, task_type=1):
    """Generate prompt for LLM to create new rules."""
    s_l_np = np.array(score_list)
    sorted_idx = np.argsort(s_l_np)[-7:]
    new_r = []
    new_dt = []
    new_s = []
    for i in sorted_idx:
        new_r.append(r_list[i])
        new_dt.append(dt_list[i])
        new_s.append(score_list[i])

    text = f"I have some rules to generate x{idx} from x1"
    for i in range(1, idx-1):
        text += f", x{i+1}"
    text += ". We also have corresponding decision tree (CART) to predict 'y' from x1"
    for i in range(1, idx):
        text += f", x{i+1}"
    text += ". The rules are arranged in ascending order based on their scores evaluated with a RandomForest model, where higher scores indicates better quality."
    text += "\n\n"

    for i in range(len(new_r)):
        text += f"Rule to generate x{idx}:\n{new_r[i]}\n"
        text += f"Decision tree (CART):\n{new_dt[i]}"
        if task_type == 1:
            text += "Score evaluated with RandomForest model:\n{:.0f}".format(new_s[i]*10000)
        else:
            text += "Score (1 - RMSE) evaluated with RandomForest model:\n{:.4f}".format(new_s[i])
        text += "\n\n"

    text += f"Give me a new rule to generate x{idx} that is totally different from the old ones and has a score as high as possible. "
    text += f"Decision trees (including both CART and RandomForest model) trained with newly generated x{idx} should be better than the old ones. "
    text += f"Write the rule to generate x{idx} from x1"
    for i in range(1, idx-1):
        text += f", x{i+1}"
    text += f" . Variables x1 ~ x{idx} are in [0, 1]. You can use various numpy function. Do not use np.log, np.sqrt, np.arcsin, np.arccos, np.arctan. Do not divide. When divide or using log, use (x+1) term. Think creatively. The new rule must be written with Python grammar."
    text += f" Return the rule only with no explanation. For Example: xn = [xi * xj]. The right side of '=' must always be inside square brackets [ ]."

    return text


def rule_template(rule, num_features):
    """Generate executable rule function code."""
    variables = "x1"
    for i in range(1, num_features):
        variables += ", x{:.0f}".format(i+1)
    target_variable = "x{:.0f}".format(num_features+1)
    text = f'''
import numpy as np

def rule(data):
    [{variables}] = data
    {rule}
    return {target_variable}[0]
    '''
    return text


def apply_rule(rule_func, xtrain, xval, xtest):
    """Apply rule function to generate new column."""
    new_col = [rule_func(xtrain[i]) for i in range(len(xtrain))]
    new_col += [rule_func(xval[i]) for i in range(len(xval))]
    new_col += [rule_func(xtest[i]) for i in range(len(xtest))]
    return new_col


def validate_new_col(new_col):
    """Check if new column values are valid (no inf or nan)."""
    for value in new_col:
        if value == np.inf or value == -np.inf or np.isnan(value):
            return False
    return True
