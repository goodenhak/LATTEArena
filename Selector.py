import pandas as pd
import os
import json
import utils
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from Evaluator import safe_scale_large_values

def calculate_rf_importance(train_data, test_data, model, target, task_type):
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

    if task_type == 1:
        r = permutation_importance(model, X_test, y_test,
                            n_repeats=5,
                            random_state=0)
    elif task_type == 0:
        r = permutation_importance(model, X_test, y_test,
                            n_repeats=5, scoring='neg_root_mean_squared_error',
                            random_state=0)
    
    return r.importances_mean

def select_top_features(df, importance, k: int):
    # Build feature importance DataFrame
    feature_importance = pd.DataFrame({
        "feature": df.columns[:-1],   # Assuming the last column is the target
        "importance": importance
    })
    # Remove features with importance < 0
    feature_importance = feature_importance[feature_importance["importance"] >= 0]
    # Sort by importance
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)
    # Select top k features
    top_features = feature_importance.head(k)["feature"].tolist()
    # Ensure target column is retained
    selected_columns = top_features + ['target']
    # Columns to be dropped (excluding target column)
    dropped_features = [col for col in df.columns if col not in selected_columns]

    return df[selected_columns].copy(), dropped_features

def keep_most_important_features(
    df: pd.DataFrame,
    test_importance: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    # Filter out features with importance below threshold
    filtered_importance = test_importance[test_importance['importance'] >= 0.0]

    # Sort by importance descending and select up to k feature names
    most_important_features = (
        filtered_importance.sort_values(by='importance', ascending=False)
        .head(k).index.tolist()
    )

    # Select these columns from original df (if they exist)
    df_filtered = df[[col for col in most_important_features if col in df.columns]].copy()

    # List of dropped columns
    dropped_columns = [col for col in df.columns if col not in df_filtered.columns]

    return df_filtered, dropped_columns
    

def update_metadata(metadata, dropped_columns):
    # Remove specified keys
    for col in dropped_columns:
        metadata.pop(col, None)  # Use pop to avoid error if key doesn't exist

    return metadata

def update_code(dropped_columns, data_name):
    code_path = os.path.join('tmp', data_name, 'feature_generation.py')
    with open(code_path, 'r', encoding='utf-8') as f:
        code = f.read()

    old_line = utils.extract_function_body(code)

    code_path = os.path.join('tmp', data_name, 'full_code.py')

    # Read the code
    with open(code_path, 'r', encoding='utf-8') as f:
        code = f.read()

    lines = code.split('\n')

    # Calculate indentation (keep consistent style)
    indent = ' ' * 4

    # Insert drop line
    drop_line = [(
        indent + f"df.drop(columns={dropped_columns}, inplace=True)"
    )]

    if dropped_columns == []:
        lines = old_line + lines
    else:
        lines = old_line + lines + drop_line

    # Reassemble the code
    updated_code = '\n'.join(lines)

    # Save back to the original file
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(updated_code)

    return updated_code