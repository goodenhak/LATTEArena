import utils
import json
import re
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any
import Evaluator
import logging

logger = logging.getLogger(__name__)

def reconstruct_string(results: List[Dict[str, Any]]) -> str:
    """
    Reconstruct the extracted feature element list into a multi-line string format.
    This is the inverse of the extract_feature_elements function.

    Args:
        results: Element list extracted by extract_feature_elements

    Returns:
        str: Reconstructed multi-line string, each line formatted as <field1><field2>...<fieldn>
    """
    lines = []

    for item in results:
        # Build field list, filtering out None values
        fields = [
            item['new_feature_name'],
            item['operator'],
            item['feature1']
        ]

        # If feature2 exists and is not None, add to field list
        if item['feature2'] is not None:
            fields.append(item['feature2'])

        # Add description
        fields.append(item['description'])

        # Wrap each field in angle brackets and join into a line
        line = ''.join(f'<{field}>' for field in fields)
        lines.append(line)

    # Join all lines with newline
    return '|'.join(lines)

def parse_rpn_expression(rpn_string: str) -> List[List[str]]:
    """
    Parse a string containing multiple RPN expressions, return a list of token lists.
    Input format: [RPN1, RPN2, RPN3,...], where each RPN is space-separated,
    and multiple RPNs are comma-separated.

    Args:
        rpn_string: RPN expression string, e.g. "[credit_amount duration divide, savings_status credit_amount divide, age credit_amount multiply]"

    Returns:
        List[List[str]]: Parsed list of token lists for multiple RPNs
    """
    # Remove line breaks and brackets
    rpn_string = rpn_string.replace('\n', '')
    pattern = r'\{(.*)\}'
    rpn_string = re.findall(pattern, rpn_string)

    # Remove all backslashes
    if rpn_string:
        rpn_string = [s.replace('\\', '') for s in rpn_string]

    # Split multiple RPN expressions by comma
    rpn_expressions = [expr.strip() for expr in rpn_string[0].split(',') if expr.strip()]

    # Split each RPN expression into tokens by space
    all_tokens = []
    for expr in rpn_expressions:
        tokens = [token.strip() for token in expr.split() if token.strip()]
        if tokens:  # Only add non-empty token lists
            all_tokens.append(tokens)

    return all_tokens

def rpn_to_operations(tokens: List[str], rpn_index: int = 1) -> List[Dict[str, Any]]:
    """
    Convert a single reverse Polish notation token list to a feature engineering operation list.
    Supports complex nested operations, e.g.: ["f0", "f0", "f2", "divide", "multiply"]

    Args:
        tokens: Token list of a single RPN expression
        rpn_index: Index of the RPN expression (used for generating feature names)

    Returns:
        List[Dict[str, Any]]: Feature engineering operation list, same format as extract_feature_elements return value
    """
    operations = []
    feature_counter = 1

    # Supported binary operator set
    binary_ops = {'plus', 'subtract', 'multiply', 'divide', 'mod', 'equal', 'greater', 'less', 'max', 'min',
                  'cross', 'concat', 'ratio', 'diff', 'bin', 'groupbythenmean', 'groupbythenmin',
                  'groupbythenmax', 'groupbythenmedian', 'groupbythenstd', 'groupbythenrank'}

    # Supported unary operator set
    unary_ops = {'square', 'sqrt', 'cosine', 'sine', 'tangent', 'exp', 'cube', 'log', 'reciprocal',
                 'sigmoid', 'abs', 'negate', 'zscore', 'minmax', 'rank', 'one_hot', 'label_encode',
                 'extract_time', 'is_weekend', 'elapsed_time','rolling_mean', 'lag', 'cumsum','target_encoding'}

    stack = []

    for token in tokens:
        if token in binary_ops:
            # Binary operator
            if len(stack) < 2:
                raise ValueError(f"RPN {rpn_index}: Binary operator {token} requires at least 2 operands, but stack has {len(stack)}")
            
            operand2 = stack.pop()
            operand1 = stack.pop()
            
            new_feature_name = f"rpn_{rpn_index}_feature_{feature_counter}"
            feature_counter += 1
            
            operation = {
                'new_feature_name': new_feature_name,
                'operator': token,
                'feature1': operand1,
                'feature2': operand2,
                'description': f"RPN{rpn_index}: {operand1} {token} {operand2}"
            }
            operations.append(operation)
            stack.append(new_feature_name)
            
        elif token in unary_ops:
            # Unary operator
            if len(stack) < 1:
                raise ValueError(f"RPN {rpn_index}: Unary operator {token} requires at least 1 operand, but stack has {len(stack)}")
            
            operand = stack.pop()
            new_feature_name = f"rpn_{rpn_index}_feature_{feature_counter}"
            feature_counter += 1
            
            operation = {
                'new_feature_name': new_feature_name,
                'operator': token,
                'feature1': operand,
                'feature2': None,
                'description': f"RPN{rpn_index}: {token}({operand})"
            }
            operations.append(operation)
            stack.append(new_feature_name)
            
        else:
            # Operand (feature name)
            stack.append(token)
    
    return operations

def RPN2exec(rpn_string: str, data_name: str, df: pd.DataFrame):
    """
    Execute multiple reverse Polish notation expressions to generate new features.
    Input format: [RPN1, RPN2, RPN3,...], where each RPN is space-separated,
    and multiple RPNs are comma-separated.

    Args:
        rpn_string: Multiple RPN expression string
        data_name: Dataset name
        df: Input DataFrame

    Returns:
        tuple: (successful operations list, modified DataFrame)
    """
    # Parse multiple reverse Polish notation expressions
    tokens_list = parse_rpn_expression(rpn_string)
    print(f"Number of parsed RPN expressions: {len(tokens_list)}")

    all_success_operations = []
    all_error_list = []
    all_intermediate_features = []
    all_final_features = []
    rpn_map = {}

    # Process each RPN expression one by one
    for rpn_index, tokens in enumerate(tokens_list, 1):
        print(f"RPN {rpn_index} tokens: {tokens}")

        # Build mapping
        rpn_map[rpn_index] = '_'.join(tokens)

        try:
            # Convert to operation list (single RPN)
            operations = rpn_to_operations(tokens, rpn_index)
            print(f"RPN {rpn_index} generated {len(operations)} operations")

            success_operations = []
            error_list = []
            intermediate_features = []
            rpn_failed = False  # Flag if current RPN failed

            # Execute current RPN operations
            for i, operation in enumerate(operations):
                if rpn_failed:
                    # If previous operation failed, skip subsequent operations
                    break

                try:
                    feature_name = operation.get("new_feature_name")
                    op = operation.get("operator")
                    feature1 = operation.get("feature1")
                    feature2 = operation.get("feature2", None)

                    # Check if dependent features exist
                    if feature1 not in df.columns:
                        raise KeyError(f"Dependent feature '{feature1}' does not exist")
                    if feature2 is not None and feature2 not in df.columns:
                        raise KeyError(f"Dependent feature '{feature2}' does not exist")

                    print(f"Executing operation {i+1}/{len(operations)}: {feature_name} = {feature1} {op} {feature2 or ''}")

                    # Execute operation using apply_operation function
                    if feature2 is None:
                        df = apply_operation(df, feature_name, op, feature1)
                    else:
                        df = apply_operation(df, feature_name, op, feature1, feature2)

                    success_operations.append(operation)
                    intermediate_features.append(feature_name)
                    print(f"Operation {i+1} succeeded, new feature: {feature_name}")

                except Exception as e:
                    error_msg = f"Warning: RPN{rpn_index} feature '{feature_name}' processing failed, skipping remaining operations. Error: {e}"
                    print(error_msg)
                    error_list.append(error_msg)
                    rpn_failed = True  # Mark failure, skip subsequent operations

            # Record current RPN's successful operations and intermediate features
            all_success_operations.extend(success_operations)
            all_error_list.extend(error_list)
            all_intermediate_features.extend(intermediate_features)

            # Determine current RPN's final result feature
            if success_operations:
                final_feature = success_operations[-1]['new_feature_name']
                all_final_features.append(final_feature)

        except Exception as e:
            error_msg = f"Warning: RPN{rpn_index} processing failed, skipping entire RPN. Error: {e}"
            print(error_msg)
            all_error_list.append(error_msg)

    # Delete all intermediate variables, keep only final results
    features_to_keep = set(all_final_features)
    features_to_drop = [f for f in all_intermediate_features if f not in features_to_keep]

    if features_to_drop:
        print(f"Dropping intermediate features: {features_to_drop}")
        df = df.drop(columns=features_to_drop)

    if all_error_list:
        logger.info("\n--- RPN Processing Error Summary ---")
        for err in all_error_list:
            logger.info(f"- {err}")

    print(f"RPN processing complete, successfully generated {len(all_final_features)} final features")

    # Use mapping rpn_map to rename retained final features
    for feature in all_final_features:
        rpn_index = int(feature.split('_')[1])
        new_name = rpn_map[rpn_index]
        df = df.rename(columns={feature: new_name})
        print(f"Renaming feature: {feature} -> {new_name}")

    df = utils.drop_duplicate_columns(df)

    return all_success_operations, df


def extract_code_blocks(text: str) -> List[str]:
    """
    Extract Python code blocks from LLM output.
    Supports multiple formats:
    - ```python ... ```end (with backticks - preferred format from template)
    - python ... end (without backticks - fallback)

    Can extract multiple code blocks from a single text.

    Args:
        text: LLM output containing code blocks

    Returns:
        List[str]: List of extracted code blocks
    """
    code_blocks = []

    # Try pattern with backticks first: ```python ... ```end
    pattern1 = r'```python\s*(.*?)\s*```end'
    matches1 = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
    if matches1:
        code_blocks.extend(matches1)

    # If no blocks found, try pattern without backticks: python ... end
    if not code_blocks:
        pattern2 = r'(?:^|\n)python\s*(.*?)\s*end(?:\n|$)'
        matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
        if matches2:
            code_blocks.extend(matches2)

    # Strip whitespace from each code block and filter empty blocks
    code_blocks = [block.strip() for block in code_blocks if block.strip()]

    print(f"Extracted {len(code_blocks)} code block(s)")
    return code_blocks


def Code2exec(text: str, data_name: str, df: pd.DataFrame, target: str = None):
    """
    Execute Python code blocks from LLM output to generate new features.

    Args:
        text: LLM output containing Python code blocks
        data_name: Dataset name
        df: Input DataFrame
        target: Target column name (should stay at position -1)

    Returns:
        tuple: (success operations list, modified DataFrame)
    """
    code_blocks = extract_code_blocks(text)

    success_operations = []
    error_list = []

    for i, code_block in enumerate(code_blocks, 1):
        try:
            print(f"\n--- Executing code block {i}/{len(code_blocks)} ---")
            print(code_block)

            # Get columns before execution
            cols_before = set(df.columns)

            # Save target column if it exists
            target_col = None
            if target and target in df.columns:
                target_col = df[target].copy()

            # Create a safe namespace with necessary imports and df
            namespace = {
                'df': df,
                'pd': pd,
                'np': np
            }

            # Execute the code block
            exec(code_block, namespace)

            # Update df from namespace
            df = namespace['df']

            # Restore target column if it was dropped during execution
            if target_col is not None:
                if target not in df.columns:
                    # Target was dropped during code execution, restore it
                    df[target] = target_col
                # Ensure target column is at the end (position -1)
                # Filter out target and any accidentally created target.N columns (pandas duplicate naming)
                non_target_cols = [col for col in df.columns
                                   if col != target and not (col.startswith(f"{target}.") and col[len(target)+1:].isdigit())]
                # Reorder: all other columns + target
                df = df[non_target_cols + [target]]

            # Get columns after execution to detect new features
            cols_after = set(df.columns)
            new_columns = cols_after - cols_before
            dropped_columns = cols_before - cols_after

            # Extract feature name and description from comments
            # New format: Line 1: # Feature name, Line 2: # Feature description
            feature_name = ""
            description = ""
            comment_lines = []

            for line in code_block.split('\n'):
                line_stripped = line.strip()
                if line_stripped.startswith('#') and not line_stripped.startswith('# Usefulness') and not line_stripped.startswith('# Input samples') and not line_stripped.startswith('# Explanation'):
                    comment_lines.append(line_stripped.strip('# ').strip())

            # Extract feature name and description
            if len(comment_lines) >= 2:
                feature_name = comment_lines[0]  # First comment line is feature name
                description = comment_lines[1]   # Second comment line is description
            elif len(comment_lines) == 1:
                feature_name = comment_lines[0]
                description = comment_lines[0]

            # Determine the operation name
            if new_columns:
                operation_name = ', '.join(sorted(new_columns))
            elif dropped_columns:
                operation_name = f"dropped: {', '.join(sorted(dropped_columns))}"
            else:
                operation_name = "code_modification"

            # Record success operation
            operation = {
                'new_feature_name': operation_name,
                'operator': 'code',
                'feature1': feature_name if feature_name else f"code_block_{i}",
                'feature2': None,
                'description': description if description else f"Code block {i}"
            }
            success_operations.append(operation)
            print(f"✓ Code block {i} executed successfully")
            if feature_name:
                print(f"  Feature name: {feature_name}")
            if description:
                print(f"  Description: {description}")
            if new_columns:
                print(f"  New columns: {', '.join(sorted(new_columns))}")
            if dropped_columns:
                print(f"  Dropped columns: {', '.join(sorted(dropped_columns))}")

        except Exception as e:
            error_msg = f"Warning: Code block {i} execution failed. Error: {e}"
            print(f"✗ {error_msg}")
            error_list.append(error_msg)

    if error_list:
        logger.info("\n--- Code Execution Error Summary ---")
        for err in error_list:
            logger.info(f"- {err}")

    print(f"\n=== Code execution complete: {len(success_operations)}/{len(code_blocks)} blocks succeeded ===")

    df = utils.drop_duplicate_columns(df)

    return success_operations, df


def extract_feature_elements(multiline_string: str) -> List[Dict[str, Any]]:
    """
    Extract elements from a multi-line string where each line follows one of two formats:
    1. <new_feature_name><operator><feature1><feature2><description>
    2. <new_feature_name><operator><feature1><description>

    Returns a list of dictionaries with extracted components for each line:
        {
            'new_feature_name': str,
            'operator': str,
            'feature1': str,
            'feature2': str | None,
            'description': str
        }

    Additional functionality: Remove any number of leading/trailing backslashes from each field.
    """
    pattern = re.compile(r'<(.*?)>')
    results: List[Dict[str, Any]] = []

    # Pre-compile regex to strip leading/trailing backslashes
    strip_slashes = re.compile(r'^\\+|\\+$')

    for line in multiline_string.splitlines():
        line = line.strip()
        if not line:
            continue

        tokens = pattern.findall(line)
        # Remove leading/trailing backslashes
        tokens = [strip_slashes.sub('', t) for t in tokens]

        if len(tokens) == 5:
            fname, op, feat1, feat2, desc = tokens
            results.append({
                'new_feature_name': fname,
                'operator': op,
                'feature1': feat1,
                'feature2': feat2,
                'description': desc
            })
        elif len(tokens) == 4:
            fname, op, feat1, desc = tokens
            results.append({
                'new_feature_name': fname,
                'operator': op,
                'feature1': feat1,
                'feature2': None,
                'description': desc
            })

    return results

def NL2exec(text, data_name, df):
    results = extract_feature_elements(text)
    code_lines = []
    success_operations = [] # Store list of successful operations
    error_list = []

    code_lines.append("def feature_generation(df):")

    for result in results:
        try:
            feature_name = result.get("new_feature_name")
            operation = result.get("operator")
            feature1 = result.get("feature1")
            feature2 = result.get("feature2", None)

            # Execute operation directly
            if feature2 is None:
                df = apply_operation(df, feature_name, operation, feature1)
                code_line = f'    df = Postprocessor.apply_operation(df, "{feature_name}","{operation}", "{feature1}")'
            else:
                df = apply_operation(df, feature_name, operation, feature1, feature2)
                code_line = f'    df = Postprocessor.apply_operation(df, "{feature_name}","{operation}", "{feature1}", "{feature2}")'

            code_lines.append(code_line)
            success_operations.append(result) # On success, add the entire result to success list

        except Exception as e:
            error_msg = f"Warning: Failed to process feature '{feature_name}'. Skipping. Error: {e}"
            print(error_msg)
            error_list.append(error_msg)
            
    code_lines.append("    return df")
    # os.makedirs(f'tmp/{data_name}', exist_ok=True)
    # with open(f'tmp/{data_name}/feature_generation.py', 'w') as f:
    #     f.write("\n".join(code_lines))

    final_code = "\n".join(code_lines)
    if error_list:
        logger.info("\n--- Summary of Errors ---")
        for err in error_list:
            logger.info(f"- {err}")

    return success_operations, df

def exec_metadata(success_operations, data_name):
    """
    Function: Based on successfully executed operations, extract feature names and descriptions,
    and save to JSON file.
    """
    results = []

    for parse_result in success_operations:
        feature_name = parse_result["new_feature_name"]
        description = parse_result["description"]
        operator = parse_result["operator"]
        feature1 = parse_result["feature1"]

        if operator == "one_hot":
            train_data, _, _ = Evaluator.load_dataset(data_name)
            x = train_data[feature1]
            dummies = pd.get_dummies(x, prefix=x.name)
            feature_name_list = dummies.columns.tolist()
            for feature_name_oh in feature_name_list: # Fix variable name conflict
                results.append((feature_name_oh, description))
        elif operator == "code":
            # Handle code operator: might have multiple columns or dropped columns
            if feature_name.startswith("dropped:"):
                # Skip dropped columns, don't add to metadata
                continue
            elif ',' in feature_name:
                # Multiple columns created, add metadata for each
                column_names = [col.strip() for col in feature_name.split(',')]
                for col_name in column_names:
                    results.append((col_name, description))
            else:
                # Single column or modification
                results.append((feature_name, description))
        else:
            results.append((feature_name, description))

    metadata_path = f'tmp/{data_name}/metadata.json'
    original_data = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        original_data = {}

    feature_dict = {feature: desc for feature, desc in results}
    merged_data = {**original_data, **feature_dict}

    return merged_data

# def NL2Code(text, data_name):
#     """
#     output:
#     <start>
#     def feature_generation(df):
#         df = apply_operation(df, feature_name, op, col1, col2)
#         return df
#     <end>
#     """
    
#     results = extract_feature_elements(text)
    
#     # Generate code lines
#     code_lines = []
#     code_lines.append("def feature_generation(df):")
    
#     for result in results:
#         feature_name = result["new_feature_name"]
#         operation = result["operator"]
#         feature1 = result["feature1"]
#         feature2 = result["feature2"]
        
#         # Generate code based on operation type and column count
#         if feature2 == None:
#             code_line = f'    df = Postprocessor.apply_operation(df, "{feature_name}","{operation}", "{feature1}")'
#         else:
#             code_line = f'    df = Postprocessor.apply_operation(df, "{feature_name}","{operation}", "{feature1}", "{feature2}")'
        
#         code_lines.append(code_line)
    
#     code_lines.append("    return df")
    
#     # Assemble final code
#     code = "\n".join(code_lines)
#     # Save to file
#     with open(f'tmp/{data_name}/feature_generation.py', 'w') as f:
#         f.write(code)

#     return f"<start>\n{code}\n<end>"

def extract_metadata(text, data_name):
    """
    Function: Extract feature names and descriptions, and save to JSON file.
    Return format: [(feature_name, description), ...]
    """
    results = []

    parse_results = extract_feature_elements(text)

    for parse_result in parse_results:
        feature_name = parse_result["new_feature_name"]
        description = parse_result["description"]
        operator = parse_result["operator"]
        feature1 = parse_result["feature1"]
        if operator == "one_hot":
            train_data, val_data, test_data = Evaluator.load_dataset(data_name)
            x = train_data[feature1]
            dummies = pd.get_dummies(x, prefix=x.name)
            feature_name_list = dummies.columns.tolist()
            for feature_name in feature_name_list:
                results.append((feature_name, description))
        else:
            results.append((feature_name, description))

    # Read original JSON file
    metadata_path = f'metadata/{data_name}.json'
    original_data = {}

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        original_data = {}

    # Convert results to dictionary format
    feature_dict = {feature: desc for feature, desc in results}

    # Merge original data and new data
    merged_data = {**original_data, **feature_dict}

    # Save to destination file
    output_path = f'tmp/{data_name}/metadata.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Metadata saved to: {output_path}")

    return results

def apply_operation(data, feature_name, op, attr1, attr2=None, **kwargs):
    """Apply feature engineering operation to DataFrame columns.
    
    Args:
        data: Input DataFrame
        op: Operation name (string)
        attr1: First column name
        attr2: Second column name (for binary operations)
        **kwargs: Additional parameters for advanced operations:
            - window: Window size for rolling_mean (default: 3)
            - n: Lag period for lag operation (default: 1)
            - bins: Number of bins for bin operation (default: 5)
            - labels: Bin labels for bin operation (default: None)
            - group_col: Group column for groupby operations
            - target_col: Target column for groupby operations
        
    Returns:
        tuple: (modified DataFrame, error message, new column name)
    """
    # Handle column naming
    if feature_name == "":
        if attr2 is None:
            new_column_name = f"{attr1}_{op}"
        else:
            new_column_name = f"{attr1}_{op}_{attr2}"
    else:
        new_column_name = feature_name

    # Operation mapping - unary operations
    unary_ops = {
        'square': lambda x: x ** 2,
        'sqrt': np.sqrt,
        'cosine': np.cos,
        'sine': np.sin,
        'tangent': np.tan,
        'exp': np.exp,
        'cube': lambda x: x ** 3,
        'log': lambda x: np.log(x + 1e-6),
        'reciprocal': lambda x: 1 / (x + 1e-6),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'abs': lambda x: x.abs(),
        'negate': lambda x: -x,
        'zscore': lambda x: (x - x.mean()) / x.std(),
        'minmax': lambda x: (x - x.min()) / (x.max() - x.min()),
        'rank': lambda x: x.rank(),
        'bin': lambda x: pd.cut(x, bins=kwargs.get('bins', 5), 
                               labels=kwargs.get('labels', None)),
        'one_hot': lambda x: pd.get_dummies(x, prefix=x.name),
        'label_encode': lambda x: x.astype("category").cat.codes,
        'extract_time': lambda x: getattr(x.dt, kwargs.get('attr', 'year')),
        'is_weekend': lambda x: x >= 5.0,
        'elapsed_time': lambda x: (x - kwargs.get('ref_time', pd.Timestamp.now())).dt.total_seconds()
    }

    # Operation mapping - binary operations
    binary_ops = {
        'plus': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / (y + 1e-6),
        'mod': lambda x, y: x % y,
        'equal': lambda x, y: x == y,
        'greater': lambda x, y: x > y,
        'less': lambda x, y: x < y,
        'max': lambda x, y: np.maximum(x, y),
        'min': lambda x, y: np.minimum(x, y),
        'cross': lambda x, y: x.astype(str) + "_" + y.astype(str),
        'concat': lambda x, y: x.astype(str) + y.astype(str),
        'diff': lambda x, y: x - y,
        'ratio': lambda x, y: x / (y + 1e-6),
        'bin': lambda x, bins: pd.cut(x, bins=int(bins), labels=None)
    }

    # Advanced operations
    advanced_ops = {
        'rolling_mean': lambda x: x.rolling(window=kwargs.get('window', 3), 
                          min_periods=1).mean(),
        'lag': lambda x: x.shift(kwargs.get('n', 1)),
        'cumsum': lambda x: x.cumsum(),
        'groupbythenmean': lambda df, g, t: df.groupby(g)[t].transform('mean'),
        'groupbythenmin': lambda df, g, t: df.groupby(g)[t].transform('min'),
        'groupbythenmax': lambda df, g, t: df.groupby(g)[t].transform('max'),
        'groupbythenmedian': lambda df, g, t: df.groupby(g)[t].transform('median'),
        'groupbythenstd': lambda df, g, t: df.groupby(g)[t].transform('std'),
        'groupbythenrank': lambda df, g, t: df.groupby(g)[t].transform('rank'),
        'target_encoding': lambda df, c, t: df[c].map(df.groupby(c)[t].mean())
    }

    # Handle unary operations
    if op in unary_ops and attr2 is None:
        if op == "one_hot":
            new_columns = unary_ops[op](data[attr1])
        else:
            new_column = unary_ops[op](data[attr1])
        
    # Handle binary operations
    elif op in binary_ops and attr2 is not None:
        if op == "bin":
            new_column = binary_ops[op](data[attr1], attr2)
        elif (op == "greater" or op == "less" or op == "diff" or op == "equal") and (attr2 not in data.columns):
            new_column = binary_ops[op](data[attr1], float(attr2))
        elif op == "plus" and (attr2 not in data.columns):
            new_column = data[attr1] + int(attr2)
        elif op == "multiply" and (attr2 not in data.columns):
            new_column = data[attr1] * int(attr2)
        elif op == "devide" and (attr2 not in data.columns):
            new_column = data[attr1] / float(attr2)
        else:
            new_column = binary_ops[op](data[attr1], data[attr2])
        
    # Handle advanced operations
    elif op in advanced_ops:
        if op in ['groupbythenmean', 'groupbythenmin', 'groupbythenmax', 
                    'groupbythenmedian', 'groupbythenstd', 'groupbythenrank']:
            new_column = advanced_ops[op](data, attr1, attr2)
        elif op == 'target_encoding':
            new_column = advanced_ops[op](data, attr1, attr2)
        else:
            new_column = advanced_ops[op](data[attr1])
    else:
        raise ValueError(f"Unknown operation: {op}")

    # Insert new column and handle errors
    if op == "one_hot":
        data = pd.concat([
            data.iloc[:, :-1],
            new_columns,
            data.iloc[:, [-1]]
        ], axis=1)
    else:
        data.insert(len(data.columns) - 1, new_column_name, new_column)

    return data

def apply_operation_without_name(data, op, attr1, attr2=None, **kwargs):
    if attr2 is None:
        new_column_name = f"{attr1}_{op}"
    else:
        new_column_name = f"{attr1}_{op}_{attr2}"

       # Operation mapping - unary operations
    unary_ops = {
        'square': lambda x: x ** 2,
        'sqrt': np.sqrt,
        'cosine': np.cos,
        'sine': np.sin,
        'tangent': np.tan,
        'exp': np.exp,
        'cube': lambda x: x ** 3,
        'log': lambda x: np.log(x + 1e-6),
        'reciprocal': lambda x: 1 / (x + 1e-6),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'abs': lambda x: x.abs(),
        'negate': lambda x: -x,
        'zscore': lambda x: (x - x.mean()) / x.std(),
        'minmax': lambda x: (x - x.min()) / (x.max() - x.min()),
        'rank': lambda x: x.rank(),
        'bin': lambda x: pd.cut(x, bins=kwargs.get('bins', 5), 
                               labels=kwargs.get('labels', None)),
        'one_hot': lambda x: pd.get_dummies(x, prefix=x.name),
        'label_encode': lambda x: x.astype("category").cat.codes,
        'extract_time': lambda x: getattr(x.dt, kwargs.get('attr', 'year')),
        'is_weekend': lambda x: x >= 5.0,
        'elapsed_time': lambda x: (x - kwargs.get('ref_time', pd.Timestamp.now())).dt.total_seconds()
    }

    # Operation mapping - binary operations
    binary_ops = {
        'plus': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / (y + 1e-6),
        'mod': lambda x, y: x % y,
        'equal': lambda x, y: x == y,
        'greater': lambda x, y: x > y,
        'less': lambda x, y: x < y,
        'max': lambda x, y: np.maximum(x, y),
        'min': lambda x, y: np.minimum(x, y),
        'cross': lambda x, y: x.astype(str) + "_" + y.astype(str),
        'concat': lambda x, y: x.astype(str) + y.astype(str),
        'diff': lambda x, y: x - y,
        'ratio': lambda x, y: x / (y + 1e-6),
        'bin': lambda x, bins: pd.cut(x, bins=int(bins), labels=None)
    }

    # Advanced operations
    advanced_ops = {
        'rolling_mean': lambda x: x.rolling(window=kwargs.get('window', 3), 
                          min_periods=1).mean(),
        'lag': lambda x: x.shift(kwargs.get('n', 1)),
        'cumsum': lambda x: x.cumsum(),
        'groupbythenmean': lambda df, g, t: df.groupby(g)[t].transform('mean'),
        'groupbythenmin': lambda df, g, t: df.groupby(g)[t].transform('min'),
        'groupbythenmax': lambda df, g, t: df.groupby(g)[t].transform('max'),
        'groupbythenmedian': lambda df, g, t: df.groupby(g)[t].transform('median'),
        'groupbythenstd': lambda df, g, t: df.groupby(g)[t].transform('std'),
        'groupbythenrank': lambda df, g, t: df.groupby(g)[t].transform('rank'),
        'target_encoding': lambda df, c, t: df[c].map(df.groupby(c)[t].mean())
    }

    # Handle unary operations
    if op in unary_ops and attr2 is None:
        if op == "one_hot":
            new_columns = unary_ops[op](data[attr1])
        else:
            new_column = unary_ops[op](data[attr1])
        
    # Handle binary operations
    elif op in binary_ops and attr2 is not None:
        if op == "bin":
            new_column = binary_ops[op](data[attr1], attr2)
        elif (op == "greater" or op == "less" or op == "diff" or op == "equal") and (attr2 not in data.columns):
            new_column = binary_ops[op](data[attr1], float(attr2))
        elif op == "plus" and (attr2 not in data.columns):
            new_column = data[attr1] + int(attr2)
        elif op == "multiply" and (attr2 not in data.columns):
            new_column = data[attr1] * int(attr2)
        elif op == "devide" and (attr2 not in data.columns):
            new_column = data[attr1] / float(attr2)
        else:
            new_column = binary_ops[op](data[attr1], data[attr2])
        
    # Handle advanced operations
    elif op in advanced_ops:
        if op in ['groupbythenmean', 'groupbythenmin', 'groupbythenmax', 
                    'groupbythenmedian', 'groupbythenstd', 'groupbythenrank']:
            new_column = advanced_ops[op](data, attr1, attr2)
        elif op == 'target_encoding':
            new_column = advanced_ops[op](data, attr1, attr2)
        else:
            new_column = advanced_ops[op](data[attr1])
    else:
        return data, f"Unknown operation: {op}", new_column_name

    # Insert new column and handle errors
    if op == "one_hot":
        data = pd.concat([
            data.iloc[:, :-1],
            new_columns,
            data.iloc[:, [-1]]
        ], axis=1)
    else:
        data.insert(len(data.columns) - 1, new_column_name, new_column)

    return data, None, new_column_name
