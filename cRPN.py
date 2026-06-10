import zss  # pip install zss

# ========= Tree Node =========
class Node(zss.Node):
    def __init__(self, label, children=None):
        super().__init__(label)
        if children:
            for c in children:
                self.addkid(c)

# ========= Postfix Expression -> Tree =========
def postfix_to_tree(expr):
    tokens = expr.split()
    if len(tokens) == 1 and tokens[0].startswith("f"):  # Atomic feature
        return Node(tokens[0])

    stack = []
    for t in tokens:
        if t.startswith("f"):
            stack.append(Node(t))
        else:  # Binary or unary operator
            if len(stack) >= 2:
                right = stack.pop()
                left = stack.pop()
                stack.append(Node(t, [left, right]))
            elif len(stack) == 1:  # Unary operation
                child = stack.pop()
                stack.append(Node(t, [child]))
            else:
                raise ValueError(f"Invalid postfix expression: {expr}")
    if len(stack) != 1:
        raise ValueError(f"Invalid postfix expression: {expr}")
    return stack[0]

# ========= Tree Edit Distance =========
def tree_edit_distance(t1, t2):
    return zss.simple_distance(
        t1, t2,
        get_children=lambda n: n.children,
        get_label=lambda n: n.label,
        insert_cost=lambda n: 1,
        remove_cost=lambda n: 1,
        update_cost=lambda a, b: 0 if a == b else 1
    )

def tree_size(t):
    return 1 + sum(tree_size(c) for c in t.children)

# ========= Similarity Between Two Expression Lists =========
def expr_list_similarity(exprs1, exprs2):
    sims = []
    for expr1 in exprs1:
        t1 = postfix_to_tree(expr1)
        best_sim = 0
        for expr2 in exprs2:
            t2 = postfix_to_tree(expr2)
            dist = tree_edit_distance(t1, t2)
            max_size = max(tree_size(t1), tree_size(t2))
            sim = 1 - dist / max_size
            best_sim = max(best_sim, sim)
        sims.append(best_sim)
    return sum(sims) / len(sims) if sims else 1.0

# # ========= Example =========
# if __name__ == "__main__":
#     exprs1 = ["f1", "f2", "f3", "f4", "f3 square"]
#     exprs2 = ["f1", "f2", "f3", "f4", "f3 square", "f3 reciprocal"]

#     sim = expr_list_similarity(exprs1, exprs2)
#     print("Similarity:", sim)



def init_feature_map(metadata):
    """
    Generate feature mapping from initial metadata.
    metadata: dict, e.g. {"Airline":"Airline","Flight":"Flight","Time":"Time","Length":"Length"}
    return: dict {feature_name: postfix_expression}
    """
    feature_map = {}
    for idx, col in enumerate(metadata.keys(), start=1):
        feature_map[col] = f"f{idx}"  # Initial features directly mapped to f1, f2, ...
    return feature_map


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def update_feature_map(feature_map, operations):
    """
    Update feature mapping based on operation list (supports chained references, multi-round updates).
    feature_map: dict, current feature mapping, {feature_name: postfix_expression}
    operations: list[dict], definition of each new feature
    return: dict, updated feature mapping
    """
    for op in operations:

        new_name = op["new_feature_name"]
        operator = op["operator"]
        f1 = op["feature1"]
        f2 = op.get("feature2")

        # Ensure input features exist
        if f1 not in feature_map:
            raise ValueError(f"Feature {f1} not found")
        if f2 and f2 not in feature_map and is_number(f2)==False:
            raise ValueError(f"Feature {f2} not found")

        # Fully expand
        expr1 = feature_map[f1]
        if f2:
            if is_number(f2)==False:
                expr2 = feature_map[f2]
            else:
                expr2 = op.get("feature2")
        else:
            expr2 = None

        if expr2:
            postfix = f"{expr1} {expr2} {operator}"
        else:
            postfix = f"{expr1} {operator}"

        feature_map[new_name] = postfix

    return feature_map

def metadata_to_expr_list(metadata, feature_map):
    """
    Generate postfix expression list based on metadata and feature mapping.

    Parameters
    ----------
    metadata : dict
        Its keys are column/feature names (order determines output order), values are irrelevant.
    feature_map : dict[str, str]
        {feature_name: postfix_expression or 'fN'}, must contain all keys in metadata.

    Returns
    -------
    list[str]
        Postfix expressions (or single fN) corresponding to metadata keys.

    Raises
    ------
    KeyError
        When a key in metadata is not found in feature_map.
    """
    expr_list = []
    for key in metadata.keys():  # Iterate by metadata key insertion order
        try:
            expr = feature_map[key]
        except KeyError:
            raise KeyError(f"metadata key '{key}' not found in feature_map")
        expr_list.append(expr)
    return expr_list


def expr_list_to_feature_names(expr_list, metadata):
    """
    Translate f_n in postfix expression list back to initial feature names.

    Parameters
    ----------
    expr_list : list[str]
        Postfix expression list containing feature references like f1, f2, etc.
    metadata : dict
        Initial metadata dictionary, whose keys are feature names,
        in order corresponding to f1, f2, f3.

    Returns
    -------
    list[str]
        Expression list with f_n replaced by corresponding feature names.

    Raises
    ------
    ValueError
        When postfix expression contains f_n that exceeds metadata range.
    """
    # Create mapping from f_n to feature name
    f_to_feature = {}
    for idx, feature_name in enumerate(metadata.keys(), start=1):
        f_to_feature[f"f{idx}"] = feature_name

    result = []
    for expr in expr_list:
        # Split expression by space
        tokens = expr.split()
        translated_tokens = []

        for token in tokens:
            if token.startswith("f") and token[1:].isdigit():
                # This is a f_n format feature reference
                if token in f_to_feature:
                    translated_tokens.append(f_to_feature[token])
                else:
                    raise ValueError(f"Feature reference '{token}' not found in metadata")
            else:
                # This is an operator or other content, keep unchanged
                translated_tokens.append(token)

        # Reassemble into expression
        translated_expr = " ".join(translated_tokens)
        result.append(translated_expr)

    return result


## ========= Example =========
# if __name__ == "__main__":
#     # Example metadata
#     metadata = {"Airline":"Airline","Flight":"Flight","Time":"Time","Length":"Length"}
#
#     # Initialize feature mapping
#     feature_map = init_feature_map(metadata)
#     print("Initial feature mapping:", feature_map)
#
#     # Convert metadata to postfix expression list
#     expr_list = metadata_to_expr_list(metadata, feature_map)
#     print("Postfix expression list:", expr_list)
#
#     # Translate postfix expression back to feature names
#     feature_names = expr_list_to_feature_names(expr_list, metadata)
#     print("Translated to feature names:", feature_names)
#
#     # Test complex expressions
#     complex_exprs = ["f1 f2 add", "f3 square", "f1 f2 multiply f3 divide"]
#     translated_complex = expr_list_to_feature_names(complex_exprs, metadata)
#     print("Complex expression translation:", translated_complex)
