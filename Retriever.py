from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast
from zss import Node, simple_distance
import graphviz

def get_semantic_embedding(text):
    """Get semantic embedding vector for text"""
    # Use pre-trained model to get text embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embedding vectors"""
    similarity = cosine_similarity(
        embedding1.reshape(1, -1),
        embedding2.reshape(1, -1)
    )
    return similarity[0][0]

def update_embedding_cache(strings, cache):
    """Update the embedding cache with new strings if not already present"""
    for s in strings:
        if s not in cache:
            cache[s] = get_semantic_embedding(s)
            print("embedding added")
    return cache

def get_top_k_similar_nodes(history_nodes, query_node, k=3, embedding_cache=None):
    if embedding_cache is None:
        embedding_cache = {}
    query_feat = list(query_node['metadata'].keys())
    num_query_feat = len(query_feat)
    query_feat_str = []
    for feat in query_feat:
        query_feat_str.append(feat + ": " + query_node['metadata'][feat])
    # Update cache with query features
    update_embedding_cache(query_feat_str, embedding_cache)
    query_embeddings = [embedding_cache[s] for s in query_feat_str]
    sim_node = []
    for node in  history_nodes:
        node_feat = list(node['metadata'].keys())
        num_node_feat = len(node_feat)
        node_feat_str = []
        for feat in node_feat:
            node_feat_str.append(feat + ": " + node['metadata'][feat])
        # Update cache with node features
        update_embedding_cache(node_feat_str, embedding_cache)
        node_embeddings = [embedding_cache[s] for s in node_feat_str]
        sum_sim = 0
        for q_emb in query_embeddings:
            for n_emb in node_embeddings:
                sum_sim += calculate_cosine_similarity(q_emb, n_emb)
        sim_node.append((node, sum_sim / (num_query_feat * num_node_feat)))
        print("node similarity added")
    # get top k nodes
    sim_node.sort(key=lambda x: x[1], reverse=True)
    return sim_node[:k]


# Extract column names (including df['a'] or df[['a', 'b']])
def extract_column(node):
    if isinstance(node.slice, ast.Constant):
        return [node.slice.value]
    elif isinstance(node.slice, ast.List):
        return [elt.value for elt in node.slice.elts if isinstance(elt, ast.Constant)]
    return ["?"]

# Expression to tree (supports variable substitution, chained methods, numpy, rolling, etc.)
def expr_to_tree(expr, var_map=None, visited_vars=None, depth=0, max_depth=50):
    if var_map is None:
        var_map = {}
    if visited_vars is None:
        visited_vars = set()
    if depth > max_depth:
        return Node("MAX_DEPTH")

    if isinstance(expr, ast.BinOp):
        op = type(expr.op).__name__
        node = Node(op)
        node.addkid(expr_to_tree(expr.left, var_map, visited_vars, depth + 1))
        node.addkid(expr_to_tree(expr.right, var_map, visited_vars, depth + 1))
        return node

    elif isinstance(expr, ast.UnaryOp):
        op = type(expr.op).__name__
        node = Node(op)
        node.addkid(expr_to_tree(expr.operand, var_map, visited_vars, depth + 1))
        return node

    elif isinstance(expr, ast.Name):
        var_name = expr.id
        if var_name in visited_vars:
            return Node(f"VAR({var_name})")
        visited_vars.add(var_name)
        if var_name in var_map:
            return expr_to_tree(var_map[var_name], var_map, visited_vars, depth + 1)
        else:
            return Node(var_name)

    elif isinstance(expr, ast.Call):
        func = expr.func
        if isinstance(func, ast.Attribute):
            node = Node(func.attr)
            node.addkid(expr_to_tree(func.value, var_map, visited_vars, depth + 1))
        elif isinstance(func, ast.Name):
            node = Node(func.id)
        else:
            node = Node("call")

        for arg in expr.args:
            node.addkid(expr_to_tree(arg, var_map, visited_vars, depth + 1))
        return node

    elif isinstance(expr, ast.Attribute):
        return Node(expr.attr)

    elif isinstance(expr, ast.Subscript):
        columns = extract_column(expr)
        if len(columns) == 1:
            return Node(columns[0])
        else:
            node = Node("MultiColumn")
            for col in columns:
                node.addkid(Node(col))
            return node

    elif isinstance(expr, ast.Constant):
        return Node(str(expr.value))

    return Node("?")

# Extract all semantic trees (supports variable assignment and reference)
def extract_all_trees(code):
    tree = ast.parse(code)
    var_map = {}
    trees = []

    for stmt in tree.body[0].body:  # Assume only one function in the code
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            value = stmt.value
            if isinstance(target, ast.Subscript):
                trees.append(expr_to_tree(value, var_map))
            elif isinstance(target, ast.Name):
                var_map[target.id] = value
    return trees

# Calculate total tree edit distance
def total_tree_edit_distance(code1, code2):
    trees1 = extract_all_trees(code1)
    trees2 = extract_all_trees(code2)
    max_len = max(len(trees1), len(trees2))
    trees1 += [Node("empty")] * (max_len - len(trees1))
    trees2 += [Node("empty")] * (max_len - len(trees2))

    total = 0
    for t1, t2 in zip(trees1, trees2):
        total += simple_distance(t1, t2)
    return total

# Convert single zss.Node tree to Graphviz DOT format
def visualize_tree_zss(node, name="Tree"):
    dot = graphviz.Digraph(name)
    node_id = 0
    id_map = {}

    def add_nodes(n):
        nonlocal node_id
        this_id = id(n)
        id_map[this_id] = f"n{node_id}"
        dot.node(f"n{node_id}", n.label)
        current = node_id
        node_id += 1
        for kid in n.children:
            child_id = add_nodes(kid)
            dot.edge(f"n{current}", f"n{child_id}")
        return current

    add_nodes(node)
    return dot

def load_history_nodes(data_name):
    """Load history node data from ./tmp/{data_name}/ directory"""
    history_nodes = []
    data_dir = f"./tmp/{data_name}/"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")

    # Get all files matching _iter_{i} and group by iteration number
    file_groups = {}
    for filename in os.listdir(data_dir):
        match = re.search(r'_iter_(\d+)\.(json|py)$', filename)
        if match:
            iter_num = int(match.group(1))  # Convert to int for sorting
            ext = match.group(2)
            if iter_num not in file_groups:
                file_groups[iter_num] = {}
            file_groups[iter_num][ext] = os.path.join(data_dir, filename)

    # Process paired json and py files in ascending order by iteration number
    for iter_num in sorted(file_groups.keys()):
        files = file_groups[iter_num]
        if 'json' in files and 'py' in files:  # Ensure paired json and py files exist
            try:
                # Load metadata
                with open(files['json'], 'r') as f:
                    metadata = json.load(f)
                # Load code
                with open(files['py'], 'r') as f:
                    code = f.read()
                # Create node
                node_data = {
                    'metadata': metadata,
                    'code': code
                }
                history_nodes.append(node_data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not process iteration {iter_num}: {str(e)}")
        else:
            print(f"Warning: Missing pair for iteration {iter_num}")

    return history_nodes