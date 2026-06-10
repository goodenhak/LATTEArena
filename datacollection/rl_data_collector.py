import os
import sys



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('./')
import torch
from logger import *
from Operation import add_unary, operation_set, O1, O3, O2, sep_token
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)
info(torch.get_num_threads())
info(torch.__config__.parallel_info())
import warnings
import math
from nni.utils import merge_parameter
import nni

torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from sklearn.feature_selection import SelectKBest
from DQN import DQN1, DQN2, DDQN1, DDQN2, DuelingDQN1, DuelingDDQN2, DuelingDQN2, DuelingDDQN1
from tools import *
from task_mapping import task_dict, task_type, base_path, task_measure, state_rep, support_rl_method
import argparse


def init_param():
    parser = argparse.ArgumentParser(description='PyTorch Experiment')
    parser.add_argument('--file-name', type=str, default='openml_616',
                        help='data name')
    parser.add_argument('--log-level', type=str, default='info', help=
    'log level, check the _utils.logger')
    parser.add_argument('--task', type=str, default='ng', help=
    'ng/cls/reg/det/rank, if provided ng, the model will take the task type in config'
                        )
    parser.add_argument('--episodes', type=int, default=30, help=
    'episodes for training')
    parser.add_argument('--steps', type=int, default=15, help=
    'steps for each episode')
    parser.add_argument('--enlarge_num', type=int, default=2, help=
    'feature space enlarge')
    parser.add_argument('--memory', type=int, default=8, help='memory capacity'
                        )
    parser.add_argument('--eps_start', type=float, default=0.9, help=
    'eps start')
    parser.add_argument('--eps_end', type=float, default=0.5, help='eps end')
    parser.add_argument('--eps_decay', type=int, default=100, help='eps decay')
    parser.add_argument('--index', type=float, default=0.5, help='file index')
    parser.add_argument('--state', type=int, default=0, help='random_state')
    parser.add_argument('--cluster_num', type=int, default=0, help=
    'cluster_num')
    parser.add_argument('--a', type=float, default=0, help='a')
    parser.add_argument('--b', type=float, default=0, help='b')
    parser.add_argument('--c', type=float, default=0, help='c')
    parser.add_argument('--rl-method', type=str, default='dqn', help=
    'used reinforcement methods')
    parser.add_argument('--state-method', type=str, default='gcn',
                        help='reinforcement state representation method')
    parser.add_argument('--default-cuda', type=int, default=-1, help=
    'the using cuda')
    # -c removing the feature clustering step of GRFG
    # -d using euclidean distance as feature distance metric in the M-clustering of GRFG
    # -b -u Third, we developed GRFG−𝑢 and GRFG−𝑏 by using random in the two feature generation scenarios
    parser.add_argument('--ablation-mode', type=str, default='', help=
    'the using cuda')

    args, _ = parser.parse_known_args()
    return args


def model_train(param, nni):
    DEVICE = param['default_cuda']
    STATE_METHOD = param['state_method']
    DIVISION_EPS = 1e-8
    assert STATE_METHOD in state_rep
    use_nni = True
    if nni is None:
        current_trial_name = 'local'
        use_nni = False
    else:
        current_trial_name = nni.get_trial_id()
    D_OPT_PATH = './tmp/' + param['file_name'] + '/'
    info('opt path is {}'.format(D_OPT_PATH))
    always_best = []
    all_results = []

    train_data_path = base_path + param['file_name'] + '_train.csv'
    val_data_path = base_path + param['file_name'] + '_val.csv'
    test_data_path = base_path + param['file_name'] + '_test.csv'

    info('read the training data from {}'.format(train_data_path))
    Dg_train = pd.read_csv(train_data_path)

    info('read the validation data from {}'.format(val_data_path))
    Dg_val = pd.read_csv(val_data_path)

    # Load test set for final evaluation
    if os.path.exists(test_data_path):
        Dg_test = pd.read_csv(test_data_path)
        info('read the test data from {}'.format(test_data_path))
        has_test_set = True
    else:
        Dg_test = None
        has_test_set = False
        info('No separate test set found at {}'.format(test_data_path))

    assert param['rl_method'] in support_rl_method
    if param['task'] == 'ng':
        task_name = task_dict[param['file_name']]
    else:
        assert param['task'] in task_type
        task_name = param['task']
    info('the task is performing ' + task_name + ' on _dataset ' + param[
        'file_name'])
    info('the chosen reinforcement learning method is ' + param['rl_method'])
    if task_name == 'cls':
        measure = "acc"
    elif task_name == 'reg':
        measure = "rmse"
    info('the related measurement is ' + measure)
    old_per = downstream_task(Dg_train, Dg_val, task_name, measure)
    info('done the base test with performance of {:.6f}'.format(old_per))

    # Get feature names from training data
    feature_names = list(Dg_train.columns)
    info('initialize the features...')

    # Process training data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = Dg_train.values[:, :-1]
    y_train = Dg_train.values[:, -1]
    Dg_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
    Dg_train.columns = [str(i + len(operation_set) + 5) for i in range(len(feature_names))]
    feature_names = Dg_train.columns

    # Process validation data with the same column names
    X_val = Dg_val.values[:, :-1]
    y_val = Dg_val.values[:, -1]
    Dg_val = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val)], axis=1)
    Dg_val.columns = feature_names  # Use same column names as training set

    # Preprocess test set with the same column names
    if has_test_set:
        X_test_init = Dg_test.values[:, :-1]
        y_test_init = Dg_test.values[:, -1]
        Dg_test = pd.concat([pd.DataFrame(X_test_init), pd.DataFrame(y_test_init)], axis=1)
        Dg_test.columns = feature_names  # Use same column names as training set
        D_original_test = Dg_test.copy()
        D_OPT_test = Dg_test.copy()
        info('test set preprocessed with {} features'.format(len(feature_names) - 1))
    # O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'stand_scaler',
    #     'minmax_scaler', 'quan_trans', 'sigmoid', 'log', 'reciprocal']
    # O2 = ['+', '-', '*', '/']
    # O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']
    # operation_set = O1 + O2
    one_hot_op = pd.get_dummies(operation_set)
    operation_emb = defaultdict()
    for item in one_hot_op.columns:
        operation_emb[item] = one_hot_op[item].values
    EPISODES = param['episodes']
    STEPS = param['steps']
    STATE_DIM = 64
    ACTION_DIM = 64
    MEMORY_CAPACITY = param['memory']
    OP_DIM = len(operation_set)
    FEATURE_LIMIT = Dg_train.shape[1] * param['enlarge_num']
    N_ACTIONS = len(operation_set)
    dqn_cluster1 = None
    dqn_operation = None
    dqn_cluster2 = None
    info('initialize the model...')
    if STATE_METHOD == 'gcn':
        STATE_DIM = X_train.shape[0]
        ACTION_DIM = X_train.shape[0]
    elif STATE_METHOD == 'ae':
        STATE_DIM = X_train.shape[0]
        ACTION_DIM = X_train.shape[0]
    elif STATE_METHOD == 'mds+ae':
        STATE_DIM = X_train.shape[0] + STATE_DIM
        ACTION_DIM = STATE_DIM
    elif STATE_METHOD == 'mds+ae+gcn':
        STATE_DIM = 2 * X_train.shape[0] + STATE_DIM
        ACTION_DIM = STATE_DIM
    if param['rl_method'] == 'dqn':
        dqn_cluster1 = DQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM,
                            MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS,
                             MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=
        ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'ddqn':
        dqn_cluster1 = DDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=ACTION_DIM,
                             MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DDQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS,
                              MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DDQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM=
        ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'dueling_dqn':
        dqn_cluster1 = DuelingDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=
        ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DuelingDQN2(N_STATES=STATE_DIM, N_ACTIONS=N_ACTIONS,
                                    MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DuelingDQN1(STATE_DIM=STATE_DIM + OP_DIM, ACTION_DIM
        =ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    elif param['rl_method'] == 'dueling_ddqn':
        dqn_cluster1 = DuelingDDQN1(STATE_DIM=STATE_DIM, ACTION_DIM=
        ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_operation = DuelingDDQN2(N_STATES=STATE_DIM, N_ACTIONS=
        N_ACTIONS, MEMORY_CAPACITY=MEMORY_CAPACITY)
        dqn_cluster2 = DuelingDDQN1(STATE_DIM=STATE_DIM + OP_DIM,
                                    ACTION_DIM=ACTION_DIM, MEMORY_CAPACITY=MEMORY_CAPACITY)
    base_per = old_per
    episode = 0
    step = 0
    best_per = old_per
    D_OPT_train = Dg_train
    D_OPT_val = Dg_val
    best_features = []
    D_original_train = Dg_train.copy()
    D_original_val = Dg_val.copy()
    steps_done = 0
    EPS_START = param['eps_start']
    EPS_END = param['eps_end']
    EPS_DECAY = param['eps_decay']
    CLUSTER_NUM = 4
    duplicate_count = 0
    a, b, c = param['a'], param['b'], param['c']
    info('initialize the model hyperparameter configure')
    info(
        'epsilon start with {}, end with {}, the decay is {}, the culster num is {}, the duplicate count is {}, the a, b, and c is set to {}, {}, and {}'
        .format(EPS_START, EPS_END, EPS_DECAY, CLUSTER_NUM, duplicate_count,
                a, b, c))
    info('the training start...')
    training_start_time = time.time()
    info('start training at ' + str(training_start_time))
    best_step = -1
    best_episode = -1
    while episode < EPISODES:
        eps_start_time = time.time()
        step = 0
        Dg_train = D_original_train.copy()
        Dg_val = D_original_val.copy()
        # Reset test set for this episode
        if has_test_set:
            Dg_test = D_original_test.copy()
        # Reset feature_names to match the original datasets
        feature_names = list(Dg_train.columns)
        Dg_train_local = D_original_train.copy()
        Dg_val_local = D_original_val.copy()
        local_best = -999
        best_per_opt = []
        while step < STEPS:
            info(f'current feature is : {list(Dg_train.columns)}')
            step_start_time = time.time()
            steps_done += 1
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 *
                                                                       steps_done / EPS_DECAY)

            # Ensure all datasets have the same columns before starting this step
            train_features = set(Dg_train.columns[:-1])
            val_features = set(Dg_val.columns[:-1])
            if has_test_set:
                test_features = set(Dg_test.columns[:-1])
                common_features = list(train_features & val_features & test_features)
            else:
                common_features = list(train_features & val_features)

            # If there's a mismatch, synchronize to common features
            if len(common_features) != len(train_features):
                info(f'Synchronizing datasets: {len(train_features)} train features, {len(common_features)} common features')
                common_features.append(Dg_train.columns[-1])  # Add target column
                Dg_train = Dg_train[common_features]
                Dg_val = Dg_val[common_features]
                if has_test_set:
                    Dg_test = Dg_test[common_features]
                feature_names = common_features

            # Use training data for clustering
            X = Dg_train.values[:, :-1]
            y = Dg_train.values[:, -1]
            clusters = cluster_features(X, y, cluster_num=3)
            action_emb_c1, state_emb_c1, f_cluster1, f_names1 = (
                select_meta_cluster1(clusters, Dg_train.values[:, :-1],
                                     feature_names, eps_threshold, dqn_cluster1, STATE_METHOD,
                                     DEVICE))
            state_emb_op, op, op_index = select_operation(f_cluster1,
                                                          operation_set, dqn_operation, steps_done, STATE_METHOD,
                                                          DEVICE)
            info('start operating in step {}'.format(step))
            info('current op is ' + str(op))
            if op in O1:
                op_sign = justify_operation_type(op)
                f_new, f_new_name = [], []
                if op == 'sqrt':
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] < 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                    f_generate = np.array(f_new).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op == 'reciprocal':
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] == 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                    f_generate = np.array(f_new).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op == 'log':
                    for i in range(f_cluster1.shape[1]):
                        if np.sum(f_cluster1[:, i] <= 0) == 0:
                            f_new.append(op_sign(f_cluster1[:, i]))
                            f_new_name.append(add_unary(op_map_r[op], f_names1[i]))
                    f_generate = np.array(f_new).T
                    final_name = f_new_name
                    if len(f_generate) == 0:
                        continue
                elif op in O3:
                    f_generate = op_sign.fit_transform(f_cluster1)
                    final_name = [add_unary(op_map_r[op], f_n) for f_n in f_names1]
                else:
                    f_generate = op_sign(f_cluster1)
                    final_name = [add_unary(op_map_r[op], f_n) for f_n in f_names1]
            if op in O2:
                op_emb = operation_emb[op]
                op_func = justify_operation_type(op)
                action_emb_c2, state_emb_c2, f_cluster2, f_names2 = (
                    select_meta_cluster2(clusters, Dg_train.values[:, :-1],
                                         feature_names, f_cluster1, op_emb, eps_threshold,
                                         dqn_cluster2, STATE_METHOD, DEVICE))
                if op == '/' and np.sum(np.abs(f_cluster2) < DIVISION_EPS) > 0:
                    continue
                f_generate, final_name = operate_two_features_new(f_cluster1,
                                                                  f_cluster2, op, op_func, f_names1, f_names2)
            if np.max(np.abs(f_generate)) > 1000:
                scaler = MinMaxScaler()
                f_generate = scaler.fit_transform(f_generate)
            f_generate = pd.DataFrame(f_generate)
            f_generate.columns = final_name

            # Generate the same features for validation set
            try:
                # Get the same features from validation set by column names
                # Check if all required features exist in validation set
                missing_features = [f for f in f_names1 if f not in Dg_val.columns]
                if missing_features:
                    info(f'Warning: Features {missing_features} not found in validation set, skipping validation feature generation')
                    f_generate_val = None
                else:
                    f_cluster1_val = Dg_val[f_names1].values
                    if op in O1:
                        op_sign_val = justify_operation_type(op)
                        f_new_val = []
                        if op == 'sqrt':
                            for i in range(f_cluster1_val.shape[1]):
                                if np.sum(f_cluster1_val[:, i] < 0) == 0:
                                    f_new_val.append(op_sign_val(f_cluster1_val[:, i]))
                            f_generate_val = np.array(f_new_val).T if f_new_val else np.array([])
                        elif op == 'reciprocal':
                            for i in range(f_cluster1_val.shape[1]):
                                if np.sum(f_cluster1_val[:, i] == 0) == 0:
                                    f_new_val.append(op_sign_val(f_cluster1_val[:, i]))
                            f_generate_val = np.array(f_new_val).T if f_new_val else np.array([])
                        elif op == 'log':
                            for i in range(f_cluster1_val.shape[1]):
                                if np.sum(f_cluster1_val[:, i] <= 0) == 0:
                                    f_new_val.append(op_sign_val(f_cluster1_val[:, i]))
                            f_generate_val = np.array(f_new_val).T if f_new_val else np.array([])
                        elif op in O3:
                            f_generate_val = op_sign_val.fit_transform(f_cluster1_val)
                        else:
                            f_generate_val = op_sign_val(f_cluster1_val)
                    elif op in O2:
                        # Check if all required features exist for binary operation
                        missing_features2 = [f for f in f_names2 if f not in Dg_val.columns]
                        if missing_features2:
                            info(f'Warning: Features {missing_features2} not found in validation set for binary op')
                            f_generate_val = None
                        else:
                            f_cluster2_val = Dg_val[f_names2].values
                            op_func_val = justify_operation_type(op)
                            f_generate_val, _ = operate_two_features_new(f_cluster1_val, f_cluster2_val, op, op_func_val, f_names1, f_names2)

                    if f_generate_val is not None and len(f_generate_val) > 0:
                        if np.max(np.abs(f_generate_val)) > 1000:
                            scaler_val = MinMaxScaler()
                            f_generate_val = scaler_val.fit_transform(f_generate_val)
                        f_generate_val = pd.DataFrame(f_generate_val)
                        f_generate_val.columns = final_name
                    else:
                        f_generate_val = None
            except Exception as e:
                info(f'Warning: Failed to generate validation features: {e}')
                f_generate_val = None

            # Generate the same features for test set
            if has_test_set:
                try:
                    # Check if all required features exist in test set
                    missing_features = [f for f in f_names1 if f not in Dg_test.columns]
                    if missing_features:
                        info(f'Warning: Features {missing_features} not found in test set, skipping test feature generation')
                        f_generate_test = None
                    else:
                        # Get the same features from test set by column names
                        f_cluster1_test = Dg_test[f_names1].values
                        if op in O1:
                            op_sign_test = justify_operation_type(op)
                            f_new_test = []
                            if op == 'sqrt':
                                for i in range(f_cluster1_test.shape[1]):
                                    if np.sum(f_cluster1_test[:, i] < 0) == 0:
                                        f_new_test.append(op_sign_test(f_cluster1_test[:, i]))
                                f_generate_test = np.array(f_new_test).T if f_new_test else np.array([])
                            elif op == 'reciprocal':
                                for i in range(f_cluster1_test.shape[1]):
                                    if np.sum(f_cluster1_test[:, i] == 0) == 0:
                                        f_new_test.append(op_sign_test(f_cluster1_test[:, i]))
                                f_generate_test = np.array(f_new_test).T if f_new_test else np.array([])
                            elif op == 'log':
                                for i in range(f_cluster1_test.shape[1]):
                                    if np.sum(f_cluster1_test[:, i] <= 0) == 0:
                                        f_new_test.append(op_sign_test(f_cluster1_test[:, i]))
                                f_generate_test = np.array(f_new_test).T if f_new_test else np.array([])
                            elif op in O3:
                                f_generate_test = op_sign_test.fit_transform(f_cluster1_test)
                            else:
                                f_generate_test = op_sign_test(f_cluster1_test)
                        elif op in O2:
                            # Check if all required features exist for binary operation
                            missing_features2 = [f for f in f_names2 if f not in Dg_test.columns]
                            if missing_features2:
                                info(f'Warning: Features {missing_features2} not found in test set for binary op')
                                f_generate_test = None
                            else:
                                f_cluster2_test = Dg_test[f_names2].values
                                op_func_test = justify_operation_type(op)
                                f_generate_test, _ = operate_two_features_new(f_cluster1_test, f_cluster2_test, op, op_func_test, f_names1, f_names2)

                        if f_generate_test is not None and len(f_generate_test) > 0:
                            if np.max(np.abs(f_generate_test)) > 1000:
                                scaler_test = MinMaxScaler()
                                f_generate_test = scaler_test.fit_transform(f_generate_test)
                            f_generate_test = pd.DataFrame(f_generate_test)
                            f_generate_test.columns = final_name
                        else:
                            f_generate_test = None
                except Exception as e:
                    info(f'Warning: Failed to generate test features: {e}')
                    f_generate_test = None
            else:
                f_generate_test = None

            public_name = np.intersect1d(np.array(Dg_train.columns), final_name)
            if len(public_name) > 0:
                reduns = np.setxor1d(final_name, public_name)
                if len(reduns) > 0:
                    f_generate = f_generate[reduns]

                    # Check if validation features were successfully generated
                    if f_generate_val is not None and all(c in f_generate_val.columns for c in reduns):
                        f_generate_val_filtered = f_generate_val[reduns]
                    else:
                        f_generate_val_filtered = None

                    # Check if test features were successfully generated
                    if has_test_set and f_generate_test is not None and all(c in f_generate_test.columns for c in reduns):
                        f_generate_test_filtered = f_generate_test[reduns]
                    else:
                        f_generate_test_filtered = None

                    # Only add features that can be added to ALL datasets
                    if f_generate_val_filtered is not None and (not has_test_set or f_generate_test_filtered is not None):
                        Dg_train = insert_generated_feature_to_original_feas(Dg_train, f_generate)
                        Dg_val = insert_generated_feature_to_original_feas(Dg_val, f_generate_val_filtered)
                        if has_test_set:
                            Dg_test = insert_generated_feature_to_original_feas(Dg_test, f_generate_test_filtered)
                    else:
                        info('Skipping feature generation as not all datasets could generate features')
                        continue
                else:
                    continue
            else:
                # Check if validation features were successfully generated
                if f_generate_val is None:
                    info('Skipping feature generation as validation features failed')
                    continue

                # Check if test features were successfully generated (if test set exists)
                if has_test_set and f_generate_test is None:
                    info('Skipping feature generation as test features failed')
                    continue

                # Add features to all datasets
                Dg_train = insert_generated_feature_to_original_feas(Dg_train, f_generate)
                Dg_val = insert_generated_feature_to_original_feas(Dg_val, f_generate_val)
                if has_test_set:
                    Dg_test = insert_generated_feature_to_original_feas(Dg_test, f_generate_test)

            if Dg_train.shape[1] > FEATURE_LIMIT:
                selector = SelectKBest(mutual_info_regression, k=FEATURE_LIMIT).fit(Dg_train.iloc[:, :-1], Dg_train.iloc[:, -1])
                cols = selector.get_support()
                X_new = Dg_train.iloc[:, :-1].loc[:, cols]
                Dg_train = pd.concat([X_new, Dg_train.iloc[:, -1]], axis=1)
                # Apply same feature selection to validation and test sets
                selected_feature_names = list(X_new.columns)
                # Only select features that exist in validation set
                available_features_val = [f for f in selected_feature_names if f in Dg_val.columns]
                if len(available_features_val) < len(selected_feature_names):
                    info(f'Warning: Some selected features not in validation set. Expected {len(selected_feature_names)}, found {len(available_features_val)}')
                X_val_new = Dg_val[available_features_val]
                Dg_val = pd.concat([X_val_new, Dg_val.iloc[:, -1]], axis=1)
                if has_test_set:
                    # Only select features that exist in test set
                    available_features_test = [f for f in selected_feature_names if f in Dg_test.columns]
                    if len(available_features_test) < len(selected_feature_names):
                        info(f'Warning: Some selected features not in test set. Expected {len(selected_feature_names)}, found {len(available_features_test)}')
                    X_test_new = Dg_test[available_features_test]
                    Dg_test = pd.concat([X_test_new, Dg_test.iloc[:, -1]], axis=1)
                # Update feature names to only include features common across all datasets
                common_features = list(Dg_train.columns[:-1])  # Exclude target
                if has_test_set:
                    common_features = [f for f in common_features if f in Dg_val.columns and f in Dg_test.columns]
                else:
                    common_features = [f for f in common_features if f in Dg_val.columns]
                # Ensure all datasets have exactly the same features
                Dg_train = pd.concat([Dg_train[common_features], Dg_train.iloc[:, -1]], axis=1)
                Dg_val = pd.concat([Dg_val[common_features], Dg_val.iloc[:, -1]], axis=1)
                if has_test_set:
                    Dg_test = pd.concat([Dg_test[common_features], Dg_test.iloc[:, -1]], axis=1)

            # Ensure feature_names represents features common to all datasets
            feature_names = list(Dg_train.columns)
            # Verify synchronization
            if not all(f in Dg_val.columns for f in feature_names[:-1]):
                info('ERROR: Feature mismatch between train and val')
                missing = [f for f in feature_names[:-1] if f not in Dg_val.columns]
                info(f'Missing in val: {missing}')
            if has_test_set and not all(f in Dg_test.columns for f in feature_names[:-1]):
                info('ERROR: Feature mismatch between train and test')
                missing = [f for f in feature_names[:-1] if f not in Dg_test.columns]
                info(f'Missing in test: {missing}')

            new_per = downstream_task(Dg_train, Dg_val, task_name, measure)
            if use_nni:
                nni.report_intermediate_result(new_per)
            reward = new_per - old_per
            r_c1, r_op, r_c2 = param['a'] / 10 * reward / 3, param['b'
            ] / 10 * reward / 3, param['c'] / 10 * reward / 3
            if new_per > best_per:
                always_best.append((Dg_train.columns, new_per, episode, step))
                best_episode = episode
                best_per = new_per
                D_OPT_train = Dg_train.copy()
                D_OPT_val = Dg_val.copy()
                # Sync best test set
                if has_test_set:
                    D_OPT_test = Dg_test.copy()
            if new_per > local_best:
                local_best = new_per
                Dg_train_local = Dg_train.copy()
                Dg_val_local = Dg_val.copy()
            all_results.append((Dg_train.columns, new_per, episode, step))
            old_per = new_per
            action_emb_c1_, state_emb_c1_, f_cluster_, clusters_ = (
                generate_next_state_of_meta_cluster1(Dg_train.values[:, :-1], y,
                                                     dqn_cluster1, cluster_num=CLUSTER_NUM, method=STATE_METHOD,
                                                     gpu=DEVICE))
            state_emb_op_, op_ = generate_next_state_of_meta_operation(
                f_cluster_, operation_set, dqn_operation, method=
                STATE_METHOD, gpu=DEVICE)
            if op in O2:
                action_emb_c2_, state_emb_c2_ = (
                    generate_next_state_of_meta_cluster2(f_cluster_,
                                                         operation_emb[op_], clusters_, Dg_train.values[:, :-1],
                                                         dqn_cluster2, method=STATE_METHOD, gpu=DEVICE))
                dqn_cluster2.store_transition(state_emb_c2, action_emb_c2,
                                              r_c2, state_emb_c2_, action_emb_c2_)
            dqn_cluster1.store_transition(state_emb_c1, action_emb_c1, r_c1,
                                          state_emb_c1_, action_emb_c1_)
            dqn_operation.store_transition(state_emb_op, op_index, r_op,
                                           state_emb_op_)
            if dqn_cluster1.memory_counter > dqn_cluster1.MEMORY_CAPACITY:
                dqn_cluster1.learn()
            if dqn_cluster2.memory_counter > dqn_cluster2.MEMORY_CAPACITY:
                dqn_cluster2.learn()
            if dqn_operation.memory_counter > dqn_operation.MEMORY_CAPACITY:
                dqn_operation.learn()

            info(
                'New performance is: {:.6f}, Best performance is: {:.6f} (e{}s{}) Base performance is: {:.6f}'
                .format(new_per, best_per, best_episode, best_step, base_per))
            info('Episode {}, Step {} ends!'.format(episode, step))
            best_per_opt.append(best_per)
            info('Current spend time for step-{} is: {:.1f}s'.format(step,
                                                                     time.time() - step_start_time))
            step += 1
        if episode != EPISODES - 1:
            best_features.append(pd.DataFrame(Dg_train_local.iloc[:, :-1]))
        else:
            best_features.append(Dg_train_local)
        episode += 1
        info('Current spend time for episode-{} is: {:.1f}s'.format(episode,
                                                                    time.time() - eps_start_time))
        if episode % 5 == 0:
            info('Best performance is: {:.6f}'.format(np.min(best_per_opt)))
            info('Episode {} ends!'.format(episode))
    info('Total spend time for is: {:.1f}s'.format(time.time() -
                                                   training_start_time))
    info('Exploration ends!')
    info('Begin evaluation...')
    if task_name == 'reg':
        mae0, mse0, rmse0 = test_task(D_original_train, D_original_val, task=task_name)
        mae1, mse1, rmse1 = test_task(D_OPT_train, D_OPT_val, task=task_name)
        if use_nni:
            nni.report_final_result(rmse1)
        info('1-MAE on original is: {:.5f}, 1-MAE on generated is: {:.5f}'.
             format(mae0, mae1))
        info('1-MSE on original is: {:.5f}, 1-MSE on generated is: {:.5f}'.
             format(mse0, mse1))
        info('1-RMSE on original is: {:.5f}, 1-RMSE on generated is: {:.5f}'.
             format(rmse0, rmse1))
    elif task_name == 'cls':
        acc0, precision0, recall0, f1_0 = test_task(D_original_train, D_original_val, task=
        task_name)
        acc1, precision1, recall1, f1_1 = test_task(D_OPT_train, D_OPT_val, task=
        task_name)
        if use_nni:
            nni.report_final_result(f1_1)
        info('Acc on original is: {:.5f}, Acc on generated is: {:.5f}'.
             format(acc0, acc1))
        info('Pre on original is: {:.5f}, Pre on generated is: {:.5f}'.
             format(precision0, precision1))
        info('Rec on original is: {:.5f}, Rec on generated is: {:.5f}'.
             format(recall0, recall1))
        info('F-1 on original is: {:.5f}, F-1 on generated is: {:.5f}'.
             format(f1_0, f1_1))
    elif task_name == 'det':
        map0, f1_0, ras0 = test_task(D_original_train, D_original_val, task=task_name)
        map1, f1_1, ras1 = test_task(D_OPT_train, D_OPT_val, task=task_name)
        if use_nni:
            nni.report_final_result(ras1)
        info(
            'Average Precision Score on original is: {:.5f}, Average Precision Score on generated is: {:.5f}'
            .format(map0, map1))
        info(
            'F1 Score on original is: {:.5f}, F1 Score on generated is: {:.5f}'
            .format(f1_0, f1_1))
        info(
            'ROC AUC Score on original is: {:.5f}, ROC AUC Score on generated is: {:.5f}'
            .format(ras0, ras1))
    else:
        error('wrong task name!!!!!')
        assert False

    # Export datasets for evaluation on another machine
    info('Exporting datasets for separate evaluation...')
    D_original_train.to_csv(base_path + param['file_name'] + '_eval_original_train.csv', index=False)
    D_original_val.to_csv(base_path + param['file_name'] + '_eval_original_val.csv', index=False)
    D_OPT_train.to_csv(base_path + param['file_name'] + '_eval_generated_train.csv', index=False)
    D_OPT_val.to_csv(base_path + param['file_name'] + '_eval_generated_val.csv', index=False)
    if has_test_set:
        D_original_test.to_csv(base_path + param['file_name'] + '_eval_original_test.csv', index=False)
        D_OPT_test.to_csv(base_path + param['file_name'] + '_eval_generated_test.csv', index=False)
        info(f'Exported 6 datasets to {base_path}')
    else:
        info(f'Exported 4 datasets to {base_path}')

    info('Total using time: {:.1f}s'.format(time.time() - training_start_time))
    D_OPT_train.to_csv(D_OPT_PATH + '/' + f'{current_trial_name}_{best_per}.csv')
    always_best_df = []
    with open(D_OPT_PATH + '/' + f'{current_trial_name}.bdata', 'w') as f:
        for col_name, per, epi, step_ in always_best:
            col_name = [str(i) for i in list(col_name)]
            line = str.join(f',{str(sep_token)},', col_name) + f',{per},{epi},{step_}\n'
            f.write(line)
    with open(D_OPT_PATH + '/' + f'{current_trial_name}.adata', 'w') as f:
        for col_name, per, epi, step_ in all_results:
            col_name = [str(i) for i in list(col_name)]
            line = str.join(f',{str(sep_token)},', col_name) + f',{per},{epi},{step_}\n'
            f.write(line)

    # Final evaluation on separate test set (model trained on training portion only)
    if has_test_set:
        info('========== Final Evaluation on Separate Test Set ==========')
        if task_name == 'reg':
            mae0_test, mse0_test, rmse0_test = test_task_separate(D_original_train, D_original_test, task=task_name)
            mae1_test, mse1_test, rmse1_test = test_task_separate(D_OPT_train, D_OPT_test, task=task_name)
            info('[SEPARATE TEST] 1-MAE on original is: {:.5f}, 1-MAE on generated is: {:.5f}'.
                 format(mae0_test, mae1_test))
            info('[SEPARATE TEST] 1-MSE on original is: {:.5f}, 1-MSE on generated is: {:.5f}'.
                 format(mse0_test, mse1_test))
            info('[SEPARATE TEST] 1-RMSE on original is: {:.5f}, 1-RMSE on generated is: {:.5f}'.
                 format(rmse0_test, rmse1_test))

            # Add TabularPredictor evaluation for regression
            info('========== TabularPredictor Evaluation on Separate Test Set ==========')
            mae0_test_ag, mse0_test_ag, rmse0_test_ag = test_task_separate_ag(D_original_train, D_original_val, D_original_test, task=task_name)
            mae1_test_ag, mse1_test_ag, rmse1_test_ag = test_task_separate_ag(D_OPT_train, D_OPT_val, D_OPT_test, task=task_name)
            info('[SEPARATE TEST AG] 1-MAE on original is: {:.5f}, 1-MAE on generated is: {:.5f}'.
                 format(mae0_test_ag if mae0_test_ag is not None else 0.0, mae1_test_ag if mae1_test_ag is not None else 0.0))
            info('[SEPARATE TEST AG] 1-MSE on original is: {:.5f}, 1-MSE on generated is: {:.5f}'.
                 format(mse0_test_ag if mse0_test_ag is not None else 0.0, mse1_test_ag if mse1_test_ag is not None else 0.0))
            info('[SEPARATE TEST AG] 1-RMSE on original is: {:.5f}, 1-RMSE on generated is: {:.5f}'.
                 format(rmse0_test_ag if rmse0_test_ag is not None else 0.0, rmse1_test_ag if rmse1_test_ag is not None else 0.0))
        elif task_name == 'cls':
            acc0_test, precision0_test, recall0_test, f1_0_test = test_task_separate(D_original_train, D_original_test, task=task_name)
            acc1_test, precision1_test, recall1_test, f1_1_test = test_task_separate(D_OPT_train, D_OPT_test, task=task_name)

            # Add TabularPredictor evaluation
            info('========== TabularPredictor Evaluation on Separate Test Set ==========')
            acc0_test_ag, precision0_test_ag, recall0_test_ag, f1_0_test_ag = test_task_separate_ag(D_original_train, D_original_val, D_original_test, task=task_name)
            acc1_test_ag, precision1_test_ag, recall1_test_ag, f1_1_test_ag = test_task_separate_ag(D_OPT_train, D_OPT_val, D_OPT_test, task=task_name)
            info('[SEPARATE TEST AG] Acc on original is: {:.5f}, Acc on generated is: {:.5f}'.
                 format(acc0_test_ag if acc0_test_ag is not None else 0.0, acc1_test_ag if acc1_test_ag is not None else 0.0))
            # info('[SEPARATE TEST AG] Pre on original is: {:.5f}, Pre on generated is: {:.5f}'.
            #      format(precision0_test_ag if precision0_test_ag is not None else 0.0, precision1_test_ag if precision1_test_ag is not None else 0.0))
            # info('[SEPARATE TEST AG] Rec on original is: {:.5f}, Rec on generated is: {:.5f}'.
            #      format(recall0_test_ag if recall0_test_ag is not None else 0.0, recall1_test_ag if recall1_test_ag is not None else 0.0))
            # info('[SEPARATE TEST AG] F-1 on original is: {:.5f}, F-1 on generated is: {:.5f}'.
            #      format(f1_0_test_ag if f1_0_test_ag is not None else 0.0, f1_1_test_ag if f1_1_test_ag is not None else 0.0))
            info('[SEPARATE TEST] Acc on original is: {:.5f}, Acc on generated is: {:.5f}'.
                 format(acc0_test, acc1_test))
            # info('[SEPARATE TEST] Pre on original is: {:.5f}, Pre on generated is: {:.5f}'.
            #      format(precision0_test, precision1_test))
            # info('[SEPARATE TEST] Rec on original is: {:.5f}, Rec on generated is: {:.5f}'.
            #      format(recall0_test, recall1_test))
            # info('[SEPARATE TEST] F-1 on original is: {:.5f}, F-1 on generated is: {:.5f}'.
            #      format(f1_0_test, f1_1_test))
        elif task_name == 'det':
            map0_test, f1_0_test, ras0_test = test_task_separate(D_original_train, D_original_test, task=task_name)
            map1_test, f1_1_test, ras1_test = test_task_separate(D_OPT_train, D_OPT_test, task=task_name)
            info('[SEPARATE TEST] Average Precision Score on original is: {:.5f}, Average Precision Score on generated is: {:.5f}'.
                 format(map0_test, map1_test))
            info('[SEPARATE TEST] F1 Score on original is: {:.5f}, F1 Score on generated is: {:.5f}'.
                 format(f1_0_test, f1_1_test))
            info('[SEPARATE TEST] ROC AUC Score on original is: {:.5f}, ROC AUC Score on generated is: {:.5f}'.
                 format(ras0_test, ras1_test))
    else:
        info('No separate test set available for final evaluation')


if __name__ == '__main__':
    try:
        args = init_param()
        tuner_params = nni.get_next_parameter()
        trail_id = nni.get_trial_id()
        params = vars(merge_parameter(args, tuner_params))
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp/')
        if not os.path.exists('./tmp/' + params['file_name'] + '/'):
            os.mkdir('./tmp/' + params['file_name'] + '/')
        start_time = str(time.asctime())
        debug(tuner_params)
        info(params)
        model_train(params, nni)
    except Exception as exception:
        error(exception)
        raise
