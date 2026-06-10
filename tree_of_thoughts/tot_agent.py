import Evaluator
import Postprocessor
import Selector
import Serialization
import utils
import re
import os
from autogluon.tabular import TabularDataset, TabularPredictor
import logging

logger = logging.getLogger(__name__)

class ToTAgent:
    """

    OpenAI Language Model API Wrapper

    Args:
        agent (Agent): Agent class from swarms
        strategy (str): Strategy to use for generating thoughts
        evaluation_strategy (str): Strategy to use for evaluating states
        enable_react (bool): Enable ReAct prompting
        k (int): Number of thoughts to generate

    Methods:
        run(task: str) -> list: Generate text from prompt using OpenAI API
        generate_thoughts(state, k, initial_prompt, mid_prompt, rejected_solutions=None) -> list: Generate thoughts from state using OpenAI API
        generate_solution(initial_prompt, state, rejected_solutions=None) -> str: Generate solution from state using OpenAI API
        evaluate_states(states, initial_prompt) -> dict: Evaluate states of reasoning using OpenAI API

    Examples:
        >>> from tree_of_thoughts.tot_agent import ToTAgent
        >>> from swarms import Agent
        >>> agent = Agent()
        >>> model = ToTAgent(agent)
        >>> thoughts = model.run("Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'.")
        >>> logger.info(thoughts)
        ['Observation:xxxx', 'Thoughts:xxxx']

    """

    def __init__(
        self,
        model: str = "DecisionTree",
        llm_model: str = "gpt-4o",
        strategy: str = "cot",
        max_f: int=20,
        evaluation_strategy: str = "value",
        enable_react: bool = True,
        k: int = 3,
        output_format: str = "cRPN",
        *args,
        **kwargs,
    ):
        self.use_chat_api = True
        self.enable_react = enable_react
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        self.k = k
        self.llm_model = llm_model
        self.model_name = model
        self.total_tokens = 0
        self.max_f = max_f
        self.output_format = output_format

        # reference : https://www.promptingguide.ai/techniques/react
        if enable_react == True:
            self.react_prompt = (
                "Write down your operations in format 'Operation:xxxx',(When using a `groupbythen` operator, ensure that the first attribute is the grouping key.)"
                " then write down the attributes you are operating on in format 'Attribute:xxxx',"
                " then write down your reasoning in format 'Reasoning:xxxx'."
            )
        else:
            if self.output_format == "NL":
                self.react_prompt = ("""
Instructions:
1. Use '<>' for new_feature_name, operator, feature1, feature2 and new_feature_description, NO space or \ between. Each line has only one operator.
2. DO NOT divide, plus or subtract categorical_feature, use cross, concat or type conversion.
3. Type Conversion:
    1. <numerical_feature><label_encode><categorical_feature>
    2. <categorical_feature><bin><numerical_feature>
4. The first parameter of the groupbythen operator is the grouping key.

Output Format:
1. <new_feature_name><operator><feature1><feature2><new_feature_description>
2. <new_feature_name><operator><feature1><new_feature_description>
3. ...

DO NOT RETURN ANYTHING ELSE.""")
            elif self.output_format == "cRPN":
                self.react_prompt = ("""
Output in Reverse Polish Notation.

For example, suppose we have feature f1 and f2 and want to create a new feature:
new_feature = log(f1 + f2)
The corresponding RPN expression would be:
f1 f2 plus log

Instructions:
1. ONLY USE {} at the beginning and end of RPNs, use commas to separate multiple RPNs, and use spaces within each RPN.
2. DO NOT divide, plus or subtract categorical_feature, use cross, concat or type conversion.
3. Type Conversion:
    1. numerical_feature = categorical_feature label_encode
    2. categorical_feature = numerical_feature 5 bin (choose the best bin num you think)
4. The first parameter of groupbythen operator is the grouping key.
    For example: categorical_feature numerical_feature groupbythenmean

Output example:
{f1 f2 plus log, f1 f4 subtract, f1 sigmoid cosine}

DO NOT RETURN ANYTHING ELSE.""")
            elif self.output_format == "Code":
                self.react_prompt = ("""
Code formatting for each added column:
```python
# Feature name
# Feature description
# Usefulness: (Description why this adds useful real world knowledge according to dataset description and attributes.)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock ends with ```end and starts with "```python"
Codeblock:
""")

    def run(self, task: str):
        """Generate text from prompt using"""
        if self.use_chat_api:
            thoughts = []
            for _ in range(self.k):
                content, token_usage = utils.query_llm(prompt=task,model=self.llm_model)
                
                thoughts += [content]
                
                # Print token usage for this call
                tokens = token_usage['total_tokens']

                logger.info(f"LLM API call consumed {tokens} tokens")
                self.total_tokens += tokens
                    
            logger.info(f"Total tokens consumed in this batch: {self.total_tokens}")
            return thoughts


    def generate_thoughts(
                self, state, k, initial_prompt, mid_prompt, operators, rejected_solutions=None, use_history=1
        ):
            """
            Generate thoughts from state using OpenAI API

            Args:
                state (str or list): State of reasoning
                k (int): Number of thoughts to generate
                initial_prompt (str): Initial prompt
                rejected_solutions (list): List of rejected solutions
                use_history (int): Whether to use history in prompt (1=yes, 0=no)

            Returns:
                list: List of thoughts
            """
            if type(state) == str:
                state_text = state
            else:
                state_text = "\n".join(state)

            features_str, data_examples = mid_prompt
            # # Update features_str
            # # 将string按行拆分
            # lines = features_str.strip().split("\n")

            # # 提取每行的feature名（冒号前去掉"- "）
            # line_features = []
            # for line in lines:
            #     if line.startswith("- "):
            #         feature = line.split(":")[0].strip("- ").strip()
            #         line_features.append(feature)

            # # 保留feature在list中的行
            # filtered_lines = [
            #     line for line in lines
            #     if line.split(":")[0].strip("- ").strip() in current_feature_names
            # ]

            # # 找出list中有但string中没有的feature
            # missing_features = [f for f in current_feature_names if f not in line_features]

            # # 在末尾添加缺失feature
            # for f in missing_features:
            #     filtered_lines.append(f"- {f}: Generated new feature,")

            # # 生成新的字符串
            # features_str = "\n".join(filtered_lines)


            # logger.info("New state generating thought:", state, "\n\n")

            # Adjust feature names display based on the number of items
            # if len(current_feature_names) > 40:
            #     features_display = ', '.join(current_feature_names[:3]) + ', ..., ' + ', '.join(
            #         current_feature_names[-6:])
            # else:
            #     features_display = ', '.join(current_feature_names)

            if state == initial_prompt:
                prompt = f"""You are an expert datascientist working to improve predictions.
You perform feature engineering that generate additional columns that are useful for a downstream task.

Task:
{initial_prompt}

Features:
{features_str}

Data Examples:
{data_examples}

Operators:
{operators}

Devise the best possible solution for the task:
"""
                if use_history:
                    prompt += f"""Here are evaluated solutions that were rejected:
###{rejected_solutions}###,
complete the task without making the same mistakes as before. Be simple, direct, and intuitive.
"""
                prompt += """Possible next three steps:
"""
            else:
                prompt = f"""You are an expert datascientist working to improve predictions.
You perform feature engineering that generate additional columns that are useful for a downstream task.

Task:
{initial_prompt}

Features:
{features_str}

Data Examples:
{data_examples}

Operators:
{operators}
"""
                if use_history:
                    prompt += f"""
Accepted solutions so far:

###'{state_text}'###
"""
                prompt += f"""
Devise the best possible solution for the task:
"""
                if use_history:
                    prompt += f"""Here are evaluated solutions that were rejected:
###{rejected_solutions}###,
complete the task without making the same mistakes as before. Be simple, direct, and intuitive.
"""
                prompt += """The possible next three step:
"""

            prompt += self.react_prompt
            # Generate k thoughts

            logger.info(f"thoughts: {prompt}")

            thoughts = self.run(prompt)
            logger.info(f"thoughts: {thoughts}")

            return thoughts

    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        try:
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                state_text = state

            prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:
            ###'{state_text}'###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            # logger.info("solution input prompt: "+prompt)
            answer = self.run(prompt)
            logger.info(f"Answer {answer}")
            # logger.info(thoughts)
            # logger.info(f"General Solution : {answer}")
            return answer
        except Exception as e:
            logger.info(f"Error in generate_solutions: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                logger.info(
                    "We receive a state of type",
                    type(state),
                    "For state: ",
                    state,
                    "\n\n",
                )
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not directly concretely making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                # logger.info("evaluate prompt(value): "+prompt)
                response, token_usage = utils.query_llm(prompt=prompt,model=self.llm_model)
                # Print token usage for this call
                tokens = token_usage['total_tokens']

                logger.info(f"LLM API call consumed {tokens} tokens")
                self.total_tokens += tokens
                try:
                    value_text = response.content
                    value = float(value_text)
                    logger.info(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == "vote":
            states_text = "\n".join([" ".join(state) for state in states])
            prompt = (
                "Given the following states of reasoning, vote for the best"
                " state utilizing an scalar value"
                f" 1-10:\n{states_text}\n\nVote, on the probability of this"
                f" state of reasoning achieveing {initial_prompt} and become"
                " very pessimistic very NOTHING ELSE"
            )
            logger.info("evaluate prompt(vote): "+prompt)
            response, token_usage = utils.query_llm(prompt=prompt,model=self.llm_model)
            # Print token usage for this call
            tokens = token_usage['total_tokens']

            logger.info(f"LLM API call consumed {tokens} tokens")
            self.total_tokens += tokens
            logger.info(f"state response: {response}")
            best_state_text, token_usage = utils.query_llm(prompt=response.choices[0],model=self.llm_model)
            # Print token usage for this call
            tokens = token_usage['total_tokens']

            logger.info(f"LLM API call consumed {tokens} tokens")
            self.total_tokens += tokens
            logger.info(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            logger.info(f"best_state: {best_state}")

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError(
                "Invalid evaluation strategy. Choose 'value' or 'vote'."
            )

    def format_states(self, states):
        if not states:
            return {}

        if self.enable_react == False:
            return states
        
        formatted_responses = []  # Initialize an empty list to store responses

        for state in states:
            if type(state) == str:
                state_text = state
            else:
                state_text = "\n".join(state)
            # logger.info(
            #     "We receive a state of type",
            #     type(state),
            #     "For state: ",
            #     state,
            #     "\n\n",
            # )

            # Adjust feature names display based on the number of items
            # if len(feature_names) > 40:
            #     features_display = ', '.join(feature_names[:3]) + ', ..., ' + ', '.join(
            #         feature_names[-6:])
            # else:
            #     features_display = ', '.join(feature_names)


            prompt = f"""
Please reformat the latest solution using the structure provided below. Evaluate the context from past solutions to align the new format.

Current solution needs reformatting:
{state_text}

Instructions:
1. Use '<>' for new_feature_name, operator, feature1, feature2 and new_feature_description, NO space or \ between. Each line has only one operator.
2. DO NOT divide, plus or subtract categorical_feature, use cross, concat or type conversion.
3. Type Conversion: 
    1. <numerical_feature><label_encode><categorical_feature>
    2. <categorical_feature><bin><numerical_feature>
4. The first parameter of the groupbythen operator is the grouping key.
                                 
Output Format:
1. <new_feature_name><operator><feature1><feature2><new_feature_description>
2. <new_feature_name><operator><feature1><new_feature_description>
3. ...

DO NOT RETURN ANYTHING ELSE.
"""


            formatted_response, token_usage = utils.query_llm(prompt=prompt,model=self.llm_model)
            # Print token usage for this call
            tokens = token_usage['total_tokens']
            logger.info(f"formatted response:{formatted_response}")

            logger.info(f"LLM API call consumed {tokens} tokens")
            self.total_tokens += tokens
            formatted_responses.append(formatted_response)  # Append the response to the list

        return formatted_responses

    def evaluate_states_feature_generation(self,
                                           data_name,
                                           target,
                                           max_f,
                                           formatted_states,
                                           initial_metrics,
                                           task_type,
                                           use_selector=1,
                                           ):
        downstream = "rf"       #[rf,ag]

        if not formatted_states:
            return {}

        evaluated_states = {}
        for state in formatted_states:

            initial_accuracy, _, _, _ = initial_metrics

            llm_output = utils.remove_bold(state)
            logger.info(f"LLM Output: {llm_output}")
            # exec_code = Postprocessor.NL2Code(llm_output, data_name)
            # exec_code = exec_code.removeprefix("<start>").removesuffix("<end>")
            # logger.info(f"Generated Code:\n{exec_code}")

            # metadata = Postprocessor.extract_metadata(llm_output, data_name)
            # logger.info(f"Extracted Metadata: {metadata}")

            train_data, val_data, test_data = Evaluator.load_dataset(data_name)
            if self.output_format == 'NL':
                success_ops, new_train = Postprocessor.NL2exec(llm_output, data_name, train_data)
                logger.info(f"Success Operators:\n{success_ops}")
                _, new_test = Postprocessor.NL2exec(llm_output, data_name, test_data)
                _, new_val = Postprocessor.NL2exec(llm_output, data_name, val_data)
            elif self.output_format == 'cRPN':
                success_ops, new_train = Postprocessor.RPN2exec(llm_output, data_name, train_data)
                logger.info(f"Success Operators:\n{success_ops}")
                _, new_test = Postprocessor.RPN2exec(llm_output, data_name, test_data)
                _, new_val = Postprocessor.RPN2exec(llm_output, data_name, val_data)
            elif self.output_format == 'Code':
                success_ops, new_train = Postprocessor.Code2exec(llm_output, data_name, train_data, target)
                logger.info(f"Success Operators:\n{success_ops}")
                _, new_test = Postprocessor.Code2exec(llm_output, data_name, test_data, target)
                _, new_val = Postprocessor.Code2exec(llm_output, data_name, val_data, target)

            metadata = Postprocessor.exec_metadata(success_ops, data_name)
            logger.info(f"Extracted Metadata: {metadata}")


            # namespace = {}
            # exec(exec_code, globals(), namespace)
            # feature_generation_func = namespace.get('feature_generation')

            # new_val = feature_generation_func(val_data)
            # new_train = feature_generation_func(train_data)
            # new_test = feature_generation_func(test_data)

            if downstream == 'ag':
                new_predictor, new_val_acc = Evaluator.train_and_evaluate(new_train, new_val, target, task_type)
                logger.info(f"new_val_acc = {new_val_acc}")
                if use_selector:
                    val_importance = new_predictor.feature_importance(new_val,time_limit=600)
                    sel_train, dropped_columns = Selector.keep_most_important_features(new_train, val_importance, max_f)
                    sel_train[target] = new_train[target]
                    sel_val, dropped_columns = Selector.keep_most_important_features(new_val, val_importance, max_f)
                    sel_val[target] = new_val[target]
                    sel_test, dropped_columns = Selector.keep_most_important_features(new_test, val_importance, max_f)
                    sel_test[target] = new_test[target]
                    logger.info(f"dropped columns = {dropped_columns}")
                    if len(dropped_columns)>1:
                        sel_predictor, sel_val_acc = Evaluator.train_and_evaluate(sel_train, sel_val, target, task_type)
                        logger.info(f"sel_val_acc = {sel_val_acc}")

                        if sel_val_acc > new_val_acc:
                            new_val_acc = sel_val_acc
                            metadata = Selector.update_metadata(metadata, dropped_columns)
                            new_train = sel_train
                            new_test = sel_test
                            new_val = sel_val
            elif downstream == "rf":
                new_predictor, new_val_acc = Evaluator.train_and_evaluate_rf(new_train, new_val, target, task_type)
                logger.info(f"new_val_acc = {new_val_acc}")
                if use_selector:
                    val_importance = Selector.calculate_rf_importance(new_train,new_val,new_predictor,target,task_type)
                    sel_train, dropped_columns = Selector.select_top_features(new_train, val_importance, max_f)
                    sel_val, dropped_columns = Selector.select_top_features(new_val, val_importance, max_f)
                    sel_test, dropped_columns = Selector.select_top_features(new_test, val_importance, max_f)
                    logger.info(f"dropped columns = {dropped_columns}")
                    if len(dropped_columns)>0:
                        sel_predictor, sel_val_acc = Evaluator.train_and_evaluate_rf(sel_train, sel_val, target, task_type)
                        logger.info(f"sel_val_acc = {sel_val_acc}")

                        if sel_val_acc > new_val_acc:
                            new_val_acc = sel_val_acc
                            metadata = Selector.update_metadata(metadata, dropped_columns)
                            new_train = sel_train
                            new_test = sel_test
                            new_val = sel_val



            # Calculate improvements
            new_accuracy = new_val_acc
            accuracy_improvement = new_accuracy - initial_accuracy
            state_summary = llm_output
            evaluated_states[state_summary] = (accuracy_improvement, new_accuracy, new_train, new_val, new_test, metadata)

        return evaluated_states
