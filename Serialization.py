# input: task_description.json, tabular_data/data_name.csv, and metadata/data_name.json
# output: initail prompt
# args: data_name, sample_size, sample_method, metadata_cat, demo_format, op_type, type
import json
import pandas as pd
import numpy as np
import utils

class Serializer:
    def __init__(self, data_name, ensemble, sample_size, sample_method, 
    demo_format, op_type, task_type, target, metadata_cat):
        self.data_name = data_name
        self.ensemble = ensemble # default: 1
        self.sample_size = sample_size # 0~10
        self.sample_method = sample_method #0:random 1:stratified
        self.metadata_cat = metadata_cat # 0:native 1:human-written(+type) 2:LLM 3:calculated
        self.demo_format = demo_format # 0:no demo 1:feature 2:+reason 3:+score
        self.op_type = op_type # 0:no op 1:simple 2:+complex 3:+extractor
        self.task_type = task_type # 0:regression 1:classification
        self.target = target
        self.token_usage = 0

    def get_token_usage(self):
        """Return the accumulated token usage and reset the counter"""
        usage = self.token_usage
        self.token_usage = 0
        return usage

    def get_samples(self, data):
        sample_data_list = []
        for i in range(self.ensemble):
            if self.task_type == 0:
                if self.sample_method == 0:
                    sample_data = data.sample(self.sample_size)
                elif self.sample_method == 1:
                    # for regression, do binning first
                    data['bin'] = pd.cut(data[self.target], bins=5, labels=False)
                    sample_data = data.groupby('bin').apply(lambda x: x.sample(self.sample_size//len(data['bin'].unique()))).reset_index(drop=True)
                    data = data.drop('bin', axis=1)
            elif self.task_type == 1:
                if self.sample_method == 0:
                    sample_data = data.sample(self.sample_size)
                elif self.sample_method == 1:
                    # for classification, do stratified sampling
                    sample_data = data.groupby(self.target).apply(lambda x: x.sample(self.sample_size//len(data[self.target].unique()))).reset_index(drop=True)
            sample_data_txt = utils.get_data_samples(sample_data, self.target)
            sample_data_list.append(sample_data_txt)
        return sample_data_list
    
    def get_metadata(self, df, is_cat):
        feature_name_list = []
        metadata_path = f"./tmp/{self.data_name}/metadata.json"

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Track if we need to save updated metadata (for metadata_cat == 2)
        metadata_updated = False

        if self.metadata_cat == 0:
            for cidx, cname in enumerate(df.columns):
                if cname == self.target:
                    continue
                desc = metadata.get(cname, "").strip()
                feature_name_list.append(f"{cname}: {desc}")
        elif self.metadata_cat == -1:
            for cidx, cname in enumerate(df.columns):
                if cname == self.target:
                    continue
                desc = metadata.get(cname, "").strip()
                feature_name_list.append(f"{cname}")
        elif self.metadata_cat == 1:
            is_cat_idx = 0
            for cidx, cname in enumerate(df.columns):
                if cname == self.target:
                    continue
                desc = metadata.get(cname, "").strip()
                ftype = "categorical" if is_cat[is_cat_idx] else "numerical"
                line = f"- {cname}: {desc} ({ftype})"
                feature_name_list.append(line)
                is_cat_idx += 1
        elif self.metadata_cat == 2:
            # Check if we have a separate rewritten metadata file
            rewritten_metadata_path = f"./tmp/{self.data_name}/metadata_rewritten.json"
            try:
                with open(rewritten_metadata_path) as f:
                    rewritten_metadata = json.load(f)
            except FileNotFoundError:
                rewritten_metadata = {}

            is_cat_idx = 0
            for cidx, cname in enumerate(df.columns):
                if cname == self.target:
                    continue

                # Check if this feature has already been rewritten
                if cname in rewritten_metadata:
                    # Use cached rewritten description
                    feature_name_list.append(rewritten_metadata[cname])
                else:
                    # Need to rewrite this feature
                    desc = metadata.get(cname, "").strip()
                    ftype = "categorical" if is_cat[is_cat_idx] else "numerical"
                    line = f"- {cname}: {desc} ({ftype})"
                    data_sample_list = self.get_samples(df)
                    prompt = f"Data Samples: {data_sample_list}. Rewrite the following a feature description to be more detailed. Do not return other features' description.\nOutput format: - name: desc (type)\n"
                    prompt += line
                    result, token_usage = utils.query_llm(prompt)
                    feature_name_list.append(result)
                    self.token_usage += token_usage["total_tokens"]

                    # Cache the rewritten metadata
                    rewritten_metadata[cname] = result
                    metadata_updated = True

                is_cat_idx += 1

            # Save updated rewritten metadata if any new features were rewritten
            if metadata_updated:
                with open(rewritten_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(rewritten_metadata, f, ensure_ascii=False, indent=2)

        elif self.metadata_cat == 3:
            is_cat_idx = 0
            for cidx, cname in enumerate(df.columns):
                if cname == self.target:
                    continue
                desc = metadata.get(cname, "").strip()
                ftype = "categorical" if is_cat[is_cat_idx] else "numerical"
                line = f"- {cname}: {desc} ({ftype})"
                col_data = df[cname]
                if ftype == "numerical":
                    min_val = col_data.min()
                    q1 = col_data.quantile(0.25)
                    median = col_data.median()
                    q3 = col_data.quantile(0.75)
                    max_val = col_data.max()
                    mean = col_data.mean()
                    std = col_data.std()
                    line += (f", range = [{min_val}, {max_val}], Q1 = {q1}, Median = {median}, Q3 = {q3}, "
                                f"Mean = {mean:.4f}, Std = {std:.4f}")
                else:
                    clist = col_data.unique().tolist()
                    clist_str = (
                        f"{clist[0]}, {clist[1]}, ..., {clist[-1]}" if len(clist) > 20 else ", ".join(map(str, clist))
                    )
                    line += f", categories = [{clist_str}]"
                feature_name_list.append(line)
                is_cat_idx += 1

        return "\n".join(feature_name_list)
    
    def generate_initial_prompt(self):
        prompt_list = []
        # get task description from json
        with open("task_descriptions.json", "r") as f:
            task_list = json.load(f)
        task_desc = task_list[self.data_name]
        # load data
        data = pd.read_csv(f"./tmp/{self.data_name}/train.csv")
        # Detect categorical features
        is_cat = [
            pd.api.types.is_string_dtype(dt) or
            pd.api.types.is_categorical_dtype(dt) or
            pd.api.types.is_bool_dtype(dt)
            for dt in data.dtypes.tolist()
        ][:-1]
        # load metadata
        metadata = self.get_metadata(data, is_cat)
        # sample data
        if self.sample_size == 0:
            data_sample_list = []
        else:
            data_sample_list = self.get_samples(data)
        # read template txt
        with open("templates/ask_llm.txt", "r") as f:
            template = f.read()
        # read operators txt
        with open("templates/operators_simple.txt", "r") as f:
            simple_op = f.read()
        with open("templates/operators_complex.txt", "r") as f:
            complex_op = f.read()
        # fill template with data, metadata, and task description
        for i in range(self.ensemble):
            if data_sample_list == []:
                data_sample = None
                fill_in_dict = {
                    "[TASK]": task_desc, 
                    "[FEATURES]": metadata,
                }
            else:
                data_sample = data_sample_list[i]
                fill_in_dict = {
                        "[TASK]": task_desc, 
                        "[EXAMPLES]": data_sample,
                        "[FEATURES]": metadata,
                    }
            prompt = utils.fill_template(fill_in_dict, template)
            if self.op_type == 1:
                prompt += "\nOperators:\n" + simple_op
            elif self.op_type == 2:
                prompt += "\nOperators:\n" + simple_op + complex_op
            elif self.op_type == 3:
                prompt += "\nOperators:\n" + simple_op + complex_op
                prompt += "Extractor" # TODO: add extractor
            prompt_list.append(prompt)
        return prompt_list
    
    def generate_prompt_components(self):
        # Get task description
        with open("task_descriptions.json", "r") as f:
            task_list = json.load(f)
        task_desc = task_list[self.data_name]

        # Get feature metadata
        data = pd.read_csv(f"./tmp/{self.data_name}/train.csv")
        # Detect categorical features
        is_cat = [
            pd.api.types.is_string_dtype(dt) or
            pd.api.types.is_categorical_dtype(dt) or
            pd.api.types.is_bool_dtype(dt)
            for dt in data.dtypes.tolist()
        ][:-1]
        metadata = self.get_metadata(data, is_cat)

        # Get data samples
        if self.sample_size == 0:
            data_sample = ""
        else:
            data_sample_list = self.get_samples(data)
            data_sample = data_sample_list[0] if data_sample_list else ""

        return task_desc, metadata, data_sample
    
    def generate_critic_prompt(self):
        prompt_list = []
        # get task description from json
        with open("task_descriptions.json", "r") as f:
            task_list = json.load(f)
        task_desc = task_list[self.data_name]
        # load data
        data = pd.read_csv(f"./tmp/{self.data_name}/train.csv")
        # Detect categorical features
        is_cat = [
            pd.api.types.is_string_dtype(dt) or
            pd.api.types.is_categorical_dtype(dt) or
            pd.api.types.is_bool_dtype(dt)
            for dt in data.dtypes.tolist()
        ][:-1]
        # load metadata
        metadata = self.get_metadata(data, is_cat)
        # sample data
        if self.sample_size == 0:
            data_sample_list = []
        else:
            data_sample_list = self.get_samples(data)
        # read template txt
        with open("templates/ask_critic.txt", "r") as f:
            template = f.read()
        # fill template with data, metadata, and task description
        for i in range(self.ensemble):
            if data_sample_list == []:
                data_sample = None
                fill_in_dict = {
                    "[TASK]": task_desc, 
                    "[FEATURES]": metadata,
                }
            else:
                data_sample = data_sample_list[i]
                fill_in_dict = {
                        "[TASK]": task_desc, 
                        "[EXAMPLES]": data_sample,
                        "[FEATURES]": metadata,
                    }
            prompt = utils.fill_template(fill_in_dict, template)
            prompt_list.append(prompt)
        return prompt_list