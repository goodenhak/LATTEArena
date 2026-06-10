import os
import random
from openai import OpenAI
import time
import torch
import json
import pandas as pd
import numpy as np
import re
import shutil
from pathlib import Path

def remove_bold(line: str) -> str:
    # Remove Markdown bold markers **
    line = re.sub(r"\*\*", "", line)
    # Remove quotation marks
    line = re.sub(r"\"", "", line)
    # Remove backticks
    line = re.sub(r"\`", "", line)
    return line.strip()

def read_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        return content

def read_txt_to_list(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove newline and whitespace, split by comma, and filter empty items
            items = [item.strip() for item in line.strip().split(',') if item.strip()]
            result.extend(items)
    return result

def extract_json_content(raw_string):
    # Find the position of the first { and last }
    start = raw_string.find('{')
    end = raw_string.rfind('}')

    if start == -1 or end == -1:
        raise ValueError("No valid JSON content wrapped in {} found in string")

    # Extract content (including {})
    return raw_string[start:end+1]

def get_data_samples(df_sampled, target):
    data_samples = ""
    df_current = df_sampled.groupby(
        target, group_keys=False
    ).apply(lambda x: x.sample(frac=1))

    for _, icl_row in df_current.iterrows():
        answer = icl_row[target]
        icl_row = icl_row.drop(labels=target)
        data_samples += serialize(icl_row)
        data_samples += f"\nAnswer: {answer}\n"
    
    return data_samples

def serialize(row):
    target_str = f""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += ". "
        else:
            # if len(attr_name.strip()) < 2:
            #     continue
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += "."
    return target_str

def fill_template(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str

def query_llm(prompt, max_tokens=8000, temperature=1, model="deepseek-chat"):
    if model=="deepseek-chat" or model=="deepseek-reasoner":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif model=="llama-3.1-8b-instruct" or model=="llama-3.1-70b-instruct":
        model = "meta-llama/" + model
        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = 1
    )
    result = response.choices[0].message.content
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }
    return result, token_usage

def query_critic(last_user ,last_llm , prompt, max_tokens=8000, temperature=1, model="deepseek-chat"):
    if model=="deepseek-chat" or model=="deepseek-reasoner":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif model=="llama-3.1-8b-instruct" or model=="llama-3.1-70b-instruct":
        model = "meta-llama/" + model
        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": last_user},
            {"role": "assistant", "content": last_llm},
            {"role":"user", "content":prompt}],
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = 1
    )
    result = response.choices[0].message.content
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }
    return result, token_usage

def extract_function_body(code):
    lines = code.split('\n')
    start_line = None
    end_line = None
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith('def feature_generation(df):'):
            start_line = i + 1
        elif start_line is not None and stripped_line.startswith('return df'):
            end_line = i
            break
    if start_line is None or end_line is None:
        return ""
    return lines[start_line:end_line]

def copy_and_rename_metadata(data_name):
    src_file = Path(f"./metadata/{data_name}.json")
    dst_dir = Path(f"./tmp/{data_name}")
    dst_file = dst_dir / "metadata.json"

    # Ensure source file exists
    if not src_file.exists():
        raise FileNotFoundError(f"Source file does not exist: {src_file}")

    # Create destination directory (if it doesn't exist)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy and rename
    shutil.copy(src_file, dst_file)
    print(f"Copied {src_file} to {dst_file}")

def clear_files(directory):
    """
    Clear the contents of files in the specified directory and subdirectories
    that match the filenames list.

    :param directory: Target folder path
    :param filenames: List of file names to clear (exact match)
    """
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return
    filenames = ["best_train.csv","best_test.csv","feature_generation.py","full_code.py","metadata.json","train.csv","val.csv","test.csv"]
    cleared_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file in filenames:
                file_path = os.path.join(root, file)
                try:
                    open(file_path, 'w').close()
                    print(f"Cleared: {file_path}")
                    cleared_files += 1
                except Exception as e:
                    print(f"Failed to clear {file_path}: {e}")

    print(f"Cleared {cleared_files} file(s) in total.")

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and remove duplicate column names from DataFrame, keeping only the first occurrence.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with duplicate columns removed
    """
    # Use pandas Index.duplicated method to detect duplicate column names
    return df.loc[:, ~df.columns.duplicated()]
