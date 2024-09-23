import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os

import determined as det
from determined.transformers import DetCallback

import pandas as pd

import os
import shutil

import pachyderm_sdk
from pachyderm_sdk.api.pfs import File, FileType

def prepare_train_data(data_file_names):
    # Convert the data to a Pandas DataFrame
    #data_df = pd.read_csv(data_file_name, sep=";")
    df_list = [pd.read_csv(data_file_name, sep=";") for data_file_name in data_file_names]
    data_df = pd.concat(df_list)
    
    data_df.reset_index(drop=True, inplace=True)
    
    # Create a new column called "text"
    data_df["text"] = data_df[["Question", "Answer"]].apply(lambda x: "<|im_start|>user\n" + x["Question"] + " <|im_end|>\n<|im_start|>assistant\n" + x["Answer"] + "<|im_end|>\n", axis=1)
    # Create a new Dataset from the DataFrame
    data = Dataset.from_pandas(data_df)
    return data


def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
      model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=True
    model.config.pretraining_tp=1
    return model, tokenizer

def safe_open_wb(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'wb')

def download_pach_repo(
    pachyderm_host,
    pachyderm_port,
    repo,
    branch,
    root,
    token,
    project="default",
    previous_commit=None,
):
    print(f"Starting to download dataset: {repo}@{branch} --> {root}")

    if not os.path.exists(root):
        os.makedirs(root)

    client = pachyderm_sdk.Client(
        host=pachyderm_host, port=pachyderm_port, auth_token=token
    )
    files = []
    if previous_commit is not None:
        for diff in client.pfs.diff_file(new_file=File.from_uri(f"{project}/{repo}@{branch}"),
            old_file=File.from_uri(f"{project}/{repo}@{previous_commit}")
        ):
            src_path = diff.new_file.file.path
            des_path = os.path.join(root, src_path[1:])
            print(f"Got src='{src_path}', des='{des_path}'")

            if diff.new_file.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))
    else:
        for file_info in client.pfs.walk_file(file=File.from_uri(f"{project}/{repo}@{branch}")):
            src_path = file_info.file.path
            des_path = os.path.join(root, src_path[1:])
            print(f"Got src='{src_path}', des='{des_path}'")

            if file_info.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))

    for src_path, des_path in files:
        src_file = client.pfs.pfs_file(file=File.from_uri(f"{project}/{repo}@{branch}:{src_path}"))
        print(f"Downloading {src_path} to {des_path}")

        with safe_open_wb(des_path) as dest_file:
            shutil.copyfileobj(src_file, dest_file)

    print("Download operation ended")
    return files

def download_data(data_config, context):
    #data_config = context.get_data_config()
    download_directory = (
            f"/tmp/data-rank{context.distributed.get_rank()}"
    )
    data_dir = os.path.join(download_directory, "data")

    files = download_pach_repo(
        data_config["pachyderm"]["host"],
        data_config["pachyderm"]["port"],
        data_config["pachyderm"]["repo"],
        data_config["pachyderm"]["branch"],
        data_dir,
        data_config["pachyderm"]["token"],
        data_config["pachyderm"]["project"],
        data_config["pachyderm"]["previous_commit"],
    )
    print(f"Data dir set to : {data_dir}")

    return [des for src, des in files]

if __name__ == "__main__":
    
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    distributed = det.core.DistributedContext.from_torch_distributed()
    training_args = TrainingArguments(**hparams["training_args"])
    
    with det.core.init(distributed=distributed) as core_context:
        det_callback = DetCallback(
            core_context,
            training_args,
        )
        
        data_config = info.user_data
        data_files = download_data(data_config, core_context)
        
        #dataset_name = hparams["train_data"]
        #dataset_name = files[0] # assume a single file is added at a time
        data = prepare_train_data(data_files)
        data = data.train_test_split(test_size=0.2, seed=42)
        
        model_id = hparams["model"]
        model, tokenizer = get_model_and_tokenizer(model_id)
        
        peft_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_args,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=1024
        )
        
        trainer.add_callback(det_callback)
        trainer.train()
