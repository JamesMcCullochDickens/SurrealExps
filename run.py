import warnings
warnings.filterwarnings('ignore')
import data
import os
import numpy as np
np.set_printoptions(suppress=True)
import torch
torch.set_printoptions(sci_mode=False)
import argparse
import ast

import General_Utils.path_utils as p_utils
import Models.model_builder as model_builder
import General_Utils.read_yaml as read_yaml
import Training.train_builder as train_builder
import Training.seg_train as seg_train
import Evaluation.seg_eval as seg_eval


model_configs_outer_path = os.path.join(os.environ["cwd"], "Model_Configs")
train_configs_outer_path = os.path.join(os.environ["cwd"], "Train_Configs")


def create_artifact_paths(model_dir, with_delete,
                          outer_path=data.cwd):
    # trained models
    trained_models_outer_fp = os.path.join(outer_path, "Trained_Models")
    trained_models_fp = os.path.join(outer_path, "Trained_Models", model_dir)
    p_utils.create_if_not_exists(trained_models_outer_fp, with_delete=False)
    p_utils.create_if_not_exists(trained_models_fp, with_delete=with_delete)

    # training logs
    training_logs_outer_fp = os.path.join(outer_path, "Trainings_Logs")
    training_logs_fp = os.path.join(outer_path, "Trainings_Logs", model_dir)
    p_utils.create_if_not_exists(training_logs_outer_fp, with_delete=False)
    p_utils.create_if_not_exists(training_logs_fp, with_delete=with_delete)

    # eval results
    outer_eval_results_fp = os.path.join(outer_path, "Eval_Results")
    eval_results_fp = os.path.join(outer_path, "Eval_Results", model_dir)
    p_utils.create_if_not_exists(outer_eval_results_fp, with_delete=False)
    p_utils.create_if_not_exists(eval_results_fp, with_delete=with_delete)
    return {"trained_models_fp": trained_models_fp,
            "training_logs_fp": training_logs_fp,
            "eval_results_fp": eval_results_fp}


def train_and_test(train_d: dict) -> None:
    task = train_d["task"]
    if not train_d["only_test"]:
        if task == "human_seg":
            seg_train.train_launch(train_d)
        else:
            print(f"Task {task} not supported.")

    if train_d["with_test"]:
        if "gpu_override" in train_d:
            gpu_id = train_d["gpu_override"][0]
        else:
            gpu_id = 0
        seg_eval.seg_eval(train_d, is_val=False, dl=None, device_id=gpu_id)


def parse_indices(indices_str: str) -> list[int]:
    try:
        indices = ast.literal_eval(indices_str)
        if not isinstance(indices, list):
            raise argparse.ArgumentTypeError('Indices must be in list format, e.g., [1, 2, 3]')
        if not all(isinstance(idx, int) for idx in indices):
            raise argparse.ArgumentTypeError('Indices must be integers')
        return indices
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError('Invalid format for indices. Please use list format, e.g., [1, 2, 3]')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parsing arguments for the Pose Experiments run.py file.")
    parser.add_argument("model_config_name", type=str, help="Name of the model configuration")
    parser.add_argument("train_config_name", type=str, help="Name of the train configuration")
    parser.add_argument("--with_load", action="store_true", help="Include if you want to load a model")
    parser.add_argument("--only_test", action="store_true", help="Include if you only want to perform testing")
    parser.add_argument("--gpu_override", nargs='?', type=parse_indices, help='List of gpus in list format, '
                                                                              'e.g., [0,1,2], without whitespace, or surrounded'
                                                                              'by quotes.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_config_name = args.model_config_name
    train_config_name = args.train_config_name
    with_load = args.with_load
    only_test = args.only_test
    gpu_override = args.gpu_override

    # build model from config
    print("Building model from config.")
    model_cfg_path = os.path.join(model_configs_outer_path, model_config_name)
    assert os.path.exists(model_cfg_path), "No such model config exists"
    model_cfg = read_yaml.yml_to_dict(model_cfg_path)

    # create train_builder from config
    print("Loading the training config and dataloader...")
    train_d = {}
    train_config_path = os.path.join(train_configs_outer_path, train_config_name)
    assert os.path.exists(train_config_path), "No such train config path exists"
    train_config = read_yaml.yml_to_dict(train_config_path)
    model_builder.get_model_from_cfg(model_cfg, train_d)
    print("Model built from config.")

    overall_cfg = {**model_cfg, **train_config}

    # create artefact paths
    model_save_dir = model_config_name[:-4] + "_" + train_config_name[:-4]
    with_delete = not with_load
    train_d["save_fps"] = create_artifact_paths(model_dir=model_save_dir,
                                                with_delete=not with_load)

    # Loading and testing info
    train_builder.train_build_from_cfg(train_config, train_d)
    train_d["with_test"] = train_config["with_test"]
    train_d["model_name"] = model_config_name[:-4]  # remove .yml extension
    train_d["with_load"] = with_load
    train_d["only_test"] = only_test

    print("Finished loading the training config and dataloader.")
    if gpu_override:
        train_d["gpu_override"] = gpu_override
    train_and_test(train_d)
