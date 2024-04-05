import warnings
warnings.filterwarnings('ignore')
import data
import os
import General_Utils.path_utils as p_utils
import Models.model_builder as model_builder
import General_Utils.read_yaml as read_yaml
import Training.train_builder as train_builder
import Training.seg_train as seg_train

model_configs_outer_path = os.path.join(os.getcwd(), "Configs/Model_Configs")
train_configs_outer_path = os.path.join(os.getcwd(), "Configs/Train_Configs")


def create_artifact_paths(model_dir, with_delete,
                          outer_path=os["cwd"]):
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


def train_and_test(train_d:dict) -> None:
    task = train_d["task"]
    if train_d["task"] == "human_seg":
        seg_train.train_launch(train_d)
    else:
        print(f"Task {task} not supported.")

    if train_d["with_test"]:
        exit()  # TODO add evaluation code


if __name__ == "__main__":
    model_config_name = "LRASPP_rgb_seg.yml"
    train_config_name = "human_seg_rgb_v1.yml"
    with_load = False
    only_test = False

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
    train_d["model"] = model_builder.from_config(model_cfg, train_d)
    print("Model built from config.")

    overall_cfg = {**model_cfg, **train_config}

    # create artefact paths
    model_save_dir = model_config_name[:-4] + "_" + train_config_name[:-4]
    with_delete = not with_load
    if with_delete:
        inp = input("Are you sure you want to delete the previous "
                    "directories (if they exist). Enter Y for yes, anything otherwise")
        if inp not in ["Y", "yes", "y", "Yes"]:
            exit()
    train_d["save_fps"] = create_artifact_paths(model_dir=model_save_dir, with_delete=not with_load)


    # Loading and testing info
    train_d = train_builder.train_build_from_cfg(train_config, train_d)
    train_d["with_test"] = train_config["with_test"]
    train_d["model_name"] = model_config_name[:-4] # remove .yml extension
    train_d["with_load"] = with_load
    train_d["only_test"] = only_test

    print("Finished loading the training config and dataloader.")
    train_and_test(train_d)
