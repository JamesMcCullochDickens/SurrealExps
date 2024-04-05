import Models.torchvision_models as tv_models


def get_model_from_cfg(cfg: dict, train_d: dict) -> None:
    task = cfg["task"]
    if task == "human_seg":
        model_name = cfg["model_name"]
        num_classes = cfg["num_classes"]
        in_channels = cfg["in_channels"]
        if model_name == "Deeplab_v3":
            num_layers = cfg["num_layers"]
            model = tv_models.get_pretrained_dlv3(num_classes, in_channels, num_layers)
        elif model_name == "LRASPP":
            model = tv_models.get_pretrained_LRASPP(num_classes, in_channels)
        else:
            print(f"Model Name {model_name} not supported.")
    else:
        print(f"Task {task} not supported.")
    train_d["model"] = model
    train_d["task"] = task
    train_d["model_name"] = model_name

