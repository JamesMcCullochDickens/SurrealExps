import Dataloaders.surreal_human_seg_dl as sh_seg_dl
import Training.optimizers as optimizers
import Training.lr_schedulers as lr_schedulers
import Training.loss_functions as loss_functions


def get_dataset(cfg: dict, dl_kwargs: dict, train_d: dict) -> None:
    dataset_name = cfg["dataset_name"]
    if dataset_name == "Surreal_Human_Segmentation":
        with_masking = cfg.get("with_rgb_masking", False)
        with_bb_cropping = cfg.get("with_bb_cropping", False)
        interp_size = cfg.get("interp_size", None)
        load_to_RAM = cfg.get("load_to_ram", False)
        dl_kwargs["num_workers"] = 1 if load_to_RAM else dl_kwargs["num_workers"]
        dataset_args = {"with_rgb_masking": with_masking, "with_bb_cropping": with_bb_cropping,
                        "interp_size": interp_size, "load_to_RAM": load_to_RAM}
        train_dataset_args, val_dataset_args, test_dataset_args = (dataset_args.copy(), dataset_args.copy(),
                                                                   dataset_args.copy())
        val_dl_kwargs = dl_kwargs.copy()
        test_dl_kwargs = dl_kwargs.copy()

        # training and validation dataloaders
        train_dataset_args["split"] = "train"
        test_dataset_args["split"] = "test"
        val_dataset_args["split"] = "val"
        dl_kwargs["shuffle"] = True
        dl_kwargs["drop_last"] = True
        val_dl_kwargs["shuffle"] = False
        val_dl_kwargs["drop_last"] = False
        val_dl_kwargs["batch_size"] = 32
        val_dl_kwargs["num_workers"] = 8
        train_d["train_dl"] = sh_seg_dl.get_surreal_human_seg_dl(dl_kwargs, train_dataset_args)
        train_d["val_dl"] = sh_seg_dl.get_surreal_human_seg_dl(val_dl_kwargs, val_dataset_args)

        # test dataloader
        test_dl_kwargs["shuffle"] = False
        test_dl_kwargs["drop_last"] = False
        test_dl_kwargs["batch_size"] = 1
        test_dl_kwargs["num_workers"] = 8
        test_dl_kwargs["persistent_workers"] = False
        test_dataset_args["interp_size"] = None
        train_d["test_dl"] = sh_seg_dl.get_surreal_human_seg_dl(test_dl_kwargs, test_dataset_args)

        # class mapping
        train_d["class_mapping_str_to_num"] = sh_seg_dl.class_mapping_str_to_num
        train_d["class_mapping_num_to_str"] = sh_seg_dl.class_mapping_num_to_str

        # class weighting
        train_d["class_weights"] = sh_seg_dl.get_class_imbalance_ratios()
        if train_d["with_16_bit_precision"]:
            train_d["class_weights"] = train_d["class_weights"].half()
    else:
        print(f"Dataset {dataset_name} not supported.")

    train_d["dataset_name"] = dataset_name


def get_optimizer(cfg: dict, train_d: dict) -> None:
    optimizer_name = cfg["optimizer_type"]
    if optimizer_name == "sgd":
        base_lr = cfg["base_lr"]
        weight_decay = cfg.get("weight_decay", 0.00005)
        nesterov = cfg.get("nesterov", True)
        momentum = cfg.get("momentum", 0.90)
        optim = optimizers.get_sgd_optim(model=train_d["model"], base_lr=base_lr, weight_decay=weight_decay,
                                     nesterov=nesterov, momentum=momentum)
    elif optimizer_name == "adamw":
        base_lr = cfg["base_lr"]
        weight_decay = cfg.get("weight_decay", 0.00005)
        optim = optimizers.get_adamw_optim(train_d["model"], base_lr=base_lr, weight_decay=weight_decay)
    else:
        print(f"Optimizer {optimizer_name} is not supported.")
        exit(-1)
    train_d["optimizer"] = optim


def get_lrs(cfg: dict, train_d: dict) -> None:
    if cfg["lrs_scheduler"] == "ms_lrs":
        gamma_val = cfg["gamma_val"]
        time_interval = cfg["time_interval"]
        num_epochs = cfg["num_epochs"]
        lr_scheduler = lr_schedulers.get_k_step_lrs(optimizer=train_d["optimizer"], num_epochs=num_epochs,
                                          gamma_val=gamma_val, interval=time_interval)
    elif cfg["lrs_scheduler"] == "cosine_annealing":
        num_epochs = cfg["num_epochs"]
        min_learning_rate = cfg["min_learning_rate"]
        lr_scheduler = lr_schedulers.get_cos_annealing_lrs(optimizer=train_d["optimizer"], num_epochs=num_epochs,
                                                 min_learning_rate=min_learning_rate)
    train_d["lr_scheduler"] = lr_scheduler


def get_loss_function(cfg: dict, train_d: dict) -> None:
    lf_name = cfg["loss_fn"]
    if lf_name == "cross_entropy":
        label_smoothing = cfg.get("label_smoothing", 0.0)
        ignore_index = cfg.get("ignore_index", -100)
        with_ce_weights = cfg.get("with_ce_weights", False)
        if with_ce_weights:
            ce_weights = train_d["class_weights"]
        else:
            ce_weights = None
        loss_fn = loss_functions.get_ce_loss(label_smoothing, ignore_index, ce_weights)
    elif lf_name == "sampled_cross_entropy":
        samples_per_im = cfg.get("num_samples", 5000)
        loss_fn = loss_functions.sampled_ce_loss(samples_per_im)
    elif lf_name == "dice_loss":
        smooth = cfg.get("smooth", 1.0)
        loss_fn = loss_functions.get_dice_loss(smooth)
    else:
        print(f"Loss function {lf_name} not supported.")
        exit(-1)
    train_d["loss_fn"] = loss_fn


def train_build_from_cfg(cfg: dict, train_d: dict) -> None:
    # dataloader arguments
    batch_size = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 0)
    persistent_workers = num_workers != 0
    pin_memory = cfg.get("pin_memory", False)
    dl_kwargs = {"batch_size": batch_size, "num_workers": num_workers,
                 "persistent_workers": persistent_workers, "pin_memory": pin_memory}
    train_d["batch_size"] = batch_size

    # training info
    train_d["with_ddp"] = cfg.get("with_ddp", False)
    train_d["num_epochs"] = cfg["num_epochs"]
    train_d["early_stopping_val"] = cfg.get("early_stopping_val", None)
    train_d["with_validation"] = cfg.get("with_validation", False)
    train_d["with_train_accuracy"] = cfg.get("with_train_accuracy", False)
    train_d["val_sampling_mod"] = cfg.get("test_sampling_mod", 1)
    train_d["starting_val_epoch"] = cfg.get("starting_val_epoch", 0)
    train_d["with_16_bit_precision"] = cfg.get("with_16_bit_precision", False)

    # testing info
    train_d["test_background"] = cfg.get("test_background", False)

    # dataset
    get_dataset(cfg, dl_kwargs, train_d)

    # optimizer and lr scheduler
    get_optimizer(cfg, train_d)
    get_lrs(cfg, train_d)

    # loss function
    get_loss_function(cfg, train_d)


