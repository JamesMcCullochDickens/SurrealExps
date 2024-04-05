import torch


def get_k_step_lrs(optimizer: torch.optim.Optimizer, num_epochs: int,
                   interval: int, gamma_val: float) -> torch.optim.lr_scheduler:
    time_steps = list(range(interval, num_epochs, interval))
    return (torch.optim.lr_scheduler.
            MultiStepLR(optimizer=optimizer, milestones=time_steps, gamma=gamma_val))


def get_cos_annealing_lrs(optimizer: torch.optim.Optimizer, num_epochs: int,
                          min_learning_rate=0.0) -> torch.optim.lr_scheduler:
    lrs = (torch.optim.lr_scheduler.
           CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_learning_rate))
    return lrs
