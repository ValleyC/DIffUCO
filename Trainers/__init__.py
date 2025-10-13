from .PPO_Trainer import PPO
from .ForwardKL_Trainer import ForwardKL

### SDDS Trainers: PPO (rKL w/ RL) and ForwardKL (fKL w/ MC)
Trainer_registry = {"PPO": PPO, "Forward_KL": ForwardKL, "REINFORCE": PPO}  # REINFORCE redirects to PPO for compatibility


def get_Trainer_class(config):
    train_mode_str = config["train_mode"]
    if train_mode_str == "REINFORCE":
        print("WARNING: REINFORCE mode is not available in SDDS. Using PPO instead.")
        train_mode_str = "PPO"

    if(train_mode_str in Trainer_registry.keys()):
        noise_class = Trainer_registry[train_mode_str]
    else:
        raise ValueError(f"Train mode {train_mode_str} is not implemented. Available: PPO, Forward_KL")
    return noise_class
