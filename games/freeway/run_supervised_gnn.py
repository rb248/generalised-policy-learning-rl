import cProfile
import pstats
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from games.freeway.freeway_envs.freeway_env import FreewayEnv
from games.model.policy import CustomCNN, CustomHeteroGNN, CustomMLPExtractor
import pygame
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model at {x[-1]} timesteps")
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                #wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
            else:
                device = "cpu"
                if self.verbose > 0:
                    print("No data available for logging.")
                #wandb.log({"timesteps": self.num_timesteps})
        return True 

log_dir = "./logs/Freeway-GNN-training/"

def make_env(lanes, max_cars, car_speed, seed=0, rank=None):
    def _init():
        env = FreewayEnv(render_mode='human', observation_type='graph')
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
        env = Monitor(env, filename=monitor_path, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    return _init

envs = DummyVecEnv([make_env([50, 80, 120], 10, 2, rank=i) for i in range(1)])

policy_kwargs = dict(
    features_extractor_class=CustomHeteroGNN,
    features_extractor_kwargs=dict(
        features_dim=64,
        hidden_size=64,
        num_layer=10,
        obj_type_id='obj',
        #arity_dict={'ChickenOnLane': 2, 'CarOnLane': 2, 'LaneNextToLane': 2, 'PlayerNearCar': 2}
        arity_dict ={'atom':2},
        game='freeway',
    ),
    net_arch=dict(pi=[64, 64], vf=[128, 128]),  # Specify the network architecture for policy and value function
    activation_fn=torch.nn.ReLU
)

# Create the PPO model with the custom feature extractor and network architecture
model = PPO("MlpPolicy", envs, policy_kwargs=policy_kwargs, verbose=2)
save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
# Initialize profiler
profiler = cProfile.Profile()

# Start profiling
profiler.enable()

# Train the model
model.learn(total_timesteps=1000000, callback=[save_callback])

# Stop profiling
profiler.disable()

# Save the model
model.save("ppo_custom_heterognn")

# Create a stats object and sort by total time spent in the function
stats = pstats.Stats(profiler).sort_stats('cumtime')

# Print the 10 most time-consuming functions
# stats.print_stats()
