import numpy as np
import copy
import pickle
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, log_dir: str, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.log_dir = log_dir

    def _init_callback(self) -> None:
        print("get log dir:", self.log_dir)
        return super()._init_callback()

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_start_time = time.time()
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print("!!!!!!!!!!!!!!!start another rollout!!!!!!!!")
        env = self.training_env.envs[0]
        env.epoch_trajectory.clear()
        
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        env = self.training_env.envs[0]
        
        self.logger.record(env.prefix + "reward", env.reward)
        self.logger.record(env.prefix + "step_executing_time", env.step_executing_time) 
        self.logger.record(env.prefix + "safety_violation_times", env.break_safety)
        
        # print(a.reward)
        # Retrieve training reward
        # print("callback after step")
        # x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        # results = load_results(self.log_dir)
        # if len(x) > 0:
        #     last_line = results.tail(1)
        #     last_success_parking_times = last_line["success_parking_times"]
        #     mean_reward = np.mean(y[-100:])
        #     print(f"Num timesteps: {self.num_timesteps}")
        #     print(f"Last mean reward per episode: {mean_reward:.2f}")
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        env = self.training_env.envs[0]
        env.training_trajectories.append( np.hstack(env.epoch_trajectory) )
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        
        
        env = self.training_env.envs[0]
        
        self.training_end_time = time.time()
        self.logger.record(env.prefix + "training_total_time", self.training_end_time - self.training_start_time)

        trajectory_save_dir = self.log_dir + "/" + env.prefix +  "trajectory.pickle"
        with open(trajectory_save_dir, 'wb') as f:
            pickle.dump(env.training_trajectories, f, pickle.HIGHEST_PROTOCOL)
        
        pass