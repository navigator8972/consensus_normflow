"""
Common utilities for training.
grabbed from dedo
"""
from datetime import datetime
import numpy as np

import os
import platform
import torch
import wandb


def object_to_str(obj):
    # Print all fields of the given object as text in tensorboard.
    text_str = ''
    for member in vars(obj):
        # Tensorboard uses markdown-like formatting, hence '  \n'.
        text_str += '  \n{:s}={:s}'.format(
            str(member), str(getattr(obj, member)))
    return text_str


def init_train(algo, args, tags=None):
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    if platform.system() == 'Linux':
        os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'
    logdir = None
    if args.logdir is not None:
        tstamp = datetime.strftime(datetime.today(), '%y%m%d_%H%M%S')
        lst = [algo, tstamp, args.env]
        subdir = '_'.join(lst)
        logdir = os.path.join(os.path.expanduser(args.logdir), subdir)
        if args.use_wandb:
            wandb.init(config=vars(args), project='consensus_normflow',
                       name=logdir, tags=tags)
            wandb.init(sync_tensorboard=False)
            try:  # patch only once, if more than one run, ignore error
                wandb.tensorboard.patch(tensorboardX=True, pytorch=True)
            except ValueError as e:
                pass
    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    return logdir, device


import pickle

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

#callback during training process, grabbed from dedo
class CustomCallback(BaseCallback):
    """
    A custom callback that runs eval and adds videos to Tensorboard.
    """

    def __init__(self, eval_env, logdir, num_train_envs, args,
                 num_steps_between_save=10000, viz=False, debug=False):
        super(CustomCallback, self).__init__(debug)
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
        self._eval_env = eval_env
        self._logdir = logdir
        self._num_train_envs = num_train_envs
        self._my_args = args
        self._num_steps_between_save = num_steps_between_save
        self._viz = viz
        self._debug = debug
        self._steps_since_save = num_steps_between_save  # save right away

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Log args to tensorboard.
        self.logger.record('args', object_to_str(self._my_args))

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self._steps_since_save += self._num_train_envs
        if self._steps_since_save >= self._num_steps_between_save:
            # Save checkpoint.
            if self._logdir is not None:
                self.model.save(os.path.join(self._logdir, 'agent'))
                pickle.dump(self._my_args,
                            open(os.path.join(self._logdir, 'args.pkl'), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
            self._steps_since_save = 0
            # Record video.
            if not self._my_args.disable_logging_video:
                screens = []

                def grab_screens(_locals, _globals):
                    screen = self._eval_env.render(
                        mode='rgb_array', width=300, height=300)
                    # PyTorch uses CxHxW vs HxWxC gym (and TF) images
                    screens.append(screen.transpose(2, 0, 1))

                evaluate_policy(
                    self.model, self._eval_env, callback=grab_screens,
                    n_eval_episodes=1, deterministic=False)
                self.logger.record(
                    'trajectory/video',
                    Video(torch.ByteTensor([screens]), fps=50),
                    exclude=('stdout', 'log', 'json', 'csv'))

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
