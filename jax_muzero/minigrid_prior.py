import os


from ray import tune


from algorithms.muzero import Experiment, Experiment_Minigrid

import wandb

if __name__ == '__main__':
    config = {
        'env_id': 'MiniGrid-Empty-6x6-v0',
        'env_kwargs': {},
        'seed': 42,
        'num_envs': 1,
        'unroll_steps': 5,
        'td_steps': 5,
        'max_search_depth': None,

        'num_bins': 601,
        'channels': 64,
        'use_resnet_v2': True,
        'output_init_scale': 0.,
        'discount_factor': 0.997 ** 4,
        'mcts_c1': 0.8,
        'mcts_c2': 3.6,
        'alpha': 0.3,
        'exploration_prob': 0.25,
        'temperature_scheduling': 'staircase',
        'q_normalize_epsilon': 0.01,
        'child_select_epsilon': 1E-6,
        'num_simulations': 100,

        'replay_min_size': 2_000,
        'replay_max_size': 100_000,
        'batch_size': 256,

        'value_coef': 0.25,
        'policy_coef': 1.,
        'max_grad_norm': 5.,
        'learning_rate': 7E-4,
        'warmup_steps': 1_000,
        'learning_rate_decay': 0.1,
        'weight_decay': 1E-4,
        'target_update_interval': 200,

        'evaluate_episodes': 32,
        'log_interval': 100,
        'total_frames': 10_000,
        
        'mcts_c3': 1.2,
        'mode': "prior",
        'save_video': True,
        'video_folder': '/tmp/minigrid_video',
        'wandb': {'project': 'MCTS-Experiments', 'entity': 'dtiapkin', 'name': 'Bernstein MuZero', 'sync_tensorboard': True}
    }
    log_filename = os.path.basename(__file__).split('.')[0] + "_prior"
    analysis = tune.run(
        Experiment_Minigrid,
        name=log_filename,
        config=config,
        stop={
            'num_updates': 12_000,
        },
        resources_per_trial={
            'gpu': 1,
        },
    )
