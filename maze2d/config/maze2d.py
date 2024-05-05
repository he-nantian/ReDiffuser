import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'seed': 0,
        'batch_size': 8,
        'plan_num': 1000,
        'device': 'cuda',

        ## diffusion model
        'horizon': 256,
        'variable_horizon': None,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': True,
        'ActionBased': False,
        'output_dim': 2,
        'RND': True,
        'NICE': True,
        'NTC': True,
        'load_best': True,
        'lmbda': 1.0,
        'input_discount': 1.0,
        'scale_u': 5.0,
        'clip': 8,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

    'robuster': {
        'device': 'cuda',
        'ActionBased': False,
        'input_discount': 1.0,
        'output_dim': 2,
        "LoadDataset": True,
        'dataset_episodic_size': 64e3,
        'batch_size': 256,
        'n_epochs': 20,
        'learning_rate': 4e-4,
        'generate_batch_size': 32,  # We compute input mean and standard variance in dataset generation process
        'clip': 8,
        'n_plot': 1,
        'plot_batch_size': 4,
        
        ## pretrain RND
        "LoadPretrainDataset": True,
        'LoadModel': False,
        'base_pretrain_dataset_size': 1e7,
        'pretrain_generate_batch_size': 1e4,
        'pretrain_batch_size': 256,
        'pretrain_n_epochs': 40,
        'pretrain_gap': 1.0,

        ## diffusion model
        'horizon': 256,
        'variable_horizon': None,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',
        
        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
        'logbase': 'logs',
        'horizon': 256,
        'n_diffusion_steps': 256,
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'plan': {
        'horizon': 128,
        'variable_horizon': None,
        'n_diffusion_steps': 64,
        'clip': 4,
    },
    'robuster': {
        'horizon': 128,
        'variable_horizon': None,
        'n_diffusion_steps': 64,
        'clip': 4,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'plan': {
        'horizon': 384,
        'variable_horizon': None,
        'n_diffusion_steps': 256,
        'clip': 12,
    },
    'robuster': {
        'horizon': 384,
        'variable_horizon': None,
        'n_diffusion_steps': 256,
        'clip': 12,
    },
}
