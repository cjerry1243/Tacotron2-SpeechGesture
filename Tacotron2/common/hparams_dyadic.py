class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create spk_embedder hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "iters_per_checkpoint": 1000,
        "seed": 16807,
        "fp16_run": False,
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "output_directory": "dyadic",  # Directory to save checkpoints.
        "log_directory": 'log',
        "checkpoint_path": '',
        "warm_start": False,
        "n_gpus": 1,  # Number of GPUs
        "device": 0,


        ################################
        # Model Parameters             #
        ################################
        "n_acoustic_feat_dims": 78,
        # "n_symbols": 387,
        # "n_symbols": 427,
        "n_symbols": 932, # 427 * 2 + 78
        # "n_symbols": 5816,
        "symbols_embedding_dim": 512,

        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 512,
        "spk_embedding_dim": 0,

        # Decoder parameters
        "decoder_rnn_dim": 1024,
        "prenet_dim": 256,
        "max_decoder_steps": 8000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,

        # Attention parameters
        "attention_rnn_dim": 1024,
        "attention_dim": 128,
        # +- time steps to look at when computing the attention. Set to None
        # to block it.
        "attention_window_size": 30,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "batch_size": 64,
        "mask_padding": True,  # set spk_embedder's padded outputs to padded values
        "mel_weight": 1,
        "gate_weight": 0,
        "vel_weight": 1,
        "pos_weight": 0.01,
        "add_l1_losss": False
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view


