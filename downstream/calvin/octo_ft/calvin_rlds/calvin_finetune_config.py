"""Octo finetuning config for CALVIN task_D_D with goal-image emphasis.

Usage:
  python octo_repo/scripts/finetune.py \
      --config calvin_rlds/calvin_finetune_config.py:full,multimodal \
      --config.pretrained_path=hf://rail-berkeley/octo-base \
      --config.save_dir=<...>

Why "multimodal" (keep_image_prob=0.5) rather than "image_conditioned" (1.0):
`task_augment_strategy="delete_task_conditioning"` stochastically deletes one
conditioning modality per sample. At 0.5 the language is dropped on ~half the
samples, which forces the policy to actually read the goal image instead of
riding on the instruction -- the classifier-free-guidance idea. Unlike 1.0 it
keeps a language-conditioned branch alive, which gives (a) task disambiguation
when a goal image is visually ambiguous, (b) a CFG knob at inference, and
(c) a same-checkpoint ablation between conditioning modes.

`goal_relabeling_strategy="uniform"` samples the goal image from a random future
step within the episode, so the model sees goals at many horizons rather than
only the window end.
"""
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    FINETUNING_KWARGS = {
        "name": "calvin_dd",
        "data_dir": placeholder(str),
        "image_obs_keys": {"primary": "rgb_static", "wrist": "rgb_gripper"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # CALVIN's gripper is already +-1; never normalise it.
        "action_normalization_mask": [True, True, True, True, True, True, False],
        "standardize_fn": ModuleSpec.create(
            "calvin_rlds.calvin_standardization:calvin_dataset_transform",
        ),
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    else:
        frozen_keys = ("octo_transformer.*", "heads_*.map_head.probe",
                       "heads_*.map_head.MultiHeadDotProductAttention_0.*")

    if task == "image_conditioned":
        goal_relabeling_strategy, keep_image_prob = "uniform", 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy, keep_image_prob = None, 0.0
    else:
        goal_relabeling_strategy, keep_image_prob = "uniform", 0.5

    # 1154 train episodes x ~59 steps ~= 68k transitions. The paper's run used
    # batch 32 for 30k steps (~14 epochs), ~4 h on one 48 GB GPU. The step count
    # matters: under the cosine schedule the step-15k checkpoint solved 0/40 of
    # the first test rows with the ground-truth goal, the final one 30/40.
    max_steps = FieldReference(30000)
    # window_size MUST match what the eval wrapper feeds at rollout time:
    # OctoModelWrapper uses history_window=2 (and pred_horizon=4, which is why
    # action_horizon below is 4). Octo's stock finetune_config ships
    # window_size=1 for its toy example; training with 1 here and evaluating
    # with 2 would silently change the model's temporal context between train
    # and test.
    window_size = FieldReference(default=2)

    return ConfigDict(dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        # The paper's run: 32 on one 48 GB GPU (128 runs out of memory). With
        # several GPUs Octo shards the batch, so it must be divisible by their number.
        batch_size=32,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        # Disabled: the eval block runs ValidationCallback AND VisualizationCallback,
        # and the visualiser OOMs on a 48 GB card (tries ~30 GB even at
        # eval_batch_size=64). It also killed a run at step 2499 via a matplotlib
        # incompatibility. Checkpoint saving is independent of this (finetune.py
        # line 406 vs 390), and checkpoints are selected by ORACLE SUCCESS RATE
        # rather than validation loss, so nothing needed is lost. Train loss is
        # still logged every log_interval steps.
        eval_interval=10**9,
        save_interval=2500,   # pick the checkpoint by closed-loop success, not by loss
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(project="octo_calvin_finetune", group=placeholder(str),
                   entity=placeholder(str)),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(name="cosine", init_value=0.0, peak_value=3e-4,
                               warmup_steps=1000, decay_steps=max_steps,
                               end_value=0.0),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(val_shuffle_buffer_size=1000, num_val_batches=16),
        viz_kwargs=dict(eval_batch_size=128, trajs_for_metrics=100,
                        trajs_for_viz=8, samples_per_state=8),
        frame_transform_threads=16,
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256), "wrist": (128, 128)},
            image_augment_kwargs=dict(
                primary=dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=["random_resized_crop", "random_brightness",
                                   "random_contrast", "random_saturation",
                                   "random_hue"],
                ),
                wrist=dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=["random_brightness", "random_contrast",
                                   "random_saturation", "random_hue"],
                ),
            ),
        ),
        traj_transform_kwargs=dict(
            window_size=window_size,
            action_horizon=4,
            goal_relabeling_strategy=goal_relabeling_strategy,
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(keep_image_prob=keep_image_prob),
        ),
    ))
