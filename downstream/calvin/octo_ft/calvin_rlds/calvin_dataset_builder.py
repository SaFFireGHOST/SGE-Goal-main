"""TFDS/RLDS builder for a task-stratified slice of CALVIN task_D_D training.

Octo's `finetune.py` trains from RLDS via `make_single_dataset`, but CALVIN ships
as ~512k loose .npz frames. This builder turns the language-annotated windows
selected by `select_frames.py` into RLDS episodes.

One episode == one annotated window (mean ~59 frames), so each episode is a
coherent trajectory with a single language instruction. That matters for Octo's
`goal_relabeling_strategy="uniform"`, which samples the goal image from a random
*future* step inside the episode: windows give it a meaningful horizon to sample
from, whereas arbitrary frame chunks would not.

Build with build_rlds.py (see the README).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow_datasets as tfds

ROOT = os.path.abspath(os.environ.get("CALVIN_ROOT", "."))   # same root as data/*.py
TRAIN_DIR = os.path.join(ROOT, "data", "task_D_D", "training")
SELECTION = os.path.join(ROOT, "calvin_train_selection.json")


def _load_frame(path):
    """Read one CALVIN frame into plain numpy arrays (closes the file handle)."""
    with np.load(path, allow_pickle=True) as z:
        return {
            "rgb_static": z["rgb_static"].astype(np.uint8),
            "rgb_gripper": z["rgb_gripper"].astype(np.uint8),
            "robot_obs": z["robot_obs"].astype(np.float32),
            "rel_actions": z["rel_actions"].astype(np.float32),
        }


class CalvinDD(tfds.core.GeneratorBasedBuilder):
    """CALVIN task_D_D, language-annotated windows, in RLDS form."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Task-stratified slice of task_D_D training."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "rgb_static": tfds.features.Image(
                            shape=(200, 200, 3), dtype=np.uint8,
                            doc="Third-person camera -> Octo image_primary."),
                        "rgb_gripper": tfds.features.Image(
                            shape=(84, 84, 3), dtype=np.uint8,
                            doc="Wrist camera -> Octo image_wrist."),
                        "robot_obs": tfds.features.Tensor(
                            shape=(15,), dtype=np.float32,
                            doc="CALVIN robot_obs; sliced to 7-dim proprio "
                                "by calvin_dataset_transform."),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc="CALVIN rel_actions: normalised d-xyz/d-rpy plus "
                            "gripper in {-1,+1}."),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "task": tfds.features.Text(),
                    "start_frame": tfds.features.Scalar(dtype=np.int64),
                    "end_frame": tfds.features.Scalar(dtype=np.int64),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        sel = json.load(open(SELECTION))
        windows = sel["windows"]
        # deterministic 95/5 split; val is only used for Octo's periodic metrics
        cut = max(1, int(0.05 * len(windows)))
        return {
            "train": self._generate_examples(windows[cut:]),
            "val": self._generate_examples(windows[:cut]),
        }

    def _generate_examples(self, windows):
        # Fail loudly rather than silently shrinking the dataset. Running this
        # while the frame download was still in flight quietly dropped every
        # window that happened to be incomplete when it was reached, producing a
        # dataset that looked fine but was missing data.
        missing = []
        for w in windows:
            a, b = int(w["start"]), int(w["end"])
            if not all(os.path.exists(os.path.join(TRAIN_DIR, f"episode_{i:07d}.npz"))
                       for i in range(a, b + 1)):
                missing.append((w["task"], a, b))
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(windows)} windows have missing frames "
                f"(e.g. {missing[:3]}). Finish the download before building, or "
                f"re-run select_frames.py against what is actually on disk."
            )

        for w in windows:
            a, b = int(w["start"]), int(w["end"])
            paths = [os.path.join(TRAIN_DIR, f"episode_{i:07d}.npz")
                     for i in range(a, b + 1)]
            # Frames live on NFS and each episode is ~60 separate .npz reads;
            # sequentially that measured ~14 s/episode (~5 h for the full build).
            # The work is I/O-bound and np.load releases the GIL, so read the
            # episode's frames in parallel.
            with ThreadPoolExecutor(max_workers=16) as pool:
                frames = list(pool.map(_load_frame, paths))

            steps = []
            n = len(frames)
            for k, z in enumerate(frames):
                steps.append({
                    "observation": {
                        "rgb_static": z["rgb_static"],
                        "rgb_gripper": z["rgb_gripper"],
                        "robot_obs": z["robot_obs"],
                    },
                    "action": z["rel_actions"],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if k == n - 1 else 0.0),
                    "is_first": bool(k == 0),
                    "is_last": bool(k == n - 1),
                    "is_terminal": bool(k == n - 1),
                    "language_instruction": w["instruction"],
                })
            yield f"{a}_{b}", {
                "steps": steps,
                "episode_metadata": {
                    "task": w["task"], "start_frame": a, "end_frame": b,
                },
            }
