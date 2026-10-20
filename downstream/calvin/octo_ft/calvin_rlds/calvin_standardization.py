"""Octo standardization transform for the CALVIN task_D_D RLDS dataset.

Modelled on ``octo.data.oxe.oxe_standardization_transforms:taco_dataset_transform``
(taco_play is the closest OXE dataset — same lab, same robot conventions, and it
is the action head Octo-base uses for CALVIN). Two deliberate differences:

1. **Proprio slicing.** taco_play uses ``robot_obs[:, :6]`` + ``robot_obs[:, 7:8]``,
   but CALVIN's ``robot_obs`` has a different layout: index 6 IS the gripper width
   (verified on a real frame: robot_obs[:8] =
   [x, y, z, rx, ry, rz, 0.0800, -0.4402] — 0.08 m is the open gripper, index 7 is
   a joint angle). So we take ``robot_obs[:, :7]`` = tcp_pos(3) + tcp_euler(3) +
   gripper_width(1).

2. **Action space / gripper convention.** We keep CALVIN's *native* ``rel_actions``
   rather than converting to taco's 0/1 gripper. CALVIN rel_actions are already
   normalised (1.0 == max_rel_pos == 2 cm, and == max_rel_orn == 0.05 rad; see
   calvin_env/robot/robot.py::relative_to_absolute), with gripper in {-1, +1}.

   Training in this space matters for evaluation: Octo un-normalises its output
   with the finetuning dataset's own statistics, so a model finetuned here emits
   actions already in CALVIN rel_actions space. The eval wrapper must therefore
   pass them through UNSCALED — i.e. the /CALVIN_MAX_REL_POS, /CALVIN_MAX_REL_ORN
   conversion required for the zero-shot taco_play head must be DISABLED for a
   finetuned checkpoint. Pair this with
   ``action_normalization_mask=[True]*6 + [False]`` so the gripper stays as +-1.
"""
from typing import Any, Dict

import tensorflow as tf


def calvin_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # CALVIN rel_actions: [dx, dy, dz, drx, dry, drz, gripper(+-1)] -- passthrough.
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)

    # tcp_pos(3) + tcp_euler(3) + gripper_width(1)
    trajectory["observation"]["proprio"] = tf.cast(
        trajectory["observation"]["robot_obs"][:, :7], tf.float32
    )
    return trajectory
