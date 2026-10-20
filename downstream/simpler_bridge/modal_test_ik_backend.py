"""
Integration test: standalone-pinocchio backend driving a real SIMPLER rollout.

Validates (in-process; standalone pinocchio does not segfault):
  1. IK self-consistency: FK(q)->pose ; IK(pose)->q' ; FK(q')->pose' ~= pose.
  2. env.step() runs through the patched controller WITHOUT segfault.
  3. Sane kinematics: zero action keeps EE ~still; +x world_vector moves EE +x.

Run:  modal run simpler-eval/modal_test_ik_backend.py
"""
import modal

app = modal.App("simpler-ik-test")

CUDA_TAG = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install("software-properties-common", "gpg-agent", "ca-certificates")
    .run_commands(
        "add-apt-repository -y ppa:kisak/kisak-mesa",
        "apt-get update",
    )
    .apt_install(
        "git", "wget", "bzip2", "libglib2.0-0", "libgl1", "libglx-mesa0",
        "libvulkan1", "vulkan-tools", "libvulkan-dev",
        "libegl1", "libgles2", "libglvnd-dev", "libosmesa6-dev", "ffmpeg",
        "build-essential", "g++", "cmake",
        "mesa-vulkan-drivers", "libgl1-mesa-dri",
        "libx11-6", "libxext6", "libxrender1", "libxi6", "libxfixes3",
        "libxxf86vm1", "libxrandr2", "libxinerama1", "libxcursor1",
        "libxcb1", "libxau6", "libxdmcp6", "libsm6", "libice6", "libxkbcommon0",
    )
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands("python -m pip install --upgrade pip 'setuptools<70' wheel")
    .pip_install("numpy==1.24.4", "scipy", "pillow", "opencv-python-headless", "mediapy")
    .pip_install("sapien==2.2.2")
    .pip_install("ruckig")
    .run_commands(
        "git clone https://github.com/simpler-env/SimplerEnv /root/SimplerEnv --recurse-submodules",
        "cd /root/SimplerEnv/ManiSkill2_real2sim && python -m pip install -e .",
        "cd /root/SimplerEnv && python -m pip install -e . --no-deps",
    )
    .pip_install("pin")
    .env({
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": "",
    })
    .add_local_python_source("pinocchio_ik_backend")
)


@app.function(image=image, timeout=1200)
def test_backend():
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})

    from pinocchio_ik_backend import patch_pd_ee_controllers, StandalonePinocchioModel
    urdf = patch_pd_ee_controllers()

    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    obs, _ = env.reset()
    robot = env.unwrapped.agent.robot
    out = {"urdf": urdf}

    # --- 1) IK self-consistency (standalone) ---------------------------------
    pm = StandalonePinocchioModel(robot, urdf)
    link_names = [l.get_name() for l in robot.get_links()]
    ee_idx = link_names.index("ee_gripper_link")
    qmask = np.zeros(robot.dof, dtype=bool)
    qmask[:6] = True  # arm joints active
    ik_report = []
    for qt in [np.array([0.1, -0.3, 0.2, 0.0, 0.4, -0.1, 0.015, 0.015]),
               np.array([-0.4, 0.5, -0.2, 0.3, -0.3, 0.6, 0.02, 0.02])]:
        pm.compute_forward_kinematics(qt)
        target = pm.get_link_pose(ee_idx)
        init = np.zeros(robot.dof)
        sol, ok, err = pm.compute_inverse_kinematics(ee_idx, target, init, qmask, 200)
        pm.compute_forward_kinematics(sol)
        reached = pm.get_link_pose(ee_idx)
        dp = float(np.linalg.norm(np.array(list(target.p)) - np.array(list(reached.p))))
        ik_report.append({"ik_success": bool(ok), "ik_err": err, "reached_pos_diff": dp})
    out["ik_selfcheck"] = ik_report

    # --- 1b) qpos reconstruction self-check (vs set_qpos ground truth) --------
    pm._setup_reconstruction()
    recon_report = []
    for qk in [np.array([0.1, -0.3, 0.2, 0.0, 0.4, -0.1, 0.015, 0.015]),
               np.array([-0.4, 0.5, -0.2, 0.3, -0.3, 0.6, 0.02, 0.02])]:
        robot.set_qpos(qk)
        qr = pm.reconstruct_qpos(robot)
        # arm joints (first 6) are what matter for control
        err_arm = float(np.linalg.norm(qr[:6] - qk[:6]))
        recon_report.append({"set": qk[:6].round(3).tolist(),
                             "recon": qr[:6].round(3).tolist(),
                             "arm_err": round(err_arm, 5)})
    out["recon_selfcheck"] = recon_report

    # --- 2)+3) rollout through the patched controller ------------------------
    def ee_world():
        ln = [l.get_name() for l in robot.get_links()]
        return np.array(list(robot.get_links()[ln.index("ee_gripper_link")].get_pose().p))

    traj = {"zero": [], "plus_x": []}
    try:
        ee0 = ee_world().tolist()
        # 15 zero-delta steps -> EE should stay roughly still
        for _ in range(15):
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, rew, term, trunc, info = env.step(a)
            traj["zero"].append(ee_world().tolist())
        # 15 steps commanding +x delta -> EE x should increase
        for _ in range(15):
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            a[0] = 0.02
            a[-1] = 1.0  # keep gripper open-ish
            obs, rew, term, trunc, info = env.step(a)
            traj["plus_x"].append(ee_world().tolist())
        out["status"] = "ROLLOUT OK (no segfault)"
        out["ee_start"] = ee0
        out["ee_after_zero"] = traj["zero"][-1]
        out["ee_after_plusx"] = traj["plus_x"][-1]
        out["zero_drift"] = float(np.linalg.norm(np.array(traj["zero"][-1]) - np.array(ee0)))
        out["plusx_dx"] = float(traj["plus_x"][-1][0] - traj["zero"][-1][0])
        out["info_keys"] = list(info.keys())
        out["success_flag"] = info.get("success", "MISSING")
        img = get_image_from_maniskill2_obs_dict(env, obs)
        out["img_shape"] = list(np.asarray(img).shape)
    except Exception:
        import traceback
        out["status"] = "ROLLOUT FAILED"
        out["trace"] = traceback.format_exc()[-2500:]
    return out


@app.function(image=image, timeout=900)
def diag_control():
    """Is get_qpos() reliable during a rollout? Compare pin_FK(get_qpos) to the
    live EE pose, and log the controller's IK inputs per step."""
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})

    from pinocchio_ik_backend import patch_pd_ee_controllers, StandalonePinocchioModel
    urdf = patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    robot = env.unwrapped.agent.robot
    pm = StandalonePinocchioModel(robot, urdf)
    link_names = [l.get_name() for l in robot.get_links()]
    ee_idx = link_names.index("ee_gripper_link")
    root_link = robot.get_links()[0]
    ee_link = robot.get_links()[ee_idx]

    def live_ee_at_base():
        return root_link.get_pose().inv() * ee_link.get_pose()

    def check(tag):
        q = np.asarray(robot.get_qpos(), dtype=float)
        pm.compute_forward_kinematics(q)
        fk = pm.get_link_pose(ee_idx)
        live = live_ee_at_base()
        dp = float(np.linalg.norm(np.array(list(fk.p)) - np.array(list(live.p))))
        return {"tag": tag, "get_qpos": [round(float(x), 4) for x in q],
                "fk_of_getqpos_p": [round(float(x), 4) for x in fk.p],
                "live_ee_at_base_p": [round(float(x), 4) for x in live.p],
                "fk_vs_live_posdiff": round(dp, 5)}

    log = [check("after_reset")]
    for i in range(4):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(a)
        log.append(check(f"after_zero_step_{i}"))
    return {"urdf": urdf, "log": log}


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(test_backend.remote(), indent=2, default=str))


@app.function(image=image, timeout=600)
def diag_drivetarget():
    """Is joint.get_drive_target() reliable (unlike the broken get_qpos array)?"""
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    robot = env.unwrapped.agent.robot
    aj = robot.get_active_joints()

    def dt():
        return [round(float(j.get_drive_target()), 4) for j in aj]

    def gq():
        return [round(float(x), 4) for x in np.asarray(robot.get_qpos(), float)]

    out = {"drive_target_after_reset": dt(), "get_qpos_after_reset": gq()}
    for i in range(3):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    out["drive_target_after_3zero"] = dt()
    out["get_qpos_after_3zero"] = gq()
    # command a clear +x; arm should move, drive targets should change
    for i in range(8):
        a = np.zeros(env.action_space.shape, dtype=np.float32); a[0] = 0.05; a[-1] = 1.0
        env.step(a)
    out["drive_target_after_8plusx"] = dt()
    return out


@app.local_entrypoint()
def diag():
    import json
    print(json.dumps(diag_control.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def test_response():
    """Does the EE follow commanded world_vector directions? (control responsiveness)"""
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    robot = env.unwrapped.agent.robot
    ln = [l.get_name() for l in robot.get_links()]
    ee_i = ln.index("ee_gripper_link")

    def ee():
        return np.array(list(robot.get_links()[ee_i].get_pose().p))

    out = {"action_space_shape": list(env.action_space.shape),
           "action_low": [round(float(x), 3) for x in np.atleast_1d(env.action_space.low)[:7]],
           "action_high": [round(float(x), 3) for x in np.atleast_1d(env.action_space.high)[:7]]}
    trials = []
    # realistic policy-scale deltas (2cm/step); 1.0/step is physically impossible
    # and makes every accumulated target unreachable -> IK fails -> no motion.
    for name, vec, mag in [("posX", [1, 0, 0], 0.02), ("negX", [-1, 0, 0], 0.02),
                           ("posZ", [0, 0, 1], 0.02), ("negZ", [0, 0, -1], 0.02)]:
        env.reset()
        for _ in range(3):  # settle
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        e0 = ee()
        for _ in range(10):
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            a[0], a[1], a[2] = mag * vec[0], mag * vec[1], mag * vec[2]
            a[-1] = 1.0
            env.step(a)
        disp = (ee() - e0)
        trials.append({"cmd": name, "ee_disp": disp.round(4).tolist(),
                       "disp_norm": round(float(np.linalg.norm(disp)), 4)})
    out["trials"] = trials
    return out


@app.function(image=image, timeout=900)
def diag_track(mag: float = 0.02, nsteps: int = 15, axis: int = 0):
    """Realistic small-delta tracking: command mag-sized EE deltas and log the
    full IK chain per step (target pose, IK solution / failure, actual EE)."""
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    robot = env.unwrapped.agent.robot
    arm = env.unwrapped.agent.controller.controllers["arm"]
    ln = [l.get_name() for l in robot.get_links()]
    ee_i = ln.index("ee_gripper_link")

    def ee():
        return np.array(list(robot.get_links()[ee_i].get_pose().p))

    for _ in range(3):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    e0 = ee()
    steps = []
    for i in range(nsteps):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[axis] = mag
        a[-1] = 1.0
        env.step(a)
        tp = getattr(arm, "_target_pose", None)
        tq = getattr(arm, "_target_qpos", None)
        sq = getattr(arm, "_start_qpos", None)
        ik_failed = (tq is not None and sq is not None
                     and np.allclose(np.asarray(tq), np.asarray(sq)))
        ee_base = arm.compute_fk(arm.qpos)
        steps.append({
            "i": i,
            "target_p": [round(float(x), 4) for x in tp.p] if tp is not None else None,
            "ee_base_p": [round(float(x), 4) for x in ee_base.p],
            "ik_eq_start(failed?)": bool(ik_failed),
            "ee_world_p": [round(float(x), 4) for x in ee()],
            "ee_disp_from_start": [round(float(x), 4) for x in (ee() - e0)],
        })
    return {"mag": mag, "axis": axis, "e0": [round(float(x), 4) for x in e0],
            "total_disp": [round(float(x), 4) for x in (ee() - e0)],
            "steps": steps}


@app.local_entrypoint()
def track(mag: float = 0.02, nsteps: int = 15, axis: int = 0):
    import json
    print(json.dumps(diag_track.remote(mag=mag, nsteps=nsteps, axis=axis),
                     indent=2, default=str))


@app.function(image=image, timeout=900)
def trace_control():
    """Trace the arm controller internals to see why actions aren't applied."""
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    robot = env.unwrapped.agent.robot
    agent = env.unwrapped.agent

    out = {"action_space_shape": list(env.action_space.shape),
           "action_low": [round(float(x), 4) for x in np.atleast_1d(env.action_space.low)],
           "action_high": [round(float(x), 4) for x in np.atleast_1d(env.action_space.high)]}

    cc = agent.controller
    out["combined_controller_type"] = type(cc).__name__
    subs = getattr(cc, "controllers", None)
    arm = None
    if subs is not None:
        out["subcontrollers"] = {k: type(v).__name__ for k, v in subs.items()}
        for k, v in subs.items():
            if "EE" in type(v).__name__ or "ee" in k.lower() or "arm" in k.lower():
                arm = v
    out["arm_controller_type"] = type(arm).__name__ if arm is not None else "NOT FOUND"
    if arm is not None:
        out["arm_config"] = {a: str(getattr(arm.config, a, None)) for a in
                             ("frame", "use_target", "use_delta", "normalize_action",
                              "pos_lower", "pos_upper", "rot_bound")}
        out["arm_action_dim"] = getattr(arm, "action_dimension", None)

    ln = [l.get_name() for l in robot.get_links()]
    ee_i = ln.index("ee_gripper_link")
    root = robot.get_links()[0]

    def ee_base():
        return list((root.get_pose().inv() * robot.get_links()[ee_i].get_pose()).p)

    log = []
    for i in range(6):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[0] = 1.0  # max +x world_vector
        a[-1] = 1.0
        env.step(a)
        tp = getattr(arm, "_target_pose", None)
        log.append({"step": i,
                    "target_pose_p": [round(float(x), 4) for x in tp.p] if tp is not None else None,
                    "ee_base_p": [round(float(x), 4) for x in ee_base()]})
    out["trace"] = log
    return out


@app.function(image=image, timeout=900)
def trace2():
    """Log the ee_align2 accumulation intermediates + quaternion norms."""
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    arm = env.unwrapped.agent.controller.controllers["arm"]

    logs = []
    orig_ctp = arm.compute_target_pose

    def wrapped(prev, action):
        try:
            cur = arm.compute_fk(arm.qpos)
            curinfo = {"cur_p": [round(float(x), 4) for x in cur.p],
                       "cur_qnorm": round(float(np.linalg.norm(cur.q)), 5)}
        except Exception as e:
            curinfo = {"cur_err": repr(e)}
        res = orig_ctp(prev, action)
        logs.append({**curinfo,
                     "prev_p": [round(float(x), 3) for x in prev.p],
                     "prev_qnorm": round(float(np.linalg.norm(prev.q)), 5),
                     "action": [round(float(x), 2) for x in np.asarray(action)],
                     "res_p": [round(float(x), 3) for x in res.p],
                     "res_qnorm": round(float(np.linalg.norm(res.q)), 5)})
        return res

    arm.compute_target_pose = wrapped
    # also capture what self.qpos returns
    try:
        logs_qpos = [round(float(x), 4) for x in np.asarray(arm.qpos)]
    except Exception as e:
        logs_qpos = repr(e)
    for _ in range(4):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[0] = 1.0
        a[-1] = 1.0
        env.step(a)
    return {"arm_qpos_property": logs_qpos, "accum": logs}


@app.local_entrypoint()
def trace2_ep():
    import json
    print(json.dumps(trace2.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def diag_init():
    """Why does the EE drift 0.31m under ZERO actions? Check the controller's
    initial _target_pose vs the true reset EE pose, and dump reset/ee_pose_at_base."""
    import inspect, numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    arm = env.unwrapped.agent.controller.controllers["arm"]
    out = {}
    for nm in ["reset", "_initialize_action_space", "set_drive_targets"]:
        try:
            out[f"src_{nm}"] = inspect.getsource(getattr(type(arm), nm))
        except Exception as e:
            out[f"src_{nm}"] = repr(e)
    try:
        out["src_ee_pose_at_base"] = inspect.getsource(type(arm).ee_pose_at_base.fget)
    except Exception as e:
        out["src_ee_pose_at_base"] = repr(e)
    # compare controller's stored initial target vs true FK(qpos)
    tp = arm._target_pose
    fk = arm.compute_fk(arm.qpos)
    epb = arm.ee_pose_at_base
    out["target_pose_p (init)"] = [round(float(x), 4) for x in tp.p]
    out["target_pose_q (init)"] = [round(float(x), 4) for x in tp.q]
    out["compute_fk(qpos)_p"] = [round(float(x), 4) for x in fk.p]
    out["compute_fk(qpos)_q"] = [round(float(x), 4) for x in fk.q]
    out["ee_pose_at_base_p (prop)"] = [round(float(x), 4) for x in epb.p]
    out["ee_pose_at_base_q (prop)"] = [round(float(x), 4) for x in epb.q]
    out["target_vs_fk_posdiff"] = round(float(
        np.linalg.norm(np.asarray(tp.p) - np.asarray(fk.p))), 5)
    return out


@app.local_entrypoint()
def diaginit():
    import json
    print(json.dumps(diag_init.remote(), indent=2, default=str))


@app.function(image=image, timeout=600)
def pose_test():
    """Is sapien.Pose multiply / inv preserving quaternion norm in this build?"""
    import numpy as np
    import sapien.core as sapien
    def qn(P): return round(float(np.linalg.norm(P.q)), 6)
    def pp(P): return {"p": [round(float(x),4) for x in P.p],
                       "q": [round(float(x),4) for x in P.q], "qnorm": qn(P)}
    cur = sapien.Pose(p=[0.25, -0.07, 0.22])
    delta = sapien.Pose([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    inv = cur.inv()
    bracket = cur * delta * inv
    out = {
        "cur": pp(cur), "delta": pp(delta), "cur_inv": pp(inv),
        "bracket(cur*delta*cur.inv)": pp(bracket),
        "cur*delta": pp(cur * delta),
        "(cur*delta)*inv": pp((cur * delta) * inv),
        "delta*delta": pp(delta * delta),
        # accumulation like the controller does:
    }
    prev = sapien.Pose(p=[0.25, -0.07, 0.22], q=[1.0, 0.0, 0.0, 0.0])
    accum = []
    for i in range(4):
        prev = bracket * prev
        accum.append({"step": i, **pp(prev)})
    out["accum_bracket*prev"] = accum
    return out


@app.local_entrypoint()
def posetest():
    import json
    print(json.dumps(pose_test.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def dump_src():
    """Print the EXACT ee_align2 controller formula + config + delta_pose quat."""
    import inspect, numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    arm = env.unwrapped.agent.controller.controllers["arm"]
    out = {}
    out["ctp_src"] = inspect.getsource(type(arm).compute_target_pose)
    try:
        out["set_action_src"] = inspect.getsource(type(arm).set_action)
    except Exception as e:
        out["set_action_src"] = repr(e)
    try:
        out["clip_scale_src"] = inspect.getsource(type(arm)._clip_and_scale_action)
    except Exception as e:
        out["clip_scale_src"] = repr(e)
    cfg = arm.config
    out["config"] = {k: repr(getattr(cfg, k)) for k in dir(cfg)
                     if not k.startswith("_")}
    # capture delta_pose by wrapping the rotation-conversion: just step once and
    # introspect locals via a patched compute_target_pose that records action math
    rec = {}
    orig = arm.compute_target_pose
    def wrap(prev, action):
        rec["raw_action"] = [round(float(x), 3) for x in np.asarray(action)]
        res = orig(prev, action)
        return res
    arm.compute_target_pose = wrap
    a = np.zeros(env.action_space.shape, dtype=np.float32); a[0] = 1.0; a[-1] = 1.0
    env.step(a)
    out["step_record"] = rec
    return out


@app.local_entrypoint()
def dump():
    import json
    print(json.dumps(dump_src.remote(), indent=2, default=str))


@app.local_entrypoint()
def trace():
    import json
    print(json.dumps(trace_control.remote(), indent=2, default=str))


@app.local_entrypoint()
def resp():
    import json
    print(json.dumps(test_response.remote(), indent=2, default=str))


@app.local_entrypoint()
def diagdt():
    import json
    print(json.dumps(diag_drivetarget.remote(), indent=2, default=str))
