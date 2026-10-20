"""
Standalone-Pinocchio kinematics backend for ManiSkill2_real2sim controllers.

SAPIEN 2.2.2's bundled pinocchio `compute_forward_kinematics` segfaults in the
Modal environment (validated: crashes for every dtype/size/host; model creation
and get_link_pose are fine, the FK compute itself crashes). We replace the
controller's `self.pmodel` with a model backed by standalone pinocchio
(`pip install pin`), which has been validated to reproduce SAPIEN-native link
poses to 0.0 error.

Usage (call BEFORE simpler_env.make so the controller picks it up):
    from pinocchio_ik_backend import patch_pd_ee_controllers
    patch_pd_ee_controllers()   # auto-globs the widowx urdf inside the container
"""
import glob


def _find_widowx_urdf():
    hits = glob.glob("/root/SimplerEnv/**/wx250s.urdf", recursive=True)
    if not hits:
        hits = glob.glob("/**/wx250s.urdf", recursive=True)
    if not hits:
        raise FileNotFoundError("wx250s.urdf not found in container")
    return hits[0]


class StandalonePinocchioModel:
    """Drop-in replacement for sapien's articulation.create_pinocchio_model(),
    exposing exactly the methods the PD-EE controllers call on `self.pmodel`."""

    def __init__(self, articulation, urdf_path):
        import pinocchio as pin
        import numpy as np
        self.pin = pin
        self.np = np
        self.model = pin.buildModelFromUrdf(urdf_path)  # fixed base
        self.data = self.model.createData()

        self.link_names = [l.get_name() for l in articulation.get_links()]
        self.ajoint_names = [j.get_name() for j in articulation.get_active_joints()]
        # map: sapien dof index -> (pin q index, pin v index)
        self.qmap = []
        for s_idx, jn in enumerate(self.ajoint_names):
            if self.model.existJointName(jn):
                jid = self.model.getJointId(jn)
                self.qmap.append((s_idx, self.model.idx_qs[jid], self.model.idx_vs[jid]))
        self._fid_cache = {}

    # -- helpers ---------------------------------------------------------------
    def _q_from_full(self, full_qpos):
        q = self.pin.neutral(self.model)
        for s_idx, qi, _vi in self.qmap:
            q[qi] = float(full_qpos[s_idx])
        return q

    def _frame_id(self, link_idx):
        name = self.link_names[link_idx]
        if name not in self._fid_cache:
            self._fid_cache[name] = self.model.getFrameId(name)
        return self._fid_cache[name]

    # -- interface used by the controllers ------------------------------------
    def compute_forward_kinematics(self, full_qpos):
        q = self._q_from_full(self.np.asarray(full_qpos, dtype=float))
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

    def get_link_pose(self, link_idx):
        import sapien.core as sapien
        oMf = self.data.oMf[self._frame_id(link_idx)]
        quat = self.pin.Quaternion(oMf.rotation)
        return sapien.Pose(p=list(oMf.translation),
                           q=[quat.w, quat.x, quat.y, quat.z])

    # -- qpos reconstruction from (reliable) link poses ------------------------
    # SAPIEN's get_qpos() returns a constant garbage array in this build (same
    # broken-array-return bug as pinocchio). get_pose() is reliable, so we
    # reconstruct each joint angle from the relative transform of its parent and
    # child links vs the model's zero-config relative transform.
    def _setup_reconstruction(self):
        pin, np = self.pin, self.np
        m, d = self.model, self.data
        sapien_names = set(self.link_names)

        def is_body(f):
            try:
                return int(f.type) == int(pin.FrameType.BODY)
            except Exception:
                return str(f.type).upper().endswith("BODY")

        body_of_joint = {}
        for f in m.frames:
            if is_body(f) and f.name in sapien_names:
                body_of_joint.setdefault(f.parentJoint, f.name)

        q0 = pin.neutral(m)
        pin.forwardKinematics(m, d, q0)
        pin.updateFramePlacements(m, d)

        def link_oMf(name):
            return d.oMf[m.getFrameId(name)].copy()

        self._recon = []
        for s_idx, jn in enumerate(self.ajoint_names):
            if not m.existJointName(jn):
                continue
            jid = m.getJointId(jn)
            cl = body_of_joint.get(jid)
            pl = body_of_joint.get(m.parents[jid])
            if cl is None or pl is None:
                continue
            Mpc0 = link_oMf(pl).inverse() * link_oMf(cl)
            eps = 1e-4
            qe = q0.copy()
            qe[m.idx_qs[jid]] += eps
            pin.forwardKinematics(m, d, qe)
            pin.updateFramePlacements(m, d)
            Mpc_e = link_oMf(pl).inverse() * link_oMf(cl)
            pin.forwardKinematics(m, d, q0)
            pin.updateFramePlacements(m, d)
            screw = pin.log6(Mpc0.inverse() * Mpc_e).vector / eps
            v, w = screw[:3], screw[3:]
            if np.linalg.norm(w) >= np.linalg.norm(v):
                kind, axis = "R", w / (np.linalg.norm(w) + 1e-12)
            else:
                kind, axis = "P", v / (np.linalg.norm(v) + 1e-12)
            self._recon.append((s_idx, pl, cl, kind, axis, Mpc0))

    def reconstruct_qpos(self, articulation):
        pin, np = self.pin, self.np
        if not hasattr(self, "_recon"):
            self._setup_reconstruction()
        pose = {l.get_name(): l.get_pose() for l in articulation.get_links()}

        def to_se3(p):
            q = p.q
            R = pin.Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])).matrix()
            return pin.SE3(R, np.asarray(p.p, dtype=float))

        qfull = np.zeros(articulation.dof, dtype=float)
        for s_idx, pl, cl, kind, axis, Mpc0 in self._recon:
            Mobs = to_se3(pose[pl]).inverse() * to_se3(pose[cl])
            delta = Mpc0.inverse() * Mobs
            if kind == "R":
                qfull[s_idx] = float(np.dot(pin.log3(delta.rotation), axis))
            else:
                qfull[s_idx] = float(np.dot(delta.translation, axis))
        return qfull

    def compute_inverse_kinematics(self, ee_link_idx, target_pose,
                                   initial_qpos, active_qmask, max_iterations=100):
        """Damped-least-squares (CLIK) frame IK. Returns (full_qpos, success, err).
        target_pose is a sapien.Pose in the robot base frame."""
        pin, np = self.pin, self.np
        fid = self._frame_id(ee_link_idx)
        tq = target_pose.q  # [w, x, y, z]
        R = pin.Quaternion(float(tq[0]), float(tq[1]), float(tq[2]), float(tq[3])).matrix()
        oMdes = pin.SE3(R, np.asarray(target_pose.p, dtype=float))

        q = self._q_from_full(np.asarray(initial_qpos, dtype=float))
        vmask = np.zeros(self.model.nv, dtype=bool)
        for s_idx, _qi, vi in self.qmap:
            if bool(active_qmask[s_idx]):
                vmask[vi] = True

        eps, damp, DT = 1e-4, 1e-6, 1.0
        success = False
        err = np.zeros(6)
        for _ in range(int(max_iterations)):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, fid)
            oMf = self.data.oMf[fid]
            iMd = oMf.actInv(oMdes)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < eps:
                success = True
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, fid)  # 6 x nv, LOCAL
            J = -pin.Jlog6(iMd.inverse()) @ J
            J[:, ~vmask] = 0.0
            v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q = pin.integrate(self.model, q, v * DT)

        result = np.array(initial_qpos, dtype=float).copy()
        for s_idx, qi, _vi in self.qmap:
            result[s_idx] = q[qi]
        return result, success, float(np.linalg.norm(err))


class _ArticulationProxy:
    """Delegates everything to the real articulation EXCEPT get_qpos(), which is
    reconstructed from reliable link poses (SAPIEN's get_qpos is broken here)."""

    def __init__(self, articulation, pmodel):
        object.__setattr__(self, "_art", articulation)
        object.__setattr__(self, "_pm", pmodel)

    def get_qpos(self):
        return self._pm.reconstruct_qpos(self._art)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_art"), name)


def _quat_mul(a, b):
    """Hamilton product of two [w,x,y,z] quaternions (numpy)."""
    import numpy as np
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


def _quat_to_mat(q):
    """[w,x,y,z] -> 3x3 rotation matrix (numpy)."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _np_compute_target_pose(self, prev_ee_pose_at_base, action):
    """Pure-numpy reimplementation of PDEEPoseController.compute_target_pose.

    SAPIEN 2.2.2's compiled Pose quaternion arithmetic is BROKEN in this build
    (cur*delta*cur.inv() returns a norm-4 quaternion instead of unit -> the
    ee_align2 accumulation explodes geometrically and the arm ignores every
    action). We replicate the exact ManiSkill formulas with numpy and return a
    re-normalized sapien.Pose, never touching sapien's Pose `*`/`.inv()`."""
    import numpy as np
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation

    cfg = self.config
    delta_pos = np.asarray(action[0:3], dtype=float)
    rot = np.asarray(action[3:6], dtype=float)
    dq = Rotation.from_rotvec(rot).as_quat()[[3, 0, 1, 2]]  # -> [w,x,y,z]

    if not cfg.use_delta:
        assert cfg.frame == "base", cfg.frame
        return sapien.Pose(delta_pos, dq / (np.linalg.norm(dq) + 1e-12))

    Rd = _quat_to_mat(dq)
    pp = np.asarray(prev_ee_pose_at_base.p, dtype=float)
    pq = np.asarray(prev_ee_pose_at_base.q, dtype=float)  # [w,x,y,z]

    if cfg.frame == "base":
        new_p = delta_pos + Rd @ pp
        new_q = _quat_mul(dq, pq)
    elif cfg.frame == "ee":
        Rp = _quat_to_mat(pq)
        new_p = pp + Rp @ delta_pos
        new_q = _quat_mul(pq, dq)
    elif cfg.frame == "ee_align":
        new_p = pp + delta_pos
        new_q = _quat_mul(dq, pq)
    elif cfg.frame == "ee_align2":
        c = np.asarray(self.compute_fk(self.qpos).p, dtype=float)
        # (Pose(p=c) * delta * Pose(p=c).inv()) * prev , done analytically:
        new_p = c + delta_pos + Rd @ (pp - c)
        new_q = _quat_mul(dq, pq)
    else:
        raise NotImplementedError(cfg.frame)

    new_q = new_q / (np.linalg.norm(new_q) + 1e-12)
    return sapien.Pose(new_p, new_q)


def patch_pd_ee_controllers(urdf_path=None):
    """Monkeypatch PDEEPosController so its pinocchio model is our standalone one
    AND its articulation.get_qpos() returns a reconstructed (reliable) qpos.
    Also replaces compute_target_pose with a numpy reimplementation because
    SAPIEN's compiled Pose arithmetic is broken in this build.
    Must be called BEFORE the env/controller is constructed."""
    if urdf_path is None:
        urdf_path = _find_widowx_urdf()
    import mani_skill2_real2sim.agents.controllers.pd_ee_pose as ppm

    _orig_init = ppm.PDEEPosController._initialize_joints

    def _patched_initialize_joints(self):
        _orig_init(self)  # builds sapien's (broken-FK but creation-safe) pmodel + qmask/ee_link_idx
        pm = StandalonePinocchioModel(self.articulation, urdf_path)
        pm._setup_reconstruction()
        self.pmodel = pm
        # route all get_qpos() reads through reliable reconstruction
        self.articulation = _ArticulationProxy(self.articulation, pm)

    ppm.PDEEPosController._initialize_joints = _patched_initialize_joints
    # bypass broken sapien Pose math in the ee target accumulation
    ppm.PDEEPoseController.compute_target_pose = _np_compute_target_pose
    return urdf_path
