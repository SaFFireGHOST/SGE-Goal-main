"""
Counterfactual Flow-DPO on Step 2 (plan_final_mask.md §5.1). Pairs (run/final_mask/build_cf_pairs.py): two HF train
rows with the same block in the same place and different tasks. Both directions are conditioned on ONE row's image and
M_i (the "conditioning" row c, the other row o); only the instruction and which mask is preferred change:

    E(M|T) = || v(z(M), sigma, I_c, M_i_c, T) - (eps - x(M)) ||^2,   same eps and sigma for all four (M, T)
    D(M|T) = E_theta(M|T) - E_ref(M|T)                               ref = frozen starting LoRA
    L_own   = -log sigmoid(-beta [D(M_c|T_c) - D(M_o|T_c)])
    L_other = -log sigmoid(-beta [D(M_o|T_o) - D(M_c|T_o)])
    L = flow MSE + 25 x channel consistency (anchor rows) + lambda_cf (L_own + L_other) / 2

Training uses each pair in both orientations (either row as c). M_i dropout: anchor rows 25% black + 25% damaged (as
midrop_step2); pair rows 25% damaged, none black, with the same damage in all four terms. Rows of the held-out episodes
(cf_pairs.json) are left out of the anchor stream too.

Validation (every --val-every updates, fixed noise, --pair-noise draws averaged per pair), on held-out pairs and on as
many train pairs (orientation c = row a only):
  text_acc     E(M|right T) < E(M|wrong T), over both masks: only the instruction differs (released ~ chance if the
               text is ignored). The main check.
  cf_acc       E(M_o|T_o) < E(M_c|T_o): under the other row's instruction the model prefers the other row's mask.
  own_acc      E(M_c|T_c) < E(M_o|T_c).
  dpo_loss, dpo_acc  as in training (relative to the reference; dpo_acc is 0 at step 0 by construction).
plus the trainer's val_loss and 10-step mask IoU on 256 validation tuples with learned M_i. A checkpoint is kept at
each validation. --eval-only scores --init-lora on the pair metrics and stops (baselines for released / midrop).

    python sge_goal/final_mask/train_step2_cfdpo.py --name cfdpo_step2 --init-lora run/final_mask/training/midrop_step2/last
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_flux_lora_lowmem import trainer  # noqa: E402  (also patches the low-memory prompt cache)
from train_step2_midrop import perturb  # noqa: E402

RUN = trainer.RUN


def episode(i):
    return i.rsplit("_atomic_", 1)[0]


class CfPairs(Dataset):
    """(conditioning row c, other row o) items over the HF train TupleDataset's images."""

    def __init__(self, ds, pairs, both_orientations):
        self.ds = ds
        self.items = [(q["a"], q["b"]) for q in pairs] + ([(q["b"], q["a"]) for q in pairs] if both_orientations else [])
        for q in pairs:
            assert ds.ids[q["a"]] == q["id_a"] and ds.ids[q["b"]] == q["id_b"], "cf_pairs.json row index mismatch"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c, o = self.items[i]
        d = self.ds
        return {"original": d._img("original_image", c), "m_i": d._img("initial_edit_region_mask", c),
                "m_own": d._img("final_edit_region_mask", c), "m_other": d._img("final_edit_region_mask", o),
                "instr_own": d.instructions[c], "instr_other": d.instructions[o], "index": i}


def collate_cf(items):
    out = {k: torch.stack([it[k] for it in items]) for k in ("original", "m_i", "m_own", "m_other")}
    for k in ("instr_own", "instr_other", "index"):
        out[k] = [it[k] for it in items]
    return out


def shared_noise(tr, n, generator):
    r = tr.args.resolution // tr.vsf
    noise = torch.randn((n, tr.vae.config.latent_channels, r, r), generator=generator, device=tr.device,
                        dtype=tr.vae.dtype)
    idx = (torch.rand(n, generator=generator, device=tr.device) * tr.scheduler.config.num_train_timesteps).long().cpu()
    return noise, idx


def errors(tr, batch, instr_key, noise, idx, adapter):
    """Per-example flow error of [preferred, dispreferred] masks under one instruction: (E_pref, E_disp), each (n,)."""
    n = batch["original"].shape[0]
    pref, disp = ("m_own", "m_other") if instr_key == "instr_own" else ("m_other", "m_own")
    both = {"original": batch["original"].repeat(2, 1, 1, 1), "m_i": batch["m_i"].repeat(2, 1, 1, 1),
            "instruction": batch[instr_key] * 2, "target": torch.cat([batch[pref], batch[disp]])}
    tr.transformer.set_adapter(adapter)
    e = tr.flow_and_aux_loss(both, noise=noise.repeat(2, 1, 1, 1), idx=idx.repeat(2), with_aux=False)[0]
    tr.transformer.set_adapter("default")
    return e[:n], e[n:]


def weighted_flow(tr, inp, m_own, m_other, w_diff):
    """Flow MSE per example with latent cells where the two masks differ up-weighted by 1 + w_diff (weighted mean)."""
    _, x0 = tr.flow_and_aux_loss(inp, with_aux=False, return_x0=True)  # x0 = noise - pred, so pred - target = x - x0
    with torch.no_grad():
        x = tr.encode(inp["target"])
        diff = ((m_own[:, :1] > 0) != (m_other[:, :1] > 0)).float().to(tr.device)
        cell = F.adaptive_max_pool2d(diff, x.shape[-2:])
        w = 1.0 + w_diff * cell  # (b, 1, h, w), broadcast over latent channels
    err = (x0.float() - x.float()) ** 2
    return ((w * err).mean((1, 2, 3)) / w.mean((1, 2, 3))), cell.mean().item()


def dpo_half(tr, batch, instr_key, noise, idx, beta):
    with torch.no_grad():
        rw, rl = errors(tr, batch, instr_key, noise, idx, "ref")
    pw, pl = errors(tr, batch, instr_key, noise, idx, "default")
    logits = -beta * ((pw - rw) - (pl - rl))
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean()


@torch.no_grad()
def pair_metrics(tr, loader, n_noise, beta, ref_cache):
    """Averages over n_noise fixed (noise, sigma) draws per pair. ref_cache[k] = reference errors (fixed noise)."""
    tr.transformer.eval()
    g = torch.Generator(device=tr.device).manual_seed(4321)
    rec = {k: [] for k in ("text_acc", "cf_acc", "own_acc", "dpo_loss", "dpo_acc", "text_gap")}
    for bi, batch in enumerate(loader):
        n = batch["original"].shape[0]
        pol = {"own": [torch.zeros(n, device=tr.device) for _ in range(2)],
               "other": [torch.zeros(n, device=tr.device) for _ in range(2)]}
        ref = {"own": [torch.zeros(n, device=tr.device) for _ in range(2)],
               "other": [torch.zeros(n, device=tr.device) for _ in range(2)]}
        for k in range(n_noise):
            noise, idx = shared_noise(tr, n, g)
            for h, key in (("own", "instr_own"), ("other", "instr_other")):
                p = errors(tr, batch, key, noise, idx, "default")
                ck = (bi, k, h)
                if ck not in ref_cache:
                    ref_cache[ck] = [x.clone() for x in errors(tr, batch, key, noise, idx, "ref")]
                r = ref_cache[ck]
                for j in range(2):
                    pol[h][j] += p[j] / n_noise
                    ref[h][j] += r[j] / n_noise
        # pol["own"] = (E(M_c|T_c), E(M_o|T_c)); pol["other"] = (E(M_o|T_o), E(M_c|T_o))
        e_cc, e_oc = pol["own"]
        e_oo, e_co = pol["other"]
        rec["text_acc"] += ((e_cc < e_co).float().tolist() + (e_oo < e_oc).float().tolist())
        rec["text_gap"] += ((e_co - e_cc) / e_cc).tolist() + ((e_oc - e_oo) / e_oo).tolist()
        rec["cf_acc"] += (e_oo < e_co).float().tolist()
        rec["own_acc"] += (e_cc < e_oc).float().tolist()
        for h in ("own", "other"):
            logits = -beta * ((pol[h][0] - ref[h][0]) - (pol[h][1] - ref[h][1]))
            rec["dpo_loss"] += (-F.logsigmoid(logits)).tolist()
            rec["dpo_acc"] += (logits > 0).float().tolist()
    tr.transformer.train()
    return {k: float(np.mean(v)) for k, v in rec.items()}


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cf-pairs", default=os.path.join(RUN, "final_mask", "cf_pairs.json"))
    p.add_argument("--lambda-cf", type=float, default=1.0)
    p.add_argument("--cf-beta", type=float, default=300.0)
    p.add_argument("--mi-drop", type=float, default=0.25, help="Anchor rows: black M_i.")
    p.add_argument("--mi-damage", type=float, default=0.25, help="Anchor rows: damaged M_i.")
    p.add_argument("--pair-mi-damage", type=float, default=0.25, help="Pair rows: damaged M_i (same in all terms).")
    p.add_argument("--pair-noise", type=int, default=4, help="Validation: noise draws per pair.")
    p.add_argument("--n-train-eval", type=int, default=115, help="Validation: train pairs scored (0 = none).")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--n-train-pairs", type=int, default=0,
                   help="Overfit test: train on this many train pairs only (0 = all); they are also the scored ones.")
    p.add_argument("--no-anchor", action="store_true", help="Overfit test: DPO terms only, no anchor flow loss.")
    p.add_argument("--skip-held", action="store_true", help="Do not score held-out pairs in validation.")
    p.add_argument("--sft-cf", action="store_true",
                   help="Supervised instead of DPO on the pairs: flow loss of M_c under T_c and of M_o under T_o (both "
                        "on row c's image and M_i), no reference, own noise per term.")
    p.add_argument("--diff-weight", type=float, default=0.0,
                   help="--sft-cf: latent cells where M_c and M_o differ (any pixel of the cell) weigh 1 + this; the "
                        "loss is the weighted mean per example (0 = plain flow loss).")
    own, rest = p.parse_known_args()
    sys.argv = sys.argv[:1] + rest + ["--mode", "dpo", "--step", "2"]
    args = trainer.parse_args()
    args.out_dir = args.out_dir if "--out-dir" in rest else os.path.join(RUN, "final_mask", "training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump({**vars(args), **{f"cf_{k}": v for k, v in vars(own).items()}}, f, indent=2)

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(os.path.join(out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    cf = json.load(open(own.cf_pairs))
    held = set(cf["summary"]["heldout_episodes"])
    pred = lambda split: os.path.join(args.pred_ier_dir, f"{split}.parquet") if args.pred_ier_dir else None
    train_ds = trainer.TupleDataset(args.train_parquet, 2, None if args.train_gt_mi else pred("train"), None, args.seed,
                                    args.resolution)
    pair_train = [q for q in cf["pairs"] if q["split"] == "train"]
    pair_held = [q for q in cf["pairs"] if q["split"] == "heldout"]
    if own.n_train_pairs:
        pair_train = random.Random(0).sample(pair_train, own.n_train_pairs)
    cf_ds = CfPairs(train_ds, pair_train, both_orientations=True)
    held_loader = DataLoader(CfPairs(train_ds, pair_held, False), batch_size=args.pairs_batch, shuffle=False,
                             num_workers=2, collate_fn=collate_cf) if not own.skip_held else None
    sub = pair_train if own.n_train_pairs else random.Random(0).sample(pair_train, min(own.n_train_eval, len(pair_train)))
    trev_loader = DataLoader(CfPairs(train_ds, sub, False), batch_size=args.pairs_batch, shuffle=False, num_workers=2,
                             collate_fn=collate_cf) if sub else None
    anchor_rows = [j for j in train_ds.order if episode(train_ds.ids[j]) not in held]
    train_ds.order = anchor_rows  # cf_ds indexes rows directly, so it is unaffected
    val_ds = trainer.TupleDataset(args.val_parquet, 2, pred("validation"), args.val_loss_samples, 0, args.resolution)
    iou_ds = trainer.TupleDataset(args.val_parquet, 2, pred("validation"), args.val_iou_samples, 1, args.resolution)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False, num_workers=2, collate_fn=trainer.collate)
    print(f"anchor rows {len(anchor_rows)} (held-out episodes {len(held)} left out), pair items {len(cf_ds)} "
          f"({len(pair_train)} pairs x 2), held-out pairs {len(pair_held)}, train pairs scored {len(sub)}", flush=True)

    tr = trainer.KontextLoraTrainer(args)
    tr.build_prompt_cache(train_ds.instructions + val_ds.instructions)
    ref_held, ref_trev = {}, {}

    def pair_val(rec):
        t = time.time()
        for name, loader, cache in (("held", held_loader, ref_held), ("train", trev_loader, ref_trev)):
            if loader is not None:
                rec.update({f"{name}_{k}": v for k, v in pair_metrics(tr, loader, own.pair_noise, own.cf_beta,
                                                                      cache).items()})
        rec["pair_val_sec"] = time.time() - t

    if own.eval_only:
        rec = {"type": "eval_only", "init_lora": args.init_lora}
        pair_val(rec)
        log(rec)
        return

    spe = len(cf_ds) // (args.pairs_batch * args.grad_accum)
    total = args.max_steps or max(1, int(args.epochs * spe))
    opt = torch.optim.AdamW(tr.lora_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))
    step, rng = 0, random.Random(args.seed)
    state_path = os.path.join(out, "last_state.pt")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location="cpu")
        tr.load_lora(os.path.join(out, "last", "pytorch_lora_weights.safetensors"))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step = st["step"]
        rng.seed(args.seed + step)
        print(f"resumed at step {step}", flush=True)

    def validate():
        rec = {"type": "val", "step": step, "val_loss": tr.val_loss(val_loader)}
        if args.val_iou_samples:
            rec["val_mask_iou"] = tr.val_mask_iou(iou_ds, args.val_iou_samples)
        pair_val(rec)
        if step:
            tr.save_lora(os.path.join(out, f"ckpt_{step:05d}"))
        log(rec)

    def loader_from(step, ds, bs, seed, coll):
        s = len(ds) // (bs * args.grad_accum)
        epoch, done = divmod(step, s)
        perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        return DataLoader(ds, batch_size=bs, sampler=perm[done * bs * args.grad_accum:], drop_last=True,
                          num_workers=args.num_workers, collate_fn=coll)

    def anchor_batches():
        s, epoch = len(train_ds) // (args.micro_batch * args.grad_accum), 0
        while True:
            yield from loader_from(epoch * s, train_ds, args.micro_batch, args.seed + 10_000, trainer.collate)
            epoch += 1

    if step == 0:
        validate()
    log({"type": "plan", "updates": total, "updates_per_epoch": spe})
    tr.transformer.train()
    t0, done = time.time(), step >= total
    torch.cuda.reset_peak_memory_stats()
    anchor = anchor_batches()
    while not done:
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader_from(step, cf_ds, args.pairs_batch, args.seed, collate_cf)):
            batch["m_i"] = perturb(batch["m_i"], 0.0, own.pair_mi_damage, rng)
            noise, idx = shared_noise(tr, batch["original"].shape[0], None)
            extra = {}
            for h, key in (("own", "instr_own"), ("other", "instr_other")):  # one graph alive at a time
                if own.sft_cf:
                    tgt = batch["m_own"] if key == "instr_own" else batch["m_other"]
                    inp = {"original": batch["original"], "m_i": batch["m_i"], "target": tgt, "instruction": batch[key]}
                    if own.diff_weight:
                        l, extra["diff_frac"] = weighted_flow(tr, inp, batch["m_own"], batch["m_other"], own.diff_weight)
                        l = l.mean()
                    else:
                        l = tr.flow_and_aux_loss(inp, with_aux=False)[0].mean()
                    (own.lambda_cf * l / 2 / args.grad_accum).backward()
                    extra[f"sft_{h}"] = l.item()
                    continue
                l, acc = dpo_half(tr, batch, key, noise, idx, own.cf_beta)
                (own.lambda_cf * l / 2 / args.grad_accum).backward()
                extra[f"dpo_{h}"], extra[f"dpo_acc_{h}"] = l.item(), acc.item()
            if own.no_anchor:
                flow, aux = torch.zeros(1), 0.0
            else:
                a = next(anchor)
                a = {**a, "m_i": perturb(a["m_i"], own.mi_drop, own.mi_damage, rng)}
                flow, aux = tr.flow_and_aux_loss(a)
                loss = flow.mean() + aux
                (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(tr.lora_params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                dt = (time.time() - t0) / args.log_every
                log({"type": "train", "step": step, "flow": flow.mean().item(), "aux": float(aux), **extra,
                     "grad_norm": gn.item(), "lr": sched.get_last_lr()[0], "sec_per_step": dt,
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                t0 = time.time()
            if args.val_every and step % args.val_every == 0:
                tr.save_lora(os.path.join(out, "last"))
                torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, state_path)
                validate()
            if step >= total:
                done = True
                break
    tr.save_lora(os.path.join(out, "last"))
    torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, state_path)
    if not args.val_every or step % args.val_every:
        validate()
    log({"type": "done", "step": step})


if __name__ == "__main__":
    main()
