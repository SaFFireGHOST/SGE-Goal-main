# Third-party code

SGE-Goal builds on several open-source projects. Full upstream repositories are **not** vendored
here; instead they are pinned to the exact commits used for the paper and fetched on demand, with
our (small) local modifications applied as patches.

```bash
bash third_party/fetch_third_party.sh            # all of them
bash third_party/fetch_third_party.sh xmem sam   # a subset: xmem | sam | susie | clover
```

| Project | Upstream | Pinned commit | Local patch | Checked out to | License |
|---|---|---|---|---|---|
| XMem | [hkchengrex/XMem](https://github.com/hkchengrex/XMem) | `f3b841d` | `patches/xmem.patch` (adds `eval_custom.py`) | `data_pipeline/XMem`, `evaluator/XMem` | MIT |
| Segment Anything | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | `dca509f` | `patches/segment-anything_data_pipeline.patch` (data pipeline only) | `data_pipeline/segment-anything`, `evaluator/segment-anything` (unpatched) | Apache-2.0 |
| SuSIE | [kvablack/susie](https://github.com/kvablack/susie) | `8177f63` | `patches/susie.patch` | `baselines/susie/susie` | MIT |
| CLOVER | [OpenDriveLab/CLOVER](https://github.com/OpenDriveLab/CLOVER) | `4b2a75e` | — | `baselines/clover/CLOVER`, `downstream/calvin/gr_mg/CLOVER-repo` | Apache-2.0 |

Smaller upstream code is vendored directly because it is imported as a local package:

| Directory | Origin | License |
|---|---|---|
| `data_pipeline/evf_sam_model/` | [hustvl/EVF-SAM](https://github.com/hustvl/EVF-SAM) (includes code from SAM, SAM 2, EfficientSAM and Microsoft UniLM/BEiT-3) | Apache-2.0 (UniLM: MIT) |
| `data_pipeline/loftr_src/`, `evaluator/loftr_src/` | [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) | Apache-2.0 |
| `downstream/calvin/gr_mg/` | fork of [bytedance/GR-MG](https://github.com/bytedance/GR-MG) — see its `USAGE.md` for our changes | Apache-2.0 |
| `downstream/calvin/ghil_glue/jaxrl_m/` | `jaxrl_m` as distributed with [SuSIE](https://github.com/kvablack/susie) / GHIL-Glue | MIT |

Please consult each upstream repository for its exact licence terms.

## Model weights

Weights are not stored in this repository. The code expects them at the following locations:

| Weights | Source | Expected path |
|---|---|---|
| SAM ViT-H (`sam_vit_h_4b8939.pth`) | [segment-anything model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) | `data_pipeline/`, `evaluator/` |
| XMem (`XMem.pth`, `XMem-s012.pth`) | [XMem releases](https://github.com/hkchengrex/XMem/releases) | `data_pipeline/XMem/saves/`, `evaluator/XMem/saves/` |
| LoFTR (`indoor_ds_new`, `indoor_ot`, `outdoor_ds`, `outdoor_ot` `.ckpt`) | [LoFTR weights](https://github.com/zju3dv/LoFTR#installation) | `data_pipeline/loftr_weights/`, `evaluator/loftr_weights/` |
| EVF-SAM | `YxZhang/evf-sam` (Hugging Face, downloaded automatically) | — |
| BLIP-2, Florence-2, Grounding DINO, SAM (HF), VideoChat-Flash, FLUX.1 Kontext [dev], FLUX.2 klein | Hugging Face Hub (downloaded automatically; FLUX models require accepting their licences) | — |
