# `baselines/` — goal generators compared in the paper

Each directory generates goal images for the same test folders used by SGE-Goal
(`<test_data>/<case>/{original_image.*, edit_instruction.txt}` → `edited_image.*`).

| Directory | Generator | Notes |
|---|---|---|
| `susie/` | SuSIE | needs `susie/susie` (`bash third_party/fetch_third_party.sh susie`) |
| `clover/` | CLOVER visual planner | needs `clover/CLOVER` (`bash third_party/fetch_third_party.sh clover`) |
| `qwen_image_edit/` | Qwen Image Edit | Modal inference scripts |
| `flux_kontext_base/` | FLUX.1 Kontext [dev], not fine-tuned | `modal_flux_kontext_pretrained_inference.py` |
| `flux2_klein_base/` | FLUX.2 klein, not fine-tuned | `modal_flux2_klein_pretrained_inference.py` |
| `gemini_3_pro_image/` | Gemini 3 Pro Image (API) | `inference.py`; needs your own API key |
| `gpt_image_1_5/` | GPT Image 1.5 (API) | `inference.py`; needs your own API key |

## Mask ablations (`ablations/`)

Step-3 goal-image synthesis retrained with fewer spatial inputs, on BridgeDataV2 and CALVIN, for
both base editors:

| Suffix | Model input | Loss |
|---|---|---|
| *(full model, see `sge_goal/`)* | original image + `M_i` + `M_f` | rectified flow + static-scene consistency |
| `_wo_final_mask` | original image + `M_i` | see each directory's `README.md` / training script |
| `_wo_masks` | original image only | rectified flow only (no masks to define the static region) |

Each ablation directory is self-contained (trainer, Modal launcher, inference script).
