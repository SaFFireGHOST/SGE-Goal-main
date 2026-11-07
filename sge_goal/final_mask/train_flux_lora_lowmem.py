"""
sge_goal/variant_a/flux_lora_trainer.py for a shared GPU: the transformer is parked on the CPU while the prompt
cache is built, so the text encoders (T5 ~9.5 GB) and the transformer are never on the GPU together
(startup peak ~39 GB -> ~29 GB). Training itself is unchanged. Same arguments as the trainer:

    python sge_goal/final_mask/train_flux_lora_lowmem.py --step 2 --mode dpo --name ... --pairs ...
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "variant_a"))
import flux_lora_trainer as trainer  # noqa: E402

_build_prompt_cache = trainer.KontextLoraTrainer.build_prompt_cache


def build_prompt_cache(self, instructions):
    self.transformer.to("cpu")  # in place: parameter objects (and self.lora_params) are kept
    torch.cuda.empty_cache()
    _build_prompt_cache(self, instructions)
    self.transformer.to(self.device)


trainer.KontextLoraTrainer.build_prompt_cache = build_prompt_cache

if __name__ == "__main__":
    trainer.main()
