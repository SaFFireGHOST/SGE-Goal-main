"""Stub flash_attn package for transformers check_imports compatibility.
Contains minimal stubs so that model files with 'import flash_attn' pass
the import check. Actual flash attention is NOT available - models should
fall back to eager attention via their has_flash_attn guards.
"""

def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn stub - flash attention not available")

def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn stub - flash attention not available")
