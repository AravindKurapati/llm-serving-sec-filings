# v1 Kaggle: What Broke and Why

This documents every error hit trying to run vLLM on Kaggle T4 GPUs. Preserved here because the debugging process is useful context for anyone trying to run modern LLM serving on older hardware.

---

## The Core Problem

Kaggle's free tier provides T4 GPUs. The T4 has compute capability **sm_75**.

Modern vLLM (0.6+) uses **FlashInfer** as its default attention backend. FlashInfer dropped sm_75 support in late 2024. This created a cascading dependency problem that no amount of pip pinning could cleanly solve.

---

## Error Timeline

### Error 1 - FlashInfer CUDA compilation failure
```
cutlass.base_dsl.common.DSLRuntimeError: ICE
NVVM Compilation Error: Target Architecture: sm_75
```
**Root cause**: vLLM imports FlashInfer on startup. FlashInfer tries to JIT-compile CUDA kernels. The NVVM compiler rejects sm_75.

**Attempted fix**: `VLLM_ATTENTION_BACKEND=XFORMERS` env var  
**Result**: Failed — env var not inherited by nohup subprocess

**Attempted fix**: `pip uninstall flashinfer`  
**Result**: vLLM fell back to xformers and started — but broke on first request (see Error 2)

---

### Error 2 - outlines import failure on first request
```
ModuleNotFoundError: No module named 'outlines.fsm'
```
**Root cause**: vLLM 0.6.3 uses `outlines` for guided decoding. Kaggle's base image had a newer outlines that restructured its module layout (`outlines.fsm` was renamed).

**Attempted fix**: `pip install outlines==0.0.46`  
**Result**: Fixed outlines.fsm — but exposed Error 3

---

### Error 3 - pyairports missing
```
ModuleNotFoundError: No module named 'pyairports'
```
**Root cause**: The version of outlines that had `outlines.fsm` also required `pyairports` as a dependency. It wasn't in the base image.

**Attempted fix**: `pip install pyairports`  
**Result**: Package installed in notebook kernel but not in vLLM's subprocess — needed vLLM restart to pick it up

**Attempted fix**: Killed and restarted vLLM after installing pyairports  
**Result**: Fixed pyairports — but exposed Error 4

---

### Error 4 - tokenizers binary incompatibility
```
AttributeError: TokenizersBackend has no attribute all_special_tokens_extended
```
**Root cause**: vLLM 0.6.3 was built against `tokenizers==0.19.x`. Kaggle's base image had a newer tokenizers that removed `all_special_tokens_extended`.

**Attempted fix**: `pip install tokenizers==0.19.1 --force-reinstall`  
**Result**: Version showed as 0.19.1 but vLLM subprocess still loaded the old cached version — required kernel restart

**After kernel restart**: tokenizers was correct, but vLLM now failed because `transformers` version was incompatible with the pinned tokenizers

---

### Error 5 - transformers/tokenizers version matrix
```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```
**Root cause**: Pinning tokenizers==0.19.1 required transformers==4.43.0, which required an older huggingface_hub, which conflicted with Kaggle's base image huggingface_hub.

At this point the dependency graph was:
- vLLM 0.6.3 needs tokenizers==0.19.x
- tokenizers==0.19.x needs transformers==4.43.x  
- transformers==4.43.x needs huggingface_hub<0.24
- Kaggle base image has huggingface_hub>=0.26
- These cannot coexist

---

### Attempted move to Colab

Moved to Google Colab hoping for a cleaner base image. Hit the exact same tokenizers conflict. Root cause confirmed: the conflict is not Kaggle-specific, it's a fundamental incompatibility between vLLM 0.6.3 and current Python environments on any platform with a pre-installed ML stack.

---

### Attempted: latest vLLM on Colab

Latest vLLM requires `transformers>=4.45` for the `mllama` module. But latest vLLM also re-introduced FlashInfer as default on T4, causing the original sm_75 error. Setting `VLLM_ATTENTION_BACKEND=XFORMERS` in the subprocess env should have worked but the env var propagation was unreliable across Modal boundaries.

---

## Root Cause Summary

| Layer | Problem |
|-------|---------|
| Hardware | T4 = sm_75, modern attention backends require sm_80+ |
| vLLM old (0.6.3) | Needs old tokenizers, conflicts with current base images |
| vLLM new (0.15+) | Needs FlashInfer, which requires sm_80+ |
| Platform | Both Kaggle and Colab pre-install conflicting versions |

There is no version of vLLM that works cleanly on T4 with current Kaggle/Colab base images. The only  solutions are:
1. Use newer GPU hardware (sm_80+)
2. Use a completely clean Docker environment with pinned deps
3. Use transformers directly (no vLLM) but it would be slower.

---

## Solution

Modal with A10G GPU (sm_86). Latest vLLM installs cleanly in Modal's Debian Slim container with no pre-installed conflicts. FlashAttention 2 works natively. Deployed in one command.

See [`../v2_modal/README.md`](../v2_modal/README.md) for the working implementation.
