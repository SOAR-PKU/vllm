# LLM-RTC baseSystem integration

The `junjie/baseSystem` branch starts from vLLM v0.11.2,
commit `275de34170654274616082721348b7edd9741d32`, and supplies the compatibility
changes used by the LLM-RTC-Audio-Omni `dev/baseSystem` branch. The parent
repository records the exact vLLM commit in its submodule gitlink; the branch
name does not automatically select a dependency version.

## Qwen3-Omni vision attention

The vision tower chooses its attention backend before constructing the vision
blocks. Each block receives that backend, the upstream FlashAttention flag,
and the attention backend override. This keeps the block implementation aligned
with the tower's attention metadata, including its `max_seqlen` calculation.

The existing automatic promotion to upstream FlashAttention when it is available
is retained. This includes the existing promotion behavior for a non-FlashAttention
override. No new kernel or multimodal regrouping policy is introduced, but changing
which existing kernel executes can affect latency and memory use. CPU constructor
checks establish parameter propagation; numerical correctness and performance of
the selected GPU kernel require separate validation.

## Multimodal reference misses

[Stable UUID inputs](features/multimodal_inputs.md#stable-uuids-for-caching-multi_modal_uuids)
can omit a raw media payload when the corresponding processor-cache entry exists.
If a supplied reference misses the cache and has no raw payload, the existing
missing-item check raises `MultiModalCacheMissError`, a subclass of `ValueError`,
from `vllm.multimodal.cache_exceptions`.

The error retains the original message and exposes:

- `modality` and `index`: the first missing item found by the processor.
- `identifier`: the effective processor-cache hash, including preprocessing
  options; it is not necessarily the caller's UUID.
- `phase="input_preprocessing"` and `admitted=False`: this miss is detected
  before the normal request is submitted to EngineCore.

Cache checks have not updated the processor's cached items at this point. Cached
references and full-payload cache misses retain their existing processing paths.
Ordinary validation errors remain distinct from the typed reference miss.

LLM-RTC's Instance and Router implement the bounded full-payload repair protocol.
vLLM does not automatically resubmit the request. Applications must preserve their
original payloads if they implement recovery, and must not interpret successful
input preprocessing as completed GPU prefill. Post-admission failures are outside
this error's recovery boundary.

## Baseline scope and validation

These changes support the baseline transport and model execution paths. They do
not introduce the project's streaming-request coalescing, query-priority
scheduling, multimodal regrouping, feature prefetch, or persistent streaming lanes.
Image UUID reuse and attention backend alignment can still affect measured
performance and should be recorded as part of the compared configuration.

The parent repository's CPU tests cover typed cache misses and the actual vision
constructors with dependency stand-ins:

- `tests/python/test_mm_reference_miss.py`
- `tests/python/test_qwen3_vision_backend_plumbing.py`

These checks cover the miss boundary, normal cache paths, backend selection, and
argument propagation. They do not execute CUDA attention or a real P/D KV transfer.
