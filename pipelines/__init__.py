from diffusers.utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
)


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_pt_objects import *  # noqa F403
else:

    from .pipeline_utils import AudioPipelineOutput, DiffusionPipeline, ImagePipelineOutput



try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:

    from .stable_diffusion import (
        StableDiffusionPipeline,
        StableDiffusionInpaintPipeline,

    )


