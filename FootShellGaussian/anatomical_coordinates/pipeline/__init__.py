"""One smooth run from prepared shoes to the older address stage.

The differentiable stages (foot fit, lower-leg fit) live here and run on the
GPU. Everything downstream - anatomical surface, the boundary target stage, the cage deformation stage, the older map lookup, the older address stage - is
the existing ``foot_prior`` implementation, driven as a subprocess and never
modified. The bridge between the two worlds is the artifact contract: this
package writes the same ``containment_fit`` and ``lower_leg_attachment``
schemas the CPU stages already validate, so nothing downstream has to know a
torch fitter produced them.
"""

from .shoe_set import PIPELINE_ROOT, UNIFIED_INPUT_ROOT, all_shoes, load_case

__all__ = ["PIPELINE_ROOT", "UNIFIED_INPUT_ROOT", "all_shoes", "load_case"]
