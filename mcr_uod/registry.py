"""Bridge between the research package and the vendored ``ultralytics`` tree.

The causal modules are *not* registered in ``ultralytics.nn.modules.__init__``
so the stock library stays clean. Instead, call :func:`register_causal_modules`
at startup to monkey-patch the causal block into ``ultralytics`` before a model
YAML is parsed.

This is intentionally a stub: registration depends on the concrete module
implementation (``mcr_uod.causal.block.CausalBlock``) which is not written yet.
"""

from __future__ import annotations


def register_causal_modules() -> None:
    """Register causal modules with ``ultralytics``.

    Intended behaviour (TODO):
        1. Import ``ultralytics.nn.modules`` and the causal block class.
        2. Inject the block into the module namespace so ``parse_model`` can
           resolve its name from a model YAML (``globals()`` lookup).
        3. Optionally patch ``WorldModel.forward`` so the confounder bank flows
           through the causal block (replacing the current ``C2fAttn`` hook).

    Raises:
        NotImplementedError: Always, until the causal block is implemented.
    """
    raise NotImplementedError("register_causal_modules is not implemented yet")
