"""Causal reasoning modules for MCR-UOD.

This subpackage is owned by the causal team. It implements:

    - ``confounder``        confounder dictionary Z + prior P(Z) (Sec 4.2)
    - ``lgre``              language-guided region exploration (Eq.5)
    - ``bacr``              backdoor adjustment causal reasoning (Eq.6-10)
    - ``nwgm``              normalised weighted geometric mean (Eq.7)
    - ``prompt_intervene``  causal intervention over the prompt bank
    - ``zoom``              region / feature magnification for distant objects
    - ``adapter``           join point with the uncertainty team
"""

__all__ = ["confounder", "lgre", "bacr", "nwgm", "prompt_intervene", "zoom", "adapter"]
