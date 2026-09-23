"""ASTRA Fabric (ADR-0011): pool GPUs across machines, exo-style, on NVIDIA.

Every machine runs ``astra agent``. The agent advertises its GPUs over UDP
discovery and HTTP (``/v1/node``) and, with ``--rpc``, serves each GPU to the
cluster through one llama.cpp ``rpc-server`` process. The head node discovers
the agents, the planner places contiguous layer ranges on the fastest set of
local and remote GPUs, and one ``llama-server`` drives them all
(``--rpc`` + ``--device`` + ``--tensor-split``) behind a single
OpenAI-compatible endpoint.
"""
