# !/usr/bin/env bash
export LD_LIBRARY_PATH=/workspace/.local/share/nsight_systems:$LD_LIBRARY_PATH

/workspace/.local/share/nsight_systems/nsys profile \
    --trace=cuda,osrt,nvtx \
    --output=dnn_profile \
    --force-overwrite true \
    python scratch_dnn.py "$@"