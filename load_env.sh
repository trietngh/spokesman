#!/usr/bin/env bash
if [ -f "$PIXI_WORKSPACE_ROOT/.env" ]; then
    set -a
    source "$PIXI_WORKSPACE_ROOT/.env"
    set +a
fi