#!/usr/bin/env bash
build_and_prune() {
    # Set TARGET and DF-SUFFIX using the passed in parameters
    local TARGET=$TARGET
    local DF_SUFFIX=$DF_SUFFIX
    local PYTHON_V=$PYTHON_V  ## same name
    local USE_PROXY=$USE_PROXY

    echo "defe is $TARGET"
    echo "df-suffix is $DF_SUFFIX"
    echo "python version is $PYTHON_V"
    echo "use proxy is $USE_PROXY"
}