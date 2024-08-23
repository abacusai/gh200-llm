#!/bin/bash

echo $GHCR_AUTH_TOKEN | docker login -u $GHCR_USER --password-stdin https://ghcr.io
existing=`docker buildx ls | grep llm-default-builder`
if [[ x$existing == "x" ]]; then
    docker buildx create --name llm-default-builder --use
fi
docker buildx build --progress plain --platform linux/arm64,linux/x86_64 -t ghcr.io/abacusai/gh200-llm/llm-train-serve:latest --push .
