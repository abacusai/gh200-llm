#!/bin/bash

echo $GHCR_AUTH_TOKEN | docker login -u oauth2accesstoken --password-stdin https://ghcr.io
docker buildx build --progress plain --platform linux/arm64,linux/x86_64 -t ghcr.io/abacusai/gh200-llm/llm-train-serve:latest --push .
