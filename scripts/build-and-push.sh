#!/bin/bash

echo $GHCR_AUT_TOKEN | docker login -u oauth2accesstoken --password-stdin https://ghci.io
docker buildx build --progress plain --platform linux/arm64 -t ghcr.io/abacusai/gh200-images/llm-train-serve:latest --push .
