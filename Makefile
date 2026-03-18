PYTHON       := python3
DOCKERFILE   := docker-compose/Dockerfile.engine
DOCKER_IMAGE := tcc-engine

# Variáveis de ambiente repassadas ao script (sobrescrevem os defaults)
# MODELS_DIR     ?= models
# GGUF_DIR       ?= models/gguf
# USE_DOCKER     ?= 1  (defina 0 para usar binários locais)
# Quando USE_DOCKER=0:
#   LLAMA_CPP_DIR  ?= llama.cpp
#   CONVERT_SCRIPT ?= llama.cpp/convert_hf_to_gguf.py
#   LLAMA_QUANTIZE ?= llama.cpp/build-cpu/bin/llama-quantize

.PHONY: build-engine quantize convert-only list-models help

## build-engine  — constrói a imagem Docker com llama.cpp + ROCm
build-engine:
	docker build -f $(DOCKERFILE) -t $(DOCKER_IMAGE) .