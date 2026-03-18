PYTHON       := python3
SCRIPT       := quantize.py

# Variáveis de ambiente repassadas ao script (sobrescrevem os defaults)
# MODELS_DIR     ?= models
# GGUF_DIR       ?= models/gguf
# USE_DOCKER     ?= 1  (defina 0 para usar binários locais)
# Quando USE_DOCKER=0:
#   LLAMA_CPP_DIR  ?= llama.cpp
#   CONVERT_SCRIPT ?= llama.cpp/convert_hf_to_gguf.py
#   LLAMA_QUANTIZE ?= llama.cpp/build-cpu/bin/llama-quantize

## quantize      — quantiza todos os modelos em models/ (binários locais por padrão)
##                 Opções: MODEL=<nome>  QUANT="Q8_0 Q4_K_M"  USE_DOCKER=1
quantize:
	$(PYTHON) $(SCRIPT) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(QUANT),--quant $(QUANT))

## convert-only  — apenas converte para FP16 GGUF, sem quantizar
##                 Opções: MODEL=<nome>  USE_DOCKER=1
convert-only:
	$(PYTHON) $(SCRIPT) --convert-only \
		$(if $(MODEL),--model $(MODEL))

## list-models   — lista os modelos disponíveis em models/
list-models:
	$(PYTHON) $(SCRIPT) --list

## help          — mostra esta mensagem
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## //'

.DEFAULT_GOAL := help