import subprocess
import time
import json
import os
import psutil

# 1. initialize() -> Configurações iniciais
config = {
    "original_model": "models/llama-3.1-8b.fp16.gguf",
    "quantization_params": ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"],
    "benchmark_params": {"iteration": 3, "n_predict": 128},
    "device_params": {"threads": 8, "gpu_layers": 35},
    "test_data": "benchmark_test_set.parquet"
}

# 2. automatic_quantization_flow() -> Gera os modelos GGUF
def automatic_quantization_flow(original, params):
    quantized_paths = []
    for q in params:
        out_path = original.replace(".fp16.gguf", f"_{q}.gguf")
        if not os.path.exists(out_path):
            print(f"🛠 Quantizando para {q}...")
            subprocess.run(["./llama-quantize", original, out_path, q])
        quantized_paths.append(out_path)
    return quantized_paths

# Funções de Cálculo de Métricas (Linhas 13-17)
def calculate_metrics(raw_output, duration, ram_peak):
    # Parsing do stderr do llama.cpp para pegar tokens exatos
    # throughput = tokens / duration
    throughput = 15.5  # Exemplo: extrair do log 'eval rate'
    latency = (1 / throughput) * 1000 if throughput > 0 else 0
    
    return {
        "flops": "calculado via profiler ou especificação",
        "throughput": throughput,
        "latency": latency,
        "accuracy": 0.85, # Comparação com gabarito
        "mbu": (ram_peak / 12) * 100 # Memory Bandwidth Utilization (estimada)
    }

# --- EXECUÇÃO DO LOOP PRINCIPAL (Linhas 4-20) ---

all_models = automatic_quantization_flow(config["original_model"], config["quantization_params"])

i = 1
while i <= config["benchmark_params"]["iteration"]:
    print(f"\n🔄 Iteração {i}/{config['benchmark_params']['iteration']}")
    
    for model in all_models:
        print(f"🚀 Deploying: {model}")
        
        # 9-10. try: run_inference()
        try:
            start_time = time.perf_counter()
            
            # Comando de inferência (adapt_and_deploy_model + run_inference)
            process = subprocess.run([
                "./llama-cli", "-m", model, 
                "-p", "Sua questão aqui", 
                "-n", str(config["benchmark_params"]["n_predict"]),
                "-t", str(config["device_params"]["threads"])
            ], capture_output=True, text=True, timeout=300) # Timeout de 5 min
            
            end_time = time.perf_counter()
            
            # 13-17. calculate_metrics
            ram_p = psutil.Process().memory_info().rss / (1024**3)
            metrics = calculate_metrics(process.stdout, end_time - start_time, ram_p)
            
            # Salvar métricas para o MLflow
            print(f"📊 Metrics for {model}: {metrics['throughput']} tokens/s")

        # 11-12. except: "time out" or "memory overflow"
        except subprocess.TimeoutExpired:
            print(f"⚠️ Timeout no modelo {model}. Pulando...")
            continue
        except MemoryError:
            print(f"❌ OOM no modelo {model}. Pulando...")
            continue
            
    i += 1

print("🏁 Todos os modelos viáveis foram testados.")