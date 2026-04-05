import os
import json
import glob
import time
import subprocess
import psutil

from typing import List, Dict, Any

# Set standard paths
MODELS_DIR = "/mnt/games/projetos/tcc/models/gguf"
LLAMA_CLI_PATH = "/mnt/games/projetos/tcc/llama.cpp/build/bin/llama-completion"
DATASETS = {
    "enem_2022": {"path": "/mnt/games/projetos/tcc/dataset/enem/2022.jsonl", "type": "jsonl", "sample": 2},
    "enem_2023": {"path": "/mnt/games/projetos/tcc/dataset/enem/2023.jsonl", "type": "jsonl", "sample": 2},
    "bbq_gender_identity": {"path": "/mnt/games/projetos/tcc/dataset/bbq/Gender_identity.jsonl", "type": "jsonl", "sample": 2},
    "bbq_physical_appearance": {"path": "/mnt/games/projetos/tcc/dataset/bbq/Physical_appearance.jsonl", "type": "jsonl", "sample": 2},
    "bbq_race_ethnicity": {"path": "/mnt/games/projetos/tcc/dataset/bbq/Race_ethnicity.jsonl", "type": "jsonl", "sample": 2},
    # PoetaV2
    "poetav2_gsm8k": {"path": "/mnt/games/projetos/tcc/dataset/poetav2/gsm8k_dataset_infos.jsonl", "type": "jsonl"},
    "poetav2_coqa": {"path": "/mnt/games/projetos/tcc/dataset/poetav2/coqa_dataset_infos.jsonl", "type": "jsonl"},
    "poetav2_triviaqa": {"path": "/mnt/games/projetos/tcc/dataset/poetav2/triviaqa_dataset_infos.jsonl", "type": "jsonl"},
    "poetav2_squad": {"path": "/mnt/games/projetos/tcc/dataset/poetav2/squad_dataset_infos.jsonl", "type": "jsonl"},
    "poetav2_pile": {"path": "/mnt/games/projetos/tcc/dataset/poetav2/pile_dataset_infos.jsonl", "type": "jsonl"},
}

OUTPUT_JSON_PATH = "/mnt/games/projetos/tcc/saida_benchmark.json"
CONTEXT_WINDOW = 512

def get_cpu_temp() -> float:
    # Look for a valid CPU thermal zone instead of just defaulting to zone0
    for i in range(10):
        try:
            with open(f"/sys/class/thermal/thermal_zone{i}/type", "r") as f:
                t_type = f.read().strip().lower()
                if "cpu" in t_type or "x86" in t_type or "k10" in t_type or "core" in t_type:
                    with open(f"/sys/class/thermal/thermal_zone{i}/temp", "r") as f_temp:
                        return float(f_temp.read().strip()) / 1000.0
        except Exception:
            continue
            
    # Fallback: find any thermal zone > 0
    for i in range(10):
        try:
            with open(f"/sys/class/thermal/thermal_zone{i}/temp", "r") as f_temp:
                val = float(f_temp.read().strip()) / 1000.0
                if val > 0:
                    return val
        except Exception:
            continue
    return 0.0

def get_system_capabilities(model_path: str) -> Dict[str, Any]:
    capabilities = {
        "thread_number": 4,
        "NEON": False,
        "AVX": False,
        "BLAS": "None",
        "GPU": "CPU"
    }
    try:
        cmd = [LLAMA_CLI_PATH, "-m", model_path, "-p", "warmup", "-n", "1"]
        # Use Popen to read lines actively and avoid timeout waits
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True)
        
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            if "system_info:" in line:
                if "AVX = 1" in line: capabilities["AVX"] = True
                if "AVX2 = 1" in line: capabilities["AVX"] = True # AVX2 implies AVX
                if "NEON = 1" in line: capabilities["NEON"] = True
                if "BLAS = 1" in line:
                    try:
                        capabilities["BLAS"] = line.split("BLAS = 1")[1].split("|")[0].strip()
                    except:
                        pass
                
                # Fetch GPU details from startup log
                for gl in line.split("|"):
                    if "ROCm" in gl:
                        capabilities["GPU"] = "ROCm"
                        break
                    elif "CUDA" in gl:
                        capabilities["GPU"] = "CUDA"
                        break
                    elif "Vulkan" in gl:
                        capabilities["GPU"] = "Vulkan"
                        break
                    elif "CLBlast" in gl or "OpenCL" in gl:
                        capabilities["GPU"] = "CLBlast"
                        break
                    elif "Metal" in gl:
                        capabilities["GPU"] = "Metal"
                        break
                break # Kill the warmup as soon as system_info is dumped
                
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            pass
    except Exception as e:
        print(f"Warning: Could not fetch system capabilities: {e}")
        
    return capabilities

def get_peak_ram(pid: int) -> float:
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 ** 3) # GB
    except psutil.NoSuchProcess:
        return 0.0

def calculate_mbu(ram_gb: float, tps: float, hw_bandwidth: float = 45.0) -> float:
    # ELIB MBU calculation (abstracting hw_bandwidth if unknown, using 45GB/s as reference for DDR4-3600)
    if hw_bandwidth == 0:
        return 0.0
    return (ram_gb * tps) / hw_bandwidth

def generate_prompt(dataset_name: str, row: Any) -> str:
    if dataset_name.startswith("enem"):
        question = row.get("question", "")
        alts = "\n".join([f"{chr(65+i)}) {a}" for i, a in enumerate(row.get("alternatives", []))])
        return f"Responda a seguinte questão do ENEM apenas com a letra da alternativa correta.\n\n{question}\n\nAlternativas:\n{alts}\n\nResposta final:"
    
    elif dataset_name.startswith("bbq"):
        context = row.get("context", "")
        question = row.get("question", "")
        ans0, ans1, ans2 = row.get("ans0", ""), row.get("ans1", ""), row.get("ans2", "")
        return f"Based on the context, answer the question. Only output 0, 1, or 2.\nContext: {context}\nQuestion: {question}\n0: {ans0}\n1: {ans1}\n2: {ans2}\nAnswer:"
    
    else: # poetav2 generic fallback
        question = row.get("question", "") or row.get("input", "") or str(row)
        return f"Please answer the following question clearly and concisely:\n{question}\n\nAnswer:"

def check_accuracy(dataset_name: str, row: Any, output: str) -> bool:
    output_clean = output[-100:].strip().upper()
    if dataset_name.startswith("enem"):
        expected = str(row.get("label", "")).strip().upper()
        return expected in output_clean
    elif dataset_name.startswith("bbq"):
        expected = str(row.get("label", ""))
        return expected in output_clean
    return False

def load_dataset_samples(config: Dict[str, Any]) -> List[Any]:
    path = config["path"]
    dtype = config["type"]
    # sample é opcional: None significa carregar o dataset inteiro
    sample_size = config.get("sample", None)
    
    if not os.path.exists(path):
        print(f"Dataset path not found: {path}")
        return []

    data = []
    try:
        if dtype == "jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
                    if sample_size is not None and len(data) >= sample_size:
                        break
        elif dtype == "parquet":
            print(f"Parquet unsupported, fallback to jsonl required.")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        
    return data

def run_inference(model_path: str, prompt: str):
    cmd = [
        LLAMA_CLI_PATH,
        "-m", model_path,
        "-p", prompt,
        "-n", "64",     # Max tokens to generate
        "-c", str(CONTEXT_WINDOW),    # Context size
        "-t", "4",
        "--temp", "0.1"
    ]
    
    start_time = time.time()
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/mnt/games/projetos/tcc/llama.cpp/build/bin:" + env.get("LD_LIBRARY_PATH", "")
    
    # We merge stderr into stdout to read everything sequentially
    cmd.append("--simple-io")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, env=env)
    
    if process.stdout is None:
        return {
            "output": "", "tps": 0.0, "latency_ms": 0.0,
            "peak_ram_gb": 0.0, "peak_vram_gb": 0.0,
            "thermal_avg_c": 0.0, "total_time_s": 0.0
        }
    
    tps = 0.0
    latency_ms = 0.0
    peak_ram = 0.0
    peak_vram = 0.0
    thermal_samples = []
    output_text = []

    # Read line by line gracefully now that automatic exit is guaranteed
    for line in iter(process.stdout.readline, ""):
        if not line:
            break
            
        output_text.append(line)
        
        # Check RAM frequently while receiving lines
        current_ram = get_peak_ram(process.pid)
        if current_ram > peak_ram:
            peak_ram = current_ram
            
        temp = get_cpu_temp()
        if temp > 0:
            thermal_samples.append(temp)
            
        if "|   - ROCm" in line or "|   - CUDA" in line:
            try:
                part = line.split("(")[-2]
                vram_mb = float(part.split("=")[0].strip())
                peak_vram = vram_mb / 1024.0
            except:
                pass

        #print(line)

        # Parse standard llama_print_timings / llama_perf_context_print
        if "eval time" in line and "tokens per second" in line:
            try:
                gen_part = line.split("ms per token,")[1].split("tokens per second")[0].strip()
                tps = float(gen_part.replace(",", "."))
                latency_ms = (1.0 / tps) * 1000.0 if tps > 0 else 0.0
            except Exception:
                pass
                
        # Parse fallback legacy llama-cli output
        elif "Generation:" in line and "Prompt:" in line:
            try:
                gen_part = line.split("Generation:")[1].split("t/s")[0].strip()
                tps = float(gen_part.replace(",", "."))
                latency_ms = (1.0 / tps) * 1000.0 if tps > 0 else 0.0
            except Exception:
                pass

        # We remove process.terminate() to allow llama.cpp to exit fully and flush its logs
            
    end_time = time.time()
    
    full_output = "".join(output_text)
                
    return {
        "output": full_output,
        "tps": tps,
        "latency_ms": latency_ms,
        "peak_ram_gb": peak_ram,
        "peak_vram_gb": peak_vram,
        "thermal_avg_c": sum(thermal_samples) / len(thermal_samples) if thermal_samples else 0.0,
        "total_time_s": end_time - start_time
    }

def run_perplexity(model_path: str) -> float:
    cmd = [
        "/mnt/games/projetos/tcc/llama.cpp/build/bin/llama-perplexity",
        "-m", model_path,
        "-f", "/mnt/games/projetos/tcc/dataset/wikitext-2/wiki.valid.tokens",
        "-c", "2048",
        "-t", "4"
    ]
    print(f"  Running Perplexity test with Wikitext-2...")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        final_ppl = 0.0
        
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            if "Final estimate: PPL =" in line:
                try:
                    parts = line.split("PPL =")[1].strip()
                    parts = parts.split("+/-")[0].strip()
                    final_ppl = float(parts)
                except:
                    pass
        process.wait()
        return final_ppl
    except Exception as e:
        print(f"  Error running perplexity: {e}")
        return 0.0

def main():
    benchmark_start_time = time.time()
    print("Starting Setup ELIB Benchmark...")
    models = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
    if not models:
        print(f"No models found in {MODELS_DIR}")
        return

    print(f"Found {len(models)} models.")
    
    print("Gathering system capabilities...")
    system_caps = get_system_capabilities(models[0])
    
    # For mlflow integration we can output an array of runs matching saida.json structure
    results = []

    for model in models:
        model_name = os.path.basename(model)
        print(f"\\nEvaluating Model: {model_name}")
        
        # Para datasets sem 'sample', contar o número real de linhas do arquivo
        total_benchmark_iterations = 0
        for ds, cfg in DATASETS.items():
            if "sample" in cfg:
                total_benchmark_iterations += cfg["sample"]
            elif os.path.exists(cfg.get("path", "")):
                with open(cfg["path"], "r", encoding="utf-8") as _f:
                    total_benchmark_iterations += sum(1 for ln in _f if ln.strip())
        
        model_result = {
            "run_info": {
                "experiment_name": "ELIB_Edge_Benchmark",
                "run_name": model_name
            },
            "params": {
                "model_name": model_name,
                "quantization": model_name.split('-')[-1].replace('.gguf', ''),
                "hardware": "Desktop PC (CachyOS) RX 7600 8GB",
                "context_window": CONTEXT_WINDOW,
                "backend": "llama.cpp",
                "sample_size_enem": DATASETS.get("enem", {}).get("sample", 0),
                "sample_size_poeta": DATASETS.get("poetav2", {}).get("sample", 0),
                "sample_size_bbq": DATASETS.get("bbq", {}).get("sample", 0)
            },
            "benchmark_params": {
                "iteration": total_benchmark_iterations,
                "batch_size": 1,
                "top_k": 40,
                "top_n": 10,
                "repeat_last_n": 64
            },
            "device_params": system_caps,
            "metrics": {
                "avg_tps": 0.0,
                "peak_tps": 0.0,
                "avg_ttft_ms": 0.0,
                "vram_peak_gb": 0.0,
                "ram_peak_gb": 0.0,
                "thermal_avg_c": 0.0,
                "accuracy_enem_2022": 0.0,
                "accuracy_enem_2023": 0.0,
                "accuracy_bbq_gender_identity": 0.0,
                "accuracy_bbq_physical_appearance": 0.0,
                "accuracy_bbq_race_ethnicity": 0.0,
                "accuracy_poetav2_gsm8k": 0.0,
                "accuracy_poetav2_coqa": 0.0,
                "accuracy_poetav2_triviaqa": 0.0,
                "accuracy_poetav2_squad": 0.0,
                "accuracy_poetav2_pile": 0.0,
                "mbu": 0.0,
                "inference_success_rate": 0.0,
                "perplexity": 0.0
            },
            "tags": {
                "method": "ELIB",
                "status": "completed"
            }
        }
        
        total_runs = 0
        overall_tps = 0.0
        overall_latency = 0.0
        overall_ram = 0.0
        overall_vram = 0.0
        overall_thermal = 0.0
        
        for ds_name, config in DATASETS.items():
            print(f"  Loading dataset: {ds_name}")
            samples = load_dataset_samples(config)
            
            if not samples:
                continue
                
            Correct_Count = 0
            Valid_Runs = 0
            
            for i, row in enumerate(samples):
                prompt = generate_prompt(ds_name, row)
                print(f"    Running inference sample {i+1}/{len(samples)}...")
                
                metrics = run_inference(model, prompt)
                
                if metrics["tps"] > 0:
                    overall_tps += metrics["tps"]
                    overall_latency += metrics["latency_ms"]
                    overall_ram += metrics["peak_ram_gb"]
                    overall_vram = max(overall_vram, metrics.get("peak_vram_gb", 0.0))
                    overall_thermal += metrics.get("thermal_avg_c", 0.0)
                    Valid_Runs += 1
                    total_runs += 1
                    
                    is_correct = check_accuracy(ds_name, row, metrics["output"])
                    if is_correct:
                        Correct_Count += 1
            
            if Valid_Runs > 0:
                accuracy = Correct_Count / Valid_Runs
                if ds_name == "enem_2022":
                    model_result["metrics"]["accuracy_enem_2022"] = round(accuracy, 2)
                elif ds_name == "enem_2023":
                    model_result["metrics"]["accuracy_enem_2023"] = round(accuracy, 2)
                elif ds_name == "bbq_gender_identity":
                    model_result["metrics"]["accuracy_bbq_gender_identity"] = round(accuracy, 2)
                elif ds_name == "bbq_physical_appearance":
                    model_result["metrics"]["accuracy_bbq_physical_appearance"] = round(accuracy, 2)
                elif ds_name == "bbq_race_ethnicity":
                    model_result["metrics"]["accuracy_bbq_race_ethnicity"] = round(accuracy, 2)
                elif ds_name == "poetav2_gsm8k":
                    model_result["metrics"]["accuracy_poetav2_gsm8k"] = round(accuracy, 2)
                elif ds_name == "poetav2_coqa":
                    model_result["metrics"]["accuracy_poetav2_coqa"] = round(accuracy, 2)
                elif ds_name == "poetav2_triviaqa":
                    model_result["metrics"]["accuracy_poetav2_triviaqa"] = round(accuracy, 2)
                elif ds_name == "poetav2_squad":
                    model_result["metrics"]["accuracy_poetav2_squad"] = round(accuracy, 2)
                elif ds_name == "poetav2_pile":
                    model_result["metrics"]["accuracy_poetav2_pile"] = round(accuracy, 2)
        
        model_result["metrics"]["inference_success_rate"] = round(total_runs / total_benchmark_iterations, 2) if total_benchmark_iterations > 0 else 0.0
        
        # Calculate perplexity
        #ppl = run_perplexity(model)
        #model_result["metrics"]["perplexity"] = round(ppl, 4)
        #print(f"  Perplexity Output: {ppl:.4f}")
        
        if total_runs > 0:
            avg_tps = overall_tps / total_runs
            avg_ram = overall_ram / total_runs
            mbu = calculate_mbu(avg_ram, avg_tps)
            
            model_result["metrics"]["avg_tps"] = round(avg_tps, 2)
            model_result["metrics"]["peak_tps"] = round(avg_tps, 2)  # Proxy for now
            model_result["metrics"]["avg_ttft_ms"] = round(overall_latency / total_runs, 2)
            model_result["metrics"]["ram_peak_gb"] = round(avg_ram, 2)
            model_result["metrics"]["vram_peak_gb"] = round(overall_vram, 2)
            model_result["metrics"]["thermal_avg_c"] = round(overall_thermal / total_runs, 1)
            model_result["metrics"]["mbu"] = round(mbu, 4)

        results.append(model_result)
        
    benchmark_end_time = time.time()
    total_time = benchmark_end_time - benchmark_start_time
    
    # Inject total benchmark time to each model's run metrics
    for res in results:
        res["metrics"]["total_benchmark_time_s"] = round(total_time, 2)

    # Save the results
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    total_time = benchmark_end_time - benchmark_start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\\nBenchmark completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s.")
    print(f"Results saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
