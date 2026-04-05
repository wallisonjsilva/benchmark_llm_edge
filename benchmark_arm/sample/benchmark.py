import pandas as pd

def create_unified_benchmark_file():
    # Configurações de amostragem
    sampling_config = {
        "enem": {"n": 45, "path": "./dataset/enem/enem_data.parquet"},
        "poeta": {"n": 100, "path": "./dataset/poetav2/trivia_dataset_infos.parquet"},
        "bbq": {"n": 200, "path": "./dataset/bbq/bbq_data.parquet"}
    }
    
    test_samples = []

    for dataset_name, config in sampling_config.items():
        try:
            # Carrega o dataset original
            df = pd.read_parquet(config["path"])
            
            # Sorteio fixo para garantir reprodutibilidade
            sample = df.sample(n=config["n"], random_state=42)
            
            # Adiciona metadados para o script de inferência saber o que avaliar
            sample["source_dataset"] = dataset_name
            test_samples.append(sample)
            
            print(f"✅ {dataset_name.upper()}: {config['n']} amostras selecionadas.")
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")

    # Consolida tudo em um único arquivo de execução
    if test_samples:
        final_df = pd.concat(test_samples, ignore_index=True)
        final_df.to_parquet("benchmark_test_set.parquet")
        print("\n🚀 Arquivo 'benchmark_test_set.parquet' gerado com sucesso!")
        print(f"Total de perguntas para rodar por modelo: {len(final_df)}")

if __name__ == "__main__":
    create_unified_benchmark_file()