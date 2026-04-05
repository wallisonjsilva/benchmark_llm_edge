Tradução Técnica das Métricas ELIB

Para o seu TCC, o algoritmo pede métricas específicas que o llama.cpp fornece no stderr. Veja como mapear:

Throughput (TPS): O llama.cpp reporta como eval rate. É o número de tokens gerados por segundo.

Latency: No contexto ELIB, é o tempo médio entre tokens (ms per token). No log do llama: t/token.

Accuracy (Perplexity): O algoritmo da imagem cita calculate_perplexity. No seu caso, você usará a Acurácia (Match com Gabarito) para ENEM/BBQ, o que é mais prático para avaliar o trade-off de inteligência.

MBU (Memory Bandwidth Utilization): Esta é uma métrica de eficiência de hardware. Ela é calculada pela fórmula:
MBU = Bandwidth Maxima do Hardware / Bandwidth Consumida (Model Size × TPS)

Exemplo S24: Se a RAM do S24 tem 60 GB/s e o seu modelo Q4 (5GB) roda a 10 TPS, você está usando 50/60=83% da banda.
