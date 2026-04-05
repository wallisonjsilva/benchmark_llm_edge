#!/usr/bin/env python3
"""
notify_telegram.py — Envia uma mensagem de resumo no Telegram ao término do benchmark.

Uso direto:
    python notify_telegram.py --output-json output/saida_benchmark_poetav2.json

Como hook automático (chamado pelo benchmark_runner):
    Defina TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID nas variáveis de ambiente
    (ou no arquivo .env do backend) e o runner chamará este script automaticamente.

Variáveis de ambiente obrigatórias:
    TELEGRAM_BOT_TOKEN  — Token do bot gerado pelo @BotFather
    TELEGRAM_CHAT_ID    — ID do chat/grupo destino (pode ser negativo para grupos)

Variáveis de ambiente opcionais:
    TELEGRAM_DISABLE    — Se "1" ou "true", desativa o envio silenciosamente
    TELEGRAM_API_URL    — URL base da API (padrão: https://api.telegram.org)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers de leitura do JSON de saída
# ---------------------------------------------------------------------------

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_str(value: object, default: str = "—") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _format_metric(label: str, value: float, unit: str = "", *, fmt: str = ".2f") -> str:
    formatted = f"{value:{fmt}}"
    unit_str = f" {unit}" if unit else ""
    return f"• *{label}:* `{formatted}{unit_str}`"


# ---------------------------------------------------------------------------
# Montagem da mensagem
# ---------------------------------------------------------------------------

def _build_message(runs: list[dict], output_path: str) -> str:
    """Monta uma mensagem Markdown-compatible (parse_mode=Markdown) com o resumo."""
    lines: list[str] = []
    lines.append("✅ *Benchmark concluído!*")
    lines.append(f"📄 `{output_path}`")
    lines.append("")

    total_runs = len(runs)
    lines.append(f"*Modelos avaliados:* {total_runs}")
    lines.append("")

    for idx, run in enumerate(runs, start=1):
        model_name = _safe_str(run.get("model_name") or run.get("model"))
        backend = _safe_str(run.get("backend_name") or run.get("backend"), "cpu")
        experiment = _safe_str(run.get("experiment_name"), "")

        lines.append(f"━━━━━━━━━━━━━━━━━━")
        header = f"🤖 *Modelo {idx}/{total_runs}:* `{model_name}`"
        if experiment:
            header += f"\n🏷️ Experimento: `{experiment}`"
        lines.append(header)
        lines.append(f"⚙️ Backend: `{backend}`")
        lines.append("")

        # ── Métricas de throughput (llama-bench) ──
        bench = run.get("llama_bench") or run.get("bench") or {}
        gen_tps = _safe_float(bench.get("gen_tps"))
        prompt_tps = _safe_float(bench.get("prompt_tps"))
        ttft_ms = _safe_float(bench.get("ttft_ms"))
        tbt_ms = _safe_float(bench.get("tbt_ms"))
        peak_tps = _safe_float(bench.get("peak_gen_tps"))

        if gen_tps > 0 or prompt_tps > 0:
            lines.append("*⚡ Throughput (llama-bench):*")
            if gen_tps > 0:
                lines.append(_format_metric("Gen TPS (avg)", gen_tps, "tok/s"))
            if peak_tps > 0:
                lines.append(_format_metric("Gen TPS (peak)", peak_tps, "tok/s"))
            if prompt_tps > 0:
                lines.append(_format_metric("Prompt TPS", prompt_tps, "tok/s"))
            if ttft_ms > 0:
                lines.append(_format_metric("TTFT", ttft_ms, "ms"))
            if tbt_ms > 0:
                lines.append(_format_metric("TBT", tbt_ms, "ms"))
            lines.append("")

        # ── Métricas macro de acurácia ──
        macro = run.get("macro_metrics") or run.get("scores") or {}
        accuracy_keys = [k for k in macro if "accuracy" in k.lower() or "score" in k.lower()]
        if accuracy_keys:
            lines.append("*📊 Acurácia:*")
            for key in sorted(accuracy_keys):
                val = _safe_float(macro.get(key))
                label = key.replace("_", " ").title()
                lines.append(_format_metric(label, val * 100 if val <= 1.0 else val, "%"))
            lines.append("")

        # ── Métricas de inferência (perplexidade, TPS inline, etc.) ──
        inf = run.get("inference_metrics") or {}
        inf_tps = _safe_float(inf.get("tps") or inf.get("tokens_per_second"))
        perplexity = _safe_float(inf.get("perplexity"))
        success_rate = _safe_float(inf.get("inference_success_rate"))
        mbu = _safe_float(inf.get("mbu") or inf.get("memory_bandwidth_utilization"))

        if any(v > 0 for v in (inf_tps, perplexity, success_rate, mbu)):
            lines.append("*🔬 Inferência:*")
            if inf_tps > 0:
                lines.append(_format_metric("TPS (médio real)", inf_tps, "tok/s"))
            if perplexity > 0:
                lines.append(_format_metric("Perplexidade", perplexity, fmt=".3f"))
            if success_rate > 0:
                lines.append(_format_metric("Taxa de sucesso", success_rate * 100 if success_rate <= 1.0 else success_rate, "%"))
            if mbu > 0:
                lines.append(_format_metric("MBU", mbu * 100 if mbu <= 1.0 else mbu, "%"))
            lines.append("")

        # ── Hardware / temperatura ──
        hw = run.get("device_params") or run.get("hardware") or {}
        thermal = run.get("thermal") or {}
        cpu_temp = _safe_float(thermal.get("cpu_temp_c") or thermal.get("cpu_avg_temp"))
        gpu_temp = _safe_float(thermal.get("gpu_temp_c") or thermal.get("gpu_avg_temp"))
        peak_ram = _safe_float(run.get("peak_ram_gb"))
        peak_vram = _safe_float(run.get("peak_vram_gb"))
        gpu_name = _safe_str(hw.get("GPU") or hw.get("gpu_name"), "")

        hw_lines: list[str] = []
        if gpu_name and gpu_name not in ("CPU", "—"):
            hw_lines.append(f"• *GPU:* `{gpu_name}`")
        if peak_ram > 0:
            hw_lines.append(_format_metric("RAM pico", peak_ram, "GB"))
        if peak_vram > 0:
            hw_lines.append(_format_metric("VRAM pico", peak_vram, "GB"))
        if cpu_temp > 0:
            hw_lines.append(_format_metric("Temp CPU (avg)", cpu_temp, "°C", fmt=".1f"))
        if gpu_temp > 0:
            hw_lines.append(_format_metric("Temp GPU (avg)", gpu_temp, "°C", fmt=".1f"))
        if hw_lines:
            lines.append("*🖥️ Hardware:*")
            lines.extend(hw_lines)
            lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append("_Mensagem gerada automaticamente pelo benchmark runner._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Envio via Telegram Bot API
# ---------------------------------------------------------------------------

def send_telegram_message(
    text: str,
    bot_token: str,
    chat_id: str,
    *,
    api_url: str = "https://api.telegram.org",
    parse_mode: str = "Markdown",
    disable_web_page_preview: bool = True,
) -> bool:
    """
    Envia `text` para `chat_id` usando o `bot_token`.
    Retorna True em sucesso, False em falha (já imprime o erro).
    """
    endpoint = f"{api_url.rstrip('/')}/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            response_json = json.loads(body)
            if not response_json.get("ok"):
                print(f"[notify_telegram] Telegram API error: {response_json}", file=sys.stderr)
                return False
            return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[notify_telegram] HTTP {exc.code}: {body}", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[notify_telegram] Erro ao enviar mensagem: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Função principal reutilizável (importável pelo runner)
# ---------------------------------------------------------------------------

def notify_benchmark_done(
    runs: list[dict],
    output_path: str | Path = "",
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    api_url: str | None = None,
) -> bool:
    """
    Ponto de entrada para uso programático.
    Lê TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID do ambiente se não fornecidos.
    Retorna True se a mensagem foi enviada com sucesso.
    """
    disabled = os.getenv("TELEGRAM_DISABLE", "").strip().lower()
    if disabled in ("1", "true", "yes"):
        print("[notify_telegram] Notificações desativadas via TELEGRAM_DISABLE.", file=sys.stderr)
        return True  # Não é um erro, apenas silenciado

    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    cid = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    url = api_url or os.getenv("TELEGRAM_API_URL", "https://api.telegram.org").strip()

    if not token:
        print(
            "[notify_telegram] TELEGRAM_BOT_TOKEN não definido. Notificação ignorada.",
            file=sys.stderr,
        )
        return False
    if not cid:
        print(
            "[notify_telegram] TELEGRAM_CHAT_ID não definido. Notificação ignorada.",
            file=sys.stderr,
        )
        return False

    message = _build_message(runs, str(output_path))
    ok = send_telegram_message(message, token, cid, api_url=url)
    if ok:
        print("[notify_telegram] Mensagem enviada com sucesso para o Telegram.")
    return ok


# ---------------------------------------------------------------------------
# CLI standalone
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Envia resumo de benchmark para o Telegram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-json",
        default="",
        help=(
            "Caminho para o JSON de saída do benchmark "
            "(padrão: OUTPUT_JSON_PATH do ambiente ou output/saida_benchmark_poetav2.json)"
        ),
    )
    parser.add_argument("--bot-token", default="", help="Token do bot (sobrescreve TELEGRAM_BOT_TOKEN)")
    parser.add_argument("--chat-id", default="", help="Chat ID destino (sobrescreve TELEGRAM_CHAT_ID)")
    parser.add_argument("--message", default="", help="Mensagem customizada (ignora o JSON)")
    args = parser.parse_args(argv)

    # ── Mensagem customizada ──
    token = args.bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    cid = args.chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    api_url = os.getenv("TELEGRAM_API_URL", "https://api.telegram.org").strip()

    if not token:
        print("Erro: TELEGRAM_BOT_TOKEN não definido.", file=sys.stderr)
        return 1
    if not cid:
        print("Erro: TELEGRAM_CHAT_ID não definido.", file=sys.stderr)
        return 1

    if args.message:
        ok = send_telegram_message(args.message, token, cid, api_url=api_url)
        return 0 if ok else 1

    # ── Leitura do JSON ──
    json_path_str = (
        args.output_json
        or os.getenv("OUTPUT_JSON_PATH", "")
        or "output/saida_benchmark_poetav2.json"
    )
    json_path = Path(json_path_str)
    if not json_path.is_absolute():
        # Resolve relativo à raiz do projeto (dois níveis acima de scripts/)
        project_root = Path(__file__).resolve().parents[2]
        json_path = project_root / json_path_str

    if not json_path.exists():
        print(f"Erro: arquivo não encontrado: {json_path}", file=sys.stderr)
        return 1

    try:
        runs = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Erro ao parsear JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(runs, list):
        runs = [runs]

    ok = notify_benchmark_done(runs, json_path, bot_token=token, chat_id=cid, api_url=api_url)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
