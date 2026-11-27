#!/usr/bin/env python3
"""Automate paired CUDA RoPE benchmarks at progressively shorter text lengths.

Run from the repository root:
    python scripts/benchmark_cuda_rope_scaling.py --input texto_120s.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import fmean

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CACHE = Path(
    "/mnt/46d6d246-1a5f-4388-b273-2ace8e54f409/"
    ".llms/.AUDIO/joe/.models/huggingface"
)
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from joe.audio_performance import format_cuda_rope_cache_benchmark_report  # noqa: E402
from joe.pocket_tts_models import default_model_variant  # noqa: E402


def _configure_hf_token(model_cache_directory: Path) -> str:
    """Reuse an existing Hugging Face token without printing or exposing it.

    HF_HOME is redirected to the Joe model cache later in the benchmark, so
    the Hub token may no longer be found at its usual user-cache location.
    """
    if os.environ.get("HF_TOKEN", "").strip():
        return "variável de ambiente"

    candidates = (
        ("cache de modelos", model_cache_directory.expanduser() / "token"),
        (
            "cache padrão do usuário",
            Path.home() / ".cache" / "huggingface" / "token",
        ),
    )
    checked: set[Path] = set()
    for source, token_path in candidates:
        try:
            resolved = token_path.expanduser().resolve()
            if resolved in checked:
                continue
            checked.add(resolved)
            if not resolved.is_file():
                continue
            token = resolved.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if token:
            os.environ["HF_TOKEN"] = token
            return source
    return ""


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving the original words and punctuation."""
    return re.sub(r"\s+", " ", str(text)).strip()


def reduce_text_to_duration(
    text: str,
    target_seconds: float,
    reference_seconds: float,
) -> str:
    """Create a shorter prefix, sized by measured full-text audio duration.

    Sentence boundaries are preferred. If a single sentence is too long for the
    remaining word budget, it is cut at a word boundary and closed with punctuation.
    """
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("O texto de entrada está vazio.")
    if target_seconds <= 0 or reference_seconds <= 0:
        raise ValueError("As durações de referência e alvo precisam ser positivas.")
    if target_seconds >= reference_seconds:
        return normalized

    words = normalized.split()
    if len(words) <= 1:
        return normalized

    fraction = min(1.0, target_seconds / reference_seconds)
    word_budget = max(1, min(len(words) - 1, math.ceil(len(words) * fraction)))
    sentences = re.split(r"(?<=[.!?])\s+", normalized)

    selected: list[str] = []
    selected_words = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        proposed_count = selected_words + len(sentence_words)
        if proposed_count <= word_budget:
            selected.append(sentence)
            selected_words = proposed_count
            continue

        remaining = word_budget - selected_words
        if remaining <= 0:
            break

        # Prefer keeping a complete sentence when the overshoot is small.
        overshoot = proposed_count - word_budget
        allowed_overshoot = max(2, math.ceil(word_budget * 0.05))
        if overshoot <= allowed_overshoot:
            selected.append(sentence)
        else:
            selected.append(" ".join(sentence_words[:remaining]))
        break

    shortened = " ".join(selected).strip()
    if not shortened:
        shortened = " ".join(words[:word_budget])
    if shortened[-1] not in ".!?…":
        shortened += "."
    return shortened


def measured_audio_duration(result) -> float:
    """Average the actual audio durations from both paired modes."""
    durations = (
        tuple(result.baseline_audio_durations_seconds)
        + tuple(result.cached_audio_durations_seconds)
    )
    return fmean(durations) if durations else 0.0


def _read_input(path_text: str) -> tuple[str, str]:
    if path_text == "-":
        content = sys.stdin.read()
        label = "stdin"
    else:
        path = Path(path_text).expanduser()
        content = path.read_text(encoding="utf-8-sig")
        label = str(path.resolve())
    content = normalize_text(content)
    if not content:
        raise ValueError("O arquivo de texto está vazio.")
    return content, label


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recebe um texto longo, mede primeiro o texto completo e reduz "
            "automaticamente versões para 30/60/120 segundos."
        )
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Arquivo .txt de entrada; use '-' para ler o texto pela entrada padrão.",
    )
    parser.add_argument("--language", default="portuguese")
    parser.add_argument(
        "--model-variant", default=None,
        help="Ex.: portuguese_24l. O padrão da língua é usado quando omitido.",
    )
    parser.add_argument(
        "--voice", default=None,
        help="Caminho do WAV/áudio de referência. O padrão do Pocket TTS é usado se omitido.",
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help=(
            "Pasta HF_HOME dos modelos. Padrão: "
            "/mnt/46d6d246-1a5f-4388-b273-2ace8e54f409/"
            ".llms/.AUDIO/joe/.models/huggingface"
        ),
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Gerações medidas por modo (3 para triagem; 7 para confirmação).",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--targets", type=float, nargs="+", default=[30.0, 60.0, 120.0],
        help="Durações-alvo em segundos. Padrão: 30 60 120.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Pasta de saída. Padrão: reports/rope-cache-scaling-DATA_HORA.",
    )
    return parser


def _write_outputs(
    output_dir: Path,
    source_label: str,
    language: str,
    model_variant: str,
    args,
    reference_audio_seconds: float,
    results: dict[float, dict[str, object]],
) -> None:
    ordered = [results[key] for key in sorted(results)]
    metadata = {
        "source": source_label,
        "language": language,
        "model_variant": model_variant,
        "voice": args.voice,
        "model_cache_directory": str(args.model_cache.expanduser().resolve()),
        "runs_per_mode": args.runs,
        "warmups_per_mode": args.warmups,
        "reference_full_text_audio_seconds": reference_audio_seconds,
        "results": ordered,
    }
    (output_dir / "resultados.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Benchmark automático — RoPE original × cache",
        "",
        f"- Texto de origem: {source_label}",
        f"- Idioma/modelo: {language} / {model_variant}",
        f"- Cache de modelos: {args.model_cache.expanduser().resolve()}",
        f"- Gerações medidas por modo: {args.runs}",
        f"- Aquecimentos descartados por modo: {args.warmups}",
        f"- Duração média medida do texto completo: {reference_audio_seconds:.2f} s",
        "",
        "Os cenários são executados do maior para o menor. A duração real do texto "
        "completo é usada para estimar a redução de palavras dos cenários menores. "
        "A duração final é medida pelo Pocket TTS em cada benchmark.",
        "",
    ]
    for entry in ordered:
        target = float(entry["target_seconds"])
        lines.extend(
            [
                f"## Alvo aproximado: {target:g} segundos",
                "",
                f"- Texto: {entry.get('text_file', 'não salvo')}",
                f"- Palavras/caracteres: {entry['text_words']} / {entry['text_characters']}",
                f"- Duração média real: {float(entry.get('actual_audio_seconds', 0.0)):.2f} s",
                "",
                str(entry.get("report", entry.get("error", "Sem relatório."))),
                "",
            ]
        )
    (output_dir / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.runs < 3:
        print("Erro: --runs precisa ser pelo menos 3.", file=sys.stderr)
        return 2
    if args.warmups < 1:
        print("Erro: --warmups precisa ser pelo menos 1.", file=sys.stderr)
        return 2
    if args.cpu_threads < 1:
        print("Erro: --cpu-threads precisa ser pelo menos 1.", file=sys.stderr)
        return 2
    if not args.targets or any(value <= 0 for value in args.targets):
        print("Erro: todos os valores de --targets precisam ser positivos.", file=sys.stderr)
        return 2
    if len(set(args.targets)) != len(args.targets):
        print("Erro: --targets não pode conter durações repetidas.", file=sys.stderr)
        return 2

    try:
        source_text, source_label = _read_input(args.input)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Erro ao ler o texto: {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else REPOSITORY_ROOT / "reports" / f"rope-cache-scaling-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    token_source = _configure_hf_token(args.model_cache)
    if token_source:
        print(
            f"[Hugging Face] Token carregado do {token_source}; "
            "o valor não será exibido.",
            flush=True,
        )
    else:
        print(
            "[Hugging Face] Nenhum token encontrado. O benchmark continuará "
            "sem autenticação, sujeito a limites menores do Hub.",
            flush=True,
        )

    language = str(args.language)
    model_variant = str(args.model_variant or default_model_variant(language))
    target_durations = sorted((float(value) for value in args.targets), reverse=True)

    # Import lazily: importing this script does not load a model or reserve VRAM.
    try:
        from joe.services.pocket_tts import PocketTTSService
    except ImportError as exc:
        print(
            "Não foi possível importar o serviço Pocket TTS. "
            "Instale as dependências do Joe antes de executar o benchmark.\n"
            f"Detalhes: {exc}",
            file=sys.stderr,
        )
        return 2

    service = PocketTTSService()
    results: dict[float, dict[str, object]] = {}
    reference_audio_seconds = 0.0
    failed = False

    try:
        for index, target_seconds in enumerate(target_durations):
            if index == 0:
                sample_text = source_text
                print(
                    f"\n[calibração] Testando o texto completo ({len(source_text)} caracteres) "
                    "para medir a duração real antes de reduzi-lo.",
                    flush=True,
                )
            else:
                sample_text = reduce_text_to_duration(
                    source_text,
                    target_seconds,
                    reference_audio_seconds,
                )
                print(
                    f"\n[alvo ~{target_seconds:g}s] Texto reduzido para "
                    f"{len(sample_text.split())} palavras / {len(sample_text)} caracteres.",
                    flush=True,
                )

            result = service.benchmark_cuda_rope_cache(
                sample_text,
                language,
                args.voice,
                model_variant,
                cpu_threads=args.cpu_threads,
                warmup_runs=args.warmups,
                measured_runs=args.runs,
                model_cache_dir=args.model_cache,
            )
            report = format_cuda_rope_cache_benchmark_report(result)
            actual_duration = measured_audio_duration(result)

            if result.error:
                failed = True
                print(report, file=sys.stderr)
                results[target_seconds] = {
                    "target_seconds": target_seconds,
                    "text_characters": len(sample_text),
                    "text_words": len(sample_text.split()),
                    "actual_audio_seconds": actual_duration,
                    "error": result.error,
                    "report": report,
                }
                break

            if index == 0:
                reference_audio_seconds = actual_duration
                if reference_audio_seconds <= 0:
                    raise RuntimeError(
                        "O Pocket TTS não retornou uma duração de áudio válida "
                        "para o texto completo."
                    )
                if reference_audio_seconds < target_seconds * 0.85:
                    print(
                        f"Aviso: o texto completo gerou em média "
                        f"{reference_audio_seconds:.1f}s, abaixo do alvo de "
                        f"{target_seconds:.0f}s. Os textos menores serão "
                        "dimensionados a partir dessa duração real.",
                        file=sys.stderr,
                        flush=True,
                    )

            safe_target = f"{target_seconds:06.1f}".replace(".", "p")
            text_file = output_dir / f"texto_{safe_target}s.txt"
            text_file.write_text(sample_text + "\n", encoding="utf-8")

            results[target_seconds] = {
                "target_seconds": target_seconds,
                "text_file": text_file.name,
                "text_characters": len(sample_text),
                "text_words": len(sample_text.split()),
                "actual_audio_seconds": actual_duration,
                "result": asdict(result),
                "report": report,
            }
            print(report, flush=True)
            _write_outputs(
                output_dir,
                source_label,
                language,
                model_variant,
                args,
                reference_audio_seconds,
                results,
            )
    except KeyboardInterrupt:
        print("\nBenchmark interrompido pelo usuário.", file=sys.stderr)
        failed = True
    except Exception as exc:  # noqa: BLE001
        print(f"\nFalha no benchmark: {type(exc).__name__}: {exc}", file=sys.stderr)
        failed = True
    finally:
        service.close()

    _write_outputs(
        output_dir,
        source_label,
        language,
        model_variant,
        args,
        reference_audio_seconds,
        results,
    )
    print(f"\nArquivos salvos em: {output_dir.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
