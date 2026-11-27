#!/usr/bin/env python3
"""Benchmark isolado de torch.compile/AMP para Pocket TTS.

Este script não altera as configurações do Joe. Ele compara a execução padrão
com uma configuração experimental e grava o áudio experimental opcionalmente.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _get_child(parent: Any, name: str, torch: Any) -> Any:
    if name.isdecimal() and isinstance(
        parent, (torch.nn.ModuleList, torch.nn.Sequential)
    ):
        return parent[int(name)]
    return getattr(parent, name)


def _set_child(parent: Any, name: str, value: Any, torch: Any) -> None:
    if name.isdecimal() and isinstance(
        parent, (torch.nn.ModuleList, torch.nn.Sequential)
    ):
        parent[int(name)] = value
    else:
        setattr(parent, name, value)


def _resolve_parent_module(root: Any, dotted_path: str, torch: Any) -> tuple[Any, str, Any]:
    """Resolve um submódulo sem depender de nomes fixos da arquitetura."""
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("Informe o caminho de um submódulo.")

    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part, torch)
    leaf = parts[-1]
    target = _get_child(parent, leaf, torch)
    if not isinstance(target, torch.nn.Module):
        raise TypeError(
            f"O caminho {dotted_path!r} não aponta para um torch.nn.Module."
        )
    return parent, leaf, target


def _list_modules(model: Any) -> None:
    """Mostra caminhos reais para escolher um submódulo a compilar."""
    modules = list(model.named_modules())
    print("Submódulos disponíveis para teste de torch.compile:")
    for name, module in modules:
        if not name:
            continue
        try:
            params = sum(
                int(parameter.numel())
                for parameter in module.parameters(recurse=False)
            )
        except Exception:  # noqa: BLE001
            params = 0
        print(f"  {name:<48} {type(module).__name__:<30} {params:>12,} parâmetros locais")


def _sync(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _set_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_context(torch: Any, device: str, precision: str):
    if precision == "none":
        return contextlib.nullcontext()
    if device != "cuda":
        raise ValueError("Autocast FP16/BF16 neste experimento exige CUDA.")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _run_case(
    *,
    label: str,
    args: argparse.Namespace,
    torch: Any,
    tts_model_class: Any,
    model_options: dict[str, Any],
    voice_source: str,
    experimental: bool,
) -> dict[str, Any]:
    """Carrega um modelo isolado e mede preparação de voz e geração."""
    quantize = args.device == "cpu" and args.cpu_int8
    started = time.perf_counter()
    model = tts_model_class.load_model(**model_options, quantize=quantize)
    if args.device == "cuda":
        model.to("cuda")
    if hasattr(model, "eval"):
        model.eval()
    _sync(torch, args.device)
    load_seconds = time.perf_counter() - started

    if experimental and args.compile_module:
        parent, leaf, target = _resolve_parent_module(
            model, args.compile_module, torch
        )
        compiled = torch.compile(target, mode=args.compile_mode)
        _set_child(parent, leaf, compiled, torch)

    previous_tf32 = None
    previous_cudnn_tf32 = None
    if experimental and args.tf32:
        if args.device != "cuda":
            raise ValueError("TF32 só pode ser testado com CUDA.")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    generation_times: list[float] = []
    preparation_times: list[float] = []
    first_call_seconds = 0.0
    last_samples: np.ndarray | None = None
    sample_rate = int(model.sample_rate)
    total_calls = args.warmup + args.repeats

    try:
        for index in range(total_calls):
            _set_seed(torch, args.seed)
            with torch.inference_mode():
                prepare_started = time.perf_counter()
                voice_state = model.get_state_for_audio_prompt(voice_source)
                _sync(torch, args.device)
                prepare_seconds = time.perf_counter() - prepare_started

                generation_started = time.perf_counter()
                with _amp_context(torch, args.device, args.autocast if experimental else "none"):
                    audio = model.generate_audio(voice_state, args.text)
                _sync(torch, args.device)
                generation_seconds = time.perf_counter() - generation_started

                if hasattr(audio, "detach"):
                    samples = audio.detach().to("cpu").float().numpy()
                else:
                    samples = np.asarray(audio, dtype=np.float32)
                samples = np.asarray(samples, dtype=np.float32)

            preparation_times.append(prepare_seconds)
            if index == 0:
                first_call_seconds = generation_seconds
            if index >= args.warmup:
                generation_times.append(generation_seconds)
                last_samples = samples.copy()

            del voice_state, audio

        if not generation_times or last_samples is None:
            raise RuntimeError("Nenhuma execução medida foi concluída.")

        duration = float(last_samples.shape[-1] / max(1, sample_rate))
        average_generation = float(sum(generation_times) / len(generation_times))
        peak_vram_mb = 0.0
        if args.device == "cuda":
            peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)

        return {
            "label": label,
            "device": args.device,
            "model_load_seconds": load_seconds,
            "voice_prepare_seconds": float(sum(preparation_times) / len(preparation_times)),
            "first_call_seconds": first_call_seconds,
            "generation_seconds_avg": average_generation,
            "audio_duration_seconds": duration,
            "realtime_factor": duration / average_generation if average_generation else 0.0,
            "peak_vram_mb": peak_vram_mb,
            "sample_rate": sample_rate,
            "sample_count": int(last_samples.size),
            "audio_samples": last_samples,
            "quantized_int8": quantize,
            "compile_module": args.compile_module if experimental else None,
            "compile_mode": args.compile_mode if experimental and args.compile_module else None,
            "autocast": args.autocast if experimental else "none",
            "tf32": bool(args.tf32 and experimental),
        }
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


def _quality_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if left.size != right.size:
        return {
            "same_sample_count": False,
            "reference_samples": int(left.size),
            "candidate_samples": int(right.size),
            "note": "As durações diferem; não foi calculada uma comparação amostra a amostra.",
        }

    difference = left - right
    signal_power = float(np.mean(np.square(left))) if left.size else 0.0
    noise_power = float(np.mean(np.square(difference))) if difference.size else 0.0
    snr_db = (
        10.0 * math.log10(signal_power / max(noise_power, 1e-20))
        if signal_power > 0
        else 0.0
    )
    return {
        "same_sample_count": True,
        "mean_absolute_difference": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "max_absolute_difference": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "snr_db": snr_db,
        "note": "SNR é uma métrica numérica; não substitui a audição do WAV.",
    }


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "audio_samples"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara Pocket TTS padrão com torch.compile/AMP de forma isolada."
    )
    parser.add_argument("--language", default="portuguese")
    parser.add_argument("--variant", default="portuguese")
    parser.add_argument("--voice", help="Áudio/estado de voz ou URI suportada pelo Pocket TTS.")
    parser.add_argument("--text", default="Olá! Este é um teste de desempenho do Pocket TTS no Joe.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-int8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-module", help="Caminho listado por --list-modules; compila somente esse módulo.")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--autocast", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--tf32", action="store_true", help="Habilita TF32 no caso experimental CUDA.")
    parser.add_argument("--list-modules", action="store_true", help="Lista módulos do modelo e encerra.")
    parser.add_argument("--output", type=Path, help="Salva o WAV do caso experimental.")
    parser.add_argument("--json-out", type=Path, help="Salva métricas em JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.text.strip() and not args.list_modules:
        raise SystemExit("--text não pode ficar vazio.")
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--threads e --repeats precisam ser positivos; --warmup não pode ser negativo.")
    if not args.list_modules and not (
        args.compile_module or args.autocast != "none" or args.tf32
    ):
        raise SystemExit(
            "Selecione --compile-module, --autocast fp16/bf16 ou --tf32 para definir o caso experimental."
        )

    try:
        import torch
        from pocket_tts import TTSModel
        from joe.model_cache import configure_model_cache
        from joe.pocket_tts_models import (
            default_voice_for_language,
            is_model_variant_for_language,
            model_loading_options,
        )
    except ImportError as exc:
        raise SystemExit(
            "Dependências ausentes. Ative o .venv do Joe e instale o projeto antes de executar."
        ) from exc

    if not is_model_variant_for_language(args.language, args.variant):
        raise SystemExit(
            f"A variante {args.variant!r} não é compatível com {args.language!r} no catálogo do Joe."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA foi selecionada, mas torch.cuda.is_available() retornou False.")
    if args.autocast != "none" and args.device != "cuda":
        raise SystemExit("FP16/BF16 autocast exige --device cuda.")
    if args.autocast == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("Esta GPU/build do PyTorch não informa suporte a BF16.")
    if args.tf32 and args.device != "cuda":
        raise SystemExit("--tf32 exige --device cuda.")

    torch.set_num_threads(args.threads)
    configure_model_cache()
    options = model_loading_options(args.variant)
    voice_source = args.voice or default_voice_for_language(args.language)
    if not voice_source:
        raise SystemExit("Nenhuma voz padrão disponível; informe --voice.")

    if args.list_modules:
        model = TTSModel.load_model(**options, quantize=args.device == "cpu" and args.cpu_int8)
        if args.device == "cuda":
            model.to("cuda")
        _list_modules(model)
        del model
        gc.collect()
        return 0

    print("Executando baseline oficial do Joe…", flush=True)
    baseline = _run_case(
        label="baseline",
        args=args,
        torch=torch,
        tts_model_class=TTSModel,
        model_options=options,
        voice_source=voice_source,
        experimental=False,
    )
    print("Executando configuração experimental…", flush=True)
    candidate = _run_case(
        label="experimental",
        args=args,
        torch=torch,
        tts_model_class=TTSModel,
        model_options=options,
        voice_source=voice_source,
        experimental=True,
    )

    speedup = (
        baseline["generation_seconds_avg"] / candidate["generation_seconds_avg"]
        if candidate["generation_seconds_avg"] > 0
        else 0.0
    )
    comparison = _quality_comparison(
        baseline["audio_samples"], candidate["audio_samples"]
    )
    report = {
        "language": args.language,
        "model_variant": args.variant,
        "text_characters": len(args.text),
        "voice": str(voice_source),
        "baseline": _public_result(baseline),
        "experimental": _public_result(candidate),
        "speedup_vs_baseline": speedup,
        "audio_comparison": comparison,
        "warning": (
            "Resultado de uma máquina/configuração específica. Não é benchmark oficial "
            "e não comprova equivalência perceptual."
        ),
    }

    for result in (baseline, candidate):
        print(
            f"{result['label']}: carga={result['model_load_seconds']:.2f}s; "
            f"voz={result['voice_prepare_seconds']:.2f}s; "
            f"primeira chamada={result['first_call_seconds']:.2f}s; "
            f"geração média={result['generation_seconds_avg']:.2f}s; "
            f"áudio={result['audio_duration_seconds']:.2f}s; "
            f"RTF={result['realtime_factor']:.3f}; "
            f"VRAM pico={result['peak_vram_mb']:.0f} MB"
        )
    print(f"Speedup experimental: {speedup:.3f}x")
    print(f"Comparação de áudio: {json.dumps(comparison, ensure_ascii=False)}")
    print(report["warning"])

    if args.output:
        import soundfile as sf

        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            args.output,
            candidate["audio_samples"].reshape(-1),
            candidate["sample_rate"],
            subtype="PCM_16",
        )
        print(f"WAV experimental salvo em: {args.output}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Métricas salvas em: {args.json_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        print(f"Falha no experimento: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
