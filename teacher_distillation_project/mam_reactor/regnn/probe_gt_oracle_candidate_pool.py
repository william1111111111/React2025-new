from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from dataset.react_2025 import ReactionDataset, custom_collate
from framework.metrics.FRC import concordance_correlation_coefficient
from framework.modules.post_processor import Processor
from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig
from regnn.emotion_query_mamba import (
    EmotionQueryMambaConfig,
    EmotionQueryResidualMamba,
)
from regnn.eval_conditional_regnn_official_test import (
    chunk_offline_stream,
    initialize_official_hydra_runtime,
)
from regnn.query_mamba import apply_residual_channel_multipliers
from tools.compute_frd_resumable import GROUPS, rolling_dtw


@dataclass(frozen=True)
class CandidateVariant:
    name: str
    residual_scale: float
    style_residual_scale: float
    take_queries: tuple[int, ...]
    gate_value: float | None = None
    mixture_weights: tuple[float, ...] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-split GT-oracle feasibility probe for quality-constrained "
            "Mam-Reactor candidate selection. Not a deployable/test evaluator."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--session-split-manifest",
        type=Path,
        default=Path("regnn/configs/eqr_session_splits_seed1.json"),
    )
    parser.add_argument("--split-name", choices=("B_fit", "B_cal"), default="B_fit")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=750)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--score-workers", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--baseline-residual-scale", type=float, default=0.95)
    parser.add_argument("--baseline-style-scale", type=float, default=0.95)
    parser.add_argument("--au-residual-multiplier", type=float, default=1.8)
    parser.add_argument("--va-residual-multiplier", type=float, default=0.4)
    parser.add_argument("--expression-residual-multiplier", type=float, default=1.6)
    parser.add_argument("--frc-slack", type=float, default=0.0)
    parser.add_argument("--frd-slack", type=float, default=0.0)
    parser.add_argument("--include-mixture-sweep", action="store_true")
    parser.add_argument("--disable-two-swap", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positives = ("clip_length", "clip_batch_size", "score_workers", "max_examples")
    for name in positives:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "baseline_residual_scale",
        "baseline_style_scale",
        "au_residual_multiplier",
        "va_residual_multiplier",
        "expression_residual_multiplier",
        "frc_slack",
        "frd_slack",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def load_session_allowlist(path: Path, split_name: str) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = payload.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        raise ValueError(f"manifest has no split {split_name}")
    values = splits[split_name]
    if not isinstance(values, list) or not values:
        raise ValueError(f"manifest split {split_name} is empty")
    return {str(value) for value in values}


def stable_rank(seed: int, clip_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{clip_id}".encode("utf-8")).digest()


def select_probe_indices(
    dataset: ReactionDataset,
    allowed_sessions: set[str],
    max_examples: int,
    seed: int,
) -> list[int]:
    eligible = [
        index
        for index, path in enumerate(dataset.speaker_path_list)
        if path.parts[1] in allowed_sessions
    ]
    if not eligible:
        raise RuntimeError("session allowlist selected zero train examples")
    eligible.sort(
        key=lambda index: stable_rank(
            seed, dataset.speaker_path_list[index].as_posix()
        )
    )
    return eligible[: min(max_examples, len(eligible))]


def build_candidate_variants(
    baseline_residual_scale: float,
    baseline_style_scale: float,
    include_mixture_sweep: bool,
) -> list[CandidateVariant]:
    variants = [
        CandidateVariant(
            "baseline",
            baseline_residual_scale,
            baseline_style_scale,
            tuple(range(10)),
        ),
        CandidateVariant(
            "dynamic_x0.80",
            0.80 * baseline_residual_scale,
            baseline_style_scale,
            tuple(range(1, 10)),
        ),
        CandidateVariant(
            "dynamic_x1.20",
            1.20 * baseline_residual_scale,
            baseline_style_scale,
            tuple(range(1, 10)),
        ),
        CandidateVariant(
            "style_x0.75",
            baseline_residual_scale,
            0.75 * baseline_style_scale,
            tuple(range(1, 10)),
        ),
        CandidateVariant(
            "style_x1.25",
            baseline_residual_scale,
            1.25 * baseline_style_scale,
            tuple(range(1, 10)),
        ),
        CandidateVariant(
            "gate_const_0.75",
            baseline_residual_scale,
            baseline_style_scale,
            tuple(range(1, 9)),
            gate_value=0.75,
        ),
        CandidateVariant(
            "gate_const_1.00",
            baseline_residual_scale,
            baseline_style_scale,
            tuple(range(1, 9)),
            gate_value=1.0,
        ),
    ]
    if include_mixture_sweep:
        variants.append(
            CandidateVariant(
                "q9_uniform",
                baseline_residual_scale,
                baseline_style_scale,
                (9,),
                mixture_weights=tuple([1.0 / 8.0] * 8),
            )
        )
        for emotion_index in range(8):
            weights = [0.0] * 8
            weights[emotion_index] = 1.0
            variants.append(
                CandidateVariant(
                    f"q9_onehot_e{emotion_index}",
                    baseline_residual_scale,
                    baseline_style_scale,
                    (9,),
                    mixture_weights=tuple(weights),
                )
            )
    return variants


def prepare_chunks(
    audio: torch.Tensor,
    emotion: torch.Tensor,
    face: torch.Tensor,
    raw_length: int,
    clip_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    audio_chunks, lengths = chunk_offline_stream(audio, raw_length, clip_length)
    emotion_chunks, emotion_lengths = chunk_offline_stream(
        emotion, raw_length, clip_length
    )
    face_chunks, face_lengths = chunk_offline_stream(face, raw_length, clip_length)
    if not (
        torch.equal(lengths, emotion_lengths)
        and torch.equal(lengths, face_lengths)
    ):
        raise RuntimeError("audio/emotion/3DMM chunk lengths disagree")
    return audio_chunks, emotion_chunks, face_chunks, lengths


def generate_variant(
    model: EmotionQueryResidualMamba,
    audio_chunks: torch.Tensor,
    emotion_chunks: torch.Tensor,
    face_chunks: torch.Tensor,
    lengths: torch.Tensor,
    raw_length: int,
    variant: CandidateVariant,
    device: torch.device,
    clip_batch_size: int,
    channel_multipliers: Sequence[float],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, audio_chunks.shape[0], clip_batch_size):
            stop = min(start + clip_batch_size, audio_chunks.shape[0])
            count = stop - start
            gate_override = None
            if variant.gate_value is not None:
                gate_override = torch.full(
                    (count, 8),
                    float(variant.gate_value),
                    device=device,
                    dtype=torch.float32,
                )
            mixture_override = None
            if variant.mixture_weights is not None:
                mixture_override = (
                    torch.tensor(
                        variant.mixture_weights,
                        device=device,
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .expand(count, -1)
                )
            output = model(
                audio_chunks[start:stop].to(device, non_blocking=True),
                emotion_chunks[start:stop].to(device, non_blocking=True),
                face_chunks[start:stop].to(device, non_blocking=True),
                lengths[start:stop].to(device, non_blocking=True),
                residual_scale=variant.residual_scale,
                style_residual_scale=variant.style_residual_scale,
                emotion_query_gates_override=gate_override,
                mixture_weights_override=mixture_override,
            )
            prediction = apply_residual_channel_multipliers(
                model, output, channel_multipliers
            )
            pieces.append(prediction.detach().float().cpu())

    chunks = torch.cat(pieces, dim=0)
    full = (
        chunks.permute(1, 0, 2, 3)
        .reshape(10, -1, 25)[:, :raw_length]
        .contiguous()
    )
    # Official offline path rounds AUs before the metrics used here.
    full[:, :, :15] = torch.round(full[:, :, :15])
    return full


def candidate_digest(candidate: torch.Tensor) -> str:
    array = candidate.detach().cpu().float().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def generate_candidate_pool(
    model: EmotionQueryResidualMamba,
    speaker_audio: torch.Tensor,
    speaker_emotion: torch.Tensor,
    speaker_3dmm: torch.Tensor,
    raw_length: int,
    variants: Sequence[CandidateVariant],
    device: torch.device,
    clip_length: int,
    clip_batch_size: int,
    channel_multipliers: Sequence[float],
) -> tuple[list[torch.Tensor], list[dict]]:
    audio_chunks, emotion_chunks, face_chunks, lengths = prepare_chunks(
        speaker_audio,
        speaker_emotion,
        speaker_3dmm,
        raw_length,
        clip_length,
    )

    candidates: list[torch.Tensor] = []
    metadata: list[dict] = []
    seen_nonbaseline: set[str] = set()

    for variant in variants:
        full = generate_variant(
            model,
            audio_chunks,
            emotion_chunks,
            face_chunks,
            lengths,
            raw_length,
            variant,
            device,
            clip_batch_size,
            channel_multipliers,
        )
        for query_index in variant.take_queries:
            candidate = full[query_index].contiguous()
            digest = candidate_digest(candidate)

            # Preserve the ten baseline slots even if two baseline trajectories
            # happen to be exactly equal. Deduplicate only extra pool entries.
            if variant.name != "baseline" and digest in seen_nonbaseline:
                continue

            candidates.append(candidate)
            metadata.append(
                {
                    "id": f"{variant.name}/q{query_index}",
                    "variant": variant.name,
                    "slot": int(query_index),
                    "query_index": int(query_index),
                }
            )
            seen_nonbaseline.add(digest)

    id_to_index = {item["id"]: i for i, item in enumerate(metadata)}
    missing = [
        f"baseline/q{slot}"
        for slot in range(10)
        if f"baseline/q{slot}" not in id_to_index
    ]
    if missing:
        raise RuntimeError(f"candidate pool lost baseline slots: {missing}")
    if len(candidates) <= 10:
        raise RuntimeError("candidate pool did not expand beyond the baseline")
    return candidates, metadata


def score_one_candidate(
    candidate: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    prediction = np.ascontiguousarray(
        candidate.detach().cpu().float().numpy(), dtype=np.float32
    )
    target_array = np.ascontiguousarray(
        targets.detach().cpu().float().numpy(), dtype=np.float32
    )
    ccc_values: list[float] = []
    best_frd = math.inf
    for target in target_array:
        ccc, _ = concordance_correlation_coefficient(target, prediction)
        ccc_values.append(float(ccc))

        total = 0.0
        for start, end, weight, _ in GROUPS:
            total += float(weight) * float(
                rolling_dtw(
                    np.ascontiguousarray(prediction[:, start:end]),
                    np.ascontiguousarray(target[:, start:end]),
                )
            )
            # Exact nonnegative pruning: remaining group contributions cannot
            # make this target beat the current best.
            if total >= best_frd:
                break
        best_frd = min(best_frd, total)

    return max(ccc_values), float(best_frd)


def score_candidate_pool(
    candidates: Sequence[torch.Tensor],
    targets: torch.Tensor,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if targets.ndim != 3 or targets.shape[0] != 10 or targets.shape[-1] != 25:
        raise ValueError(f"targets must be [10,T,25], got {tuple(targets.shape)}")
    if any(tuple(candidate.shape) != tuple(targets.shape[1:]) for candidate in candidates):
        raise ValueError("candidate and target [T,25] shapes disagree")

    # Compile numba before worker threads touch the kernel.
    warm = np.zeros((2, 1), dtype=np.float32)
    rolling_dtw(warm, warm)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        scores = list(
            executor.map(
                lambda candidate: score_one_candidate(candidate, targets),
                candidates,
            )
        )
    frc = np.asarray([value[0] for value in scores], dtype=np.float64)
    frd = np.asarray([value[1] for value in scores], dtype=np.float64)
    return frc, frd


def pairwise_official_frdiv(
    candidates: Sequence[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    stacked = torch.stack([candidate.float() for candidate in candidates], dim=0)
    flattened = stacked.flatten(start_dim=1).to(device, non_blocking=True)
    with torch.no_grad():
        norms = flattened.square().sum(dim=1)
        matrix = (
            norms[:, None]
            + norms[None, :]
            - 2.0 * (flattened @ flattened.transpose(0, 1))
        ).clamp_min(0.0)
        matrix = matrix / float(flattened.shape[1])
    return matrix.cpu().numpy().astype(np.float64, copy=False)


def set_diversity(pairwise: np.ndarray, selected: Sequence[int]) -> float:
    indices = np.asarray(selected, dtype=np.int64)
    k = len(indices)
    if k <= 1:
        return 0.0
    block = pairwise[np.ix_(indices, indices)]
    return float(block.sum(dtype=np.float64) / (k * (k - 1)))


def feasible(
    selected: Sequence[int],
    frc: np.ndarray,
    frd: np.ndarray,
    frc_floor: float,
    frd_ceiling: float,
) -> bool:
    indices = np.asarray(selected, dtype=np.int64)
    return (
        float(frc[indices].sum(dtype=np.float64)) >= frc_floor - 1e-10
        and float(frd[indices].sum(dtype=np.float64)) <= frd_ceiling + 1e-10
    )


def quality_constrained_local_search(
    pairwise_diversity: np.ndarray,
    frc_scores: np.ndarray,
    frd_scores: np.ndarray,
    baseline_indices: Sequence[int],
    slots: Sequence[int],
    frc_slack: float = 0.0,
    frd_slack: float = 0.0,
    enable_two_swap: bool = True,
    min_gain: float = 1e-12,
) -> dict:
    """Slot-preserving search with q0 fixed.

    The selected set begins at the exact baseline. A one-swap pass is repeated
    until locally stable. If enabled, a two-swap pass then looks for quality
    compensation across two semantic slots while preserving the complete
    set-level FRC/FRD constraints.
    """
    selected = list(map(int, baseline_indices))
    if len(selected) != 10 or len(set(selected)) != 10:
        raise ValueError("baseline_indices must contain ten unique candidates")
    if pairwise_diversity.shape != (len(frc_scores), len(frc_scores)):
        raise ValueError("pairwise diversity shape mismatch")
    if len(frd_scores) != len(frc_scores) or len(slots) != len(frc_scores):
        raise ValueError("candidate metadata length mismatch")

    baseline_frc = float(frc_scores[selected].sum(dtype=np.float64))
    baseline_frd = float(frd_scores[selected].sum(dtype=np.float64))
    frc_floor = baseline_frc - float(frc_slack)
    frd_ceiling = baseline_frd + float(frd_slack)
    current_diversity = set_diversity(pairwise_diversity, selected)
    swaps: list[dict] = []

    def replacements(position: int, selected_set: set[int]) -> Iterable[int]:
        old = selected[position]
        slot = int(slots[old])
        for candidate_index in range(len(frc_scores)):
            if candidate_index in selected_set:
                continue
            if int(slots[candidate_index]) != slot:
                continue
            yield candidate_index

    while True:
        selected_set = set(selected)
        best = None

        # q0 is position 0 and remains frozen.
        for position in range(1, len(selected)):
            old = selected[position]
            for new in replacements(position, selected_set):
                proposal = selected.copy()
                proposal[position] = new
                if not feasible(
                    proposal, frc_scores, frd_scores, frc_floor, frd_ceiling
                ):
                    continue
                diversity = set_diversity(pairwise_diversity, proposal)
                gain = diversity - current_diversity
                key = (gain, -new, -position)
                if gain > min_gain and (best is None or key > best[0]):
                    best = (
                        key,
                        proposal,
                        {
                            "kind": "one_swap",
                            "position": position,
                            "old_index": int(old),
                            "new_index": int(new),
                            "diversity_gain": float(gain),
                        },
                    )

        if best is not None:
            _, selected, event = best
            current_diversity = set_diversity(pairwise_diversity, selected)
            swaps.append(event)
            continue

        if not enable_two_swap:
            break

        selected_set = set(selected)
        best_two = None
        positions = list(range(1, len(selected)))
        for offset, first_position in enumerate(positions):
            first_old = selected[first_position]
            first_candidates = list(replacements(first_position, selected_set))
            for second_position in positions[offset + 1 :]:
                second_old = selected[second_position]
                second_candidates = list(replacements(second_position, selected_set))
                for first_new in first_candidates:
                    for second_new in second_candidates:
                        if first_new == second_new:
                            continue
                        proposal = selected.copy()
                        proposal[first_position] = first_new
                        proposal[second_position] = second_new
                        if len(set(proposal)) != 10:
                            continue
                        if not feasible(
                            proposal,
                            frc_scores,
                            frd_scores,
                            frc_floor,
                            frd_ceiling,
                        ):
                            continue
                        diversity = set_diversity(pairwise_diversity, proposal)
                        gain = diversity - current_diversity
                        key = (
                            gain,
                            -first_new,
                            -second_new,
                            -first_position,
                            -second_position,
                        )
                        if gain > min_gain and (
                            best_two is None or key > best_two[0]
                        ):
                            best_two = (
                                key,
                                proposal,
                                {
                                    "kind": "two_swap",
                                    "positions": [
                                        int(first_position),
                                        int(second_position),
                                    ],
                                    "old_indices": [
                                        int(first_old),
                                        int(second_old),
                                    ],
                                    "new_indices": [
                                        int(first_new),
                                        int(second_new),
                                    ],
                                    "diversity_gain": float(gain),
                                },
                            )

        if best_two is None:
            break
        _, selected, event = best_two
        current_diversity = set_diversity(pairwise_diversity, selected)
        swaps.append(event)

    indices = np.asarray(selected, dtype=np.int64)
    return {
        "selected_indices": selected,
        "baseline_frc": baseline_frc,
        "baseline_frd": baseline_frd,
        "baseline_frdiv": set_diversity(pairwise_diversity, baseline_indices),
        "selected_frc": float(frc_scores[indices].sum(dtype=np.float64)),
        "selected_frd": float(frd_scores[indices].sum(dtype=np.float64)),
        "selected_frdiv": current_diversity,
        "frc_floor": frc_floor,
        "frd_ceiling": frd_ceiling,
        "swaps": swaps,
    }


def mean_key(records: Sequence[dict], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records]))


def build_summary(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("cannot summarize empty record list")
    baseline_frc = mean_key(records, "baseline_frc")
    baseline_frd = mean_key(records, "baseline_frd")
    baseline_frdiv = mean_key(records, "baseline_frdiv")
    selected_frc = mean_key(records, "selected_frc")
    selected_frd = mean_key(records, "selected_frd")
    selected_frdiv = mean_key(records, "selected_frdiv")
    gains = np.asarray(
        [
            float(record["selected_frdiv"] - record["baseline_frdiv"])
            for record in records
        ],
        dtype=np.float64,
    )
    return {
        "num_examples": len(records),
        "baseline": {
            "FRC": baseline_frc,
            "exact_FRD": baseline_frd,
            "FRDiv": baseline_frdiv,
        },
        "oracle_selected": {
            "FRC": selected_frc,
            "exact_FRD": selected_frd,
            "FRDiv": selected_frdiv,
        },
        "delta": {
            "FRC": selected_frc - baseline_frc,
            "exact_FRD": selected_frd - baseline_frd,
            "FRDiv": selected_frdiv - baseline_frdiv,
        },
        "examples_with_positive_frdiv_gain": int((gains > 1e-12).sum()),
        "positive_frdiv_gain_fraction": float((gains > 1e-12).mean()),
        "median_frdiv_gain": float(np.median(gains)),
        "max_frdiv_gain": float(gains.max()),
        "mean_pool_size": float(
            np.mean([float(record["pool_size"]) for record in records])
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("candidate generation requires CUDA")

    project_dir = Path(__file__).resolve().parents[1]
    initialize_official_hydra_runtime(project_dir)

    data_dir = args.data_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.session_split_manifest
    if not manifest_path.is_absolute():
        manifest_path = project_dir / manifest_path
    manifest_path = manifest_path.resolve()
    allowed_sessions = load_session_allowlist(manifest_path, args.split_name)

    random.seed(args.selection_seed)
    np.random.seed(args.selection_seed)
    torch.manual_seed(args.selection_seed)
    torch.cuda.manual_seed_all(args.selection_seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("method") not in {
        "Emotion-Gated-Query-Mamba",
        "EQR-Mamba",
    }:
        raise ValueError("checkpoint is not an Emotion-Query Mamba checkpoint")

    anchor_config = ConditionalREGNNConfig(**checkpoint["anchor_model_config"])
    query_config = EmotionQueryMambaConfig.from_checkpoint_payload(
        dict(checkpoint["emotion_query_model_config"])
    )
    if anchor_config.max_seq_len != args.clip_length:
        raise ValueError(
            f"anchor max_seq_len={anchor_config.max_seq_len} "
            f"but --clip-length={args.clip_length}"
        )
    if query_config.num_queries != 10 or query_config.num_emotions != 8:
        raise ValueError("probe requires 10 queries and 8 emotion queries")

    anchor = ConditionalREGNN(anchor_config)
    model = EmotionQueryResidualMamba(anchor, query_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Keep train-side file paths, but use the official test-style __getitem__
    # protocol for full-length speaker streams and ten raw listener GTs.
    # This local ReactionDataset ignores a full_sequence_eval keyword.
    dataset = ReactionDataset(
        root_dir=str(data_dir),
        split="train",
        clip_length=args.clip_length,
        target_size=224,
        crop_size=224,
        fps=30,
        audio_feature_type="wav2vec",
        load_video_s=False,
        load_video_l=False,
        load_audio=True,
        load_emotion_s=True,
        load_emotion_l=True,
        load_3dmm_s=True,
        load_3dmm_l=False,
        load_video_fea_s=False,
        load_video_fea_l=False,
        normalize_3dmm="standard",
    )
    dataset._split = "test"
    probe_indices = select_probe_indices(
        dataset,
        allowed_sessions,
        args.max_examples,
        args.selection_seed,
    )

    variants = build_candidate_variants(
        args.baseline_residual_scale,
        args.baseline_style_scale,
        args.include_mixture_sweep,
    )
    channel_multipliers = (
        [args.au_residual_multiplier] * 15
        + [args.va_residual_multiplier] * 2
        + [args.expression_residual_multiplier] * 8
    )
    processor = Processor(
        config_name="configs/model/emotion_autoencoder.yaml",
        ckpt_dir=str(project_dir / "pretrained_models" / "post_processor"),
        cfg_dir=str(project_dir),
        clip_len_test=1000,
        device=device,
        num_preds=query_config.num_queries,
    )

    records: list[dict] = []
    for offset, dataset_index in enumerate(probe_indices):
        clip_id = dataset.speaker_path_list[dataset_index].as_posix()
        sample_seed = int.from_bytes(
            stable_rank(args.selection_seed, clip_id)[:8], "big"
        )
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))
        torch.manual_seed(sample_seed % (2**63 - 1))

        batch = custom_collate([dataset[dataset_index]])
        speaker_audio = batch[0][0]
        speaker_emotion = batch[2][0]
        speaker_3dmm = batch[3][0]
        raw_targets = batch[5][0]
        raw_length = int(batch[7][0])
        if not isinstance(raw_targets, list) or len(raw_targets) != 10:
            raise RuntimeError(f"{clip_id}: expected ten raw listener GTs")

        # The official processor uses only prediction length to align the
        # variable-length GTs. A zero placeholder gives the same target path.
        with torch.inference_mode():
            targets = processor.forward(
                [torch.zeros(raw_length, 25)], [raw_targets]
            )[0].float().cpu().contiguous()

        expected_shape = (10, raw_length, 25)
        if tuple(targets.shape) != expected_shape:
            raise RuntimeError(
                f"{clip_id}: expected GT shape {expected_shape}, "
                f"got {tuple(targets.shape)}"
            )

        candidates, candidate_metadata = generate_candidate_pool(
            model,
            speaker_audio,
            speaker_emotion,
            speaker_3dmm,
            raw_length,
            variants,
            device,
            args.clip_length,
            args.clip_batch_size,
            channel_multipliers,
        )
        frc_scores, frd_scores = score_candidate_pool(
            candidates, targets, args.score_workers
        )
        pairwise = pairwise_official_frdiv(candidates, device)

        id_to_index = {
            metadata["id"]: index
            for index, metadata in enumerate(candidate_metadata)
        }
        baseline_indices = [
            id_to_index[f"baseline/q{slot}"] for slot in range(10)
        ]
        slots = [int(metadata["slot"]) for metadata in candidate_metadata]

        result = quality_constrained_local_search(
            pairwise,
            frc_scores,
            frd_scores,
            baseline_indices,
            slots,
            frc_slack=args.frc_slack,
            frd_slack=args.frd_slack,
            enable_two_swap=not args.disable_two_swap,
        )
        selected_indices = result["selected_indices"]
        selected_ids = [
            candidate_metadata[index]["id"] for index in selected_indices
        ]
        record = {
            "dataset_index": int(dataset_index),
            "clip_id": clip_id,
            "sequence_length": raw_length,
            "pool_size": len(candidates),
            **{
                key: value
                for key, value in result.items()
                if key != "selected_indices"
            },
            "selected_indices": selected_indices,
            "selected_ids": selected_ids,
            "candidate_labels": [
                {
                    **metadata,
                    "FRC_candidate": float(frc_scores[index]),
                    "exact_FRD_candidate": float(frd_scores[index]),
                }
                for index, metadata in enumerate(candidate_metadata)
            ],
        }
        records.append(record)
        print(
            f"[{offset + 1}/{len(probe_indices)}] {clip_id} "
            f"pool={len(candidates)} | "
            f"FRDiv {record['baseline_frdiv']:.6f} -> "
            f"{record['selected_frdiv']:.6f} | "
            f"FRC {record['baseline_frc']:.6f} -> "
            f"{record['selected_frc']:.6f} | "
            f"FRD {record['baseline_frd']:.3f} -> "
            f"{record['selected_frd']:.3f}",
            flush=True,
        )

    summary = build_summary(records)
    payload = {
        "experiment": "mam_reactor_gt_oracle_candidate_pool_offline_v1",
        "scope": (
            "train/B_fit GT-oracle feasibility probe; "
            "not deployable and not a held-out test score"
        ),
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "split_manifest": str(manifest_path),
        "split_name": args.split_name,
        "selection_seed": args.selection_seed,
        "probe_indices": probe_indices,
        "quality_contract": {
            "per_example_frc_slack": args.frc_slack,
            "per_example_exact_frd_slack": args.frd_slack,
            "slot_preserving": True,
            "keep_anchor_q0": True,
            "two_swap_search": not args.disable_two_swap,
        },
        "candidate_policy": {
            "baseline_residual_scale": args.baseline_residual_scale,
            "baseline_style_scale": args.baseline_style_scale,
            "channel_multipliers": {
                "AU": args.au_residual_multiplier,
                "VA": args.va_residual_multiplier,
                "expression": args.expression_residual_multiplier,
            },
            "include_mixture_sweep": args.include_mixture_sweep,
            "variants": [
                {
                    "name": variant.name,
                    "residual_scale": variant.residual_scale,
                    "style_residual_scale": variant.style_residual_scale,
                    "take_queries": list(variant.take_queries),
                    "gate_value": variant.gate_value,
                    "mixture_weights": (
                        list(variant.mixture_weights)
                        if variant.mixture_weights is not None
                        else None
                    ),
                }
                for variant in variants
            ],
        },
        "summary": summary,
        "examples": records,
    }

    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
