#!/usr/bin/env python3
"""MP3 audio to Open Note Block Studio (.nbs) converter.

The high-quality path first separates the mix with Demucs, transcribes the
melodic stems with Basic Pitch, quantizes the resulting note events to the NBS
grid, and classifies drum onsets into Minecraft note-block percussion.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence


NBS_VERSION = 5
VANILLA_INSTRUMENT_COUNT = 16
NBS_LOWEST_MIDI = 21  # NBS key 0 is A0.
NBS_KEY_MIN = 0
NBS_KEY_MAX = 87
MINECRAFT_KEY_MIN = 33
MINECRAFT_KEY_MAX = 57
MAX_NBS_TICK = 65_535
DEMUCS_MODEL = "htdemucs_ft"
DEMUCS_MODEL_COUNT = 4
DEMUCS_SHIFTS = 10
DEMUCS_OVERLAP = 0.50
SEPARATION_CACHE_VERSION = 2
DEMUCS_STEMS = ("vocals", "bass", "drums", "other")
DEMUCS_INSTRUMENT_MODEL = "htdemucs_6s"
DEMUCS_INSTRUMENT_MODEL_COUNT = 1
DEMUCS_INSTRUMENT_SHIFTS = 2
DEMUCS_INSTRUMENT_STEMS = (
    "vocals",
    "bass",
    "drums",
    "guitar",
    "piano",
    "other",
)
INSTRUMENT_SEPARATION_CACHE_VERSION = 1
MINECRAFT_TIMELINE_TPS = 10.0
# Open Note Block Studio stores tempo directly as ticks per second.  A 40 TPS
# native timeline keeps source attacks within 12.5 ms while leaving ample room
# in the unsigned 16-bit song-length field for ordinary songs.  The explicitly
# selected ``minecraft`` mode remains fixed at the redstone-compatible 10 TPS.
PRECISE_TIMELINE_TPS = 40.0
MINIMUM_NBS_TIMELINE_TPS = 0.25
ADAPTIVE_TRANSCRIPTION_CACHE_VERSION = 1
YAMNET_SAMPLE_RATE = 16_000
YAMNET_MODEL_URL = (
    "https://huggingface.co/audiomagic/yamnet-onnx/resolve/"
    "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd/yamnet.onnx?download=true"
)
YAMNET_MODEL_SHA256 = (
    "d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e"
)
# AudioSet classes: Speech; Singing through Humming; Opera; Vocal music;
# A capella.  Gender-specific voice labels were removed from this YAMNet build.
YAMNET_VOCAL_CLASS_INDICES = (0, *range(24, 33), 233, 249, 250)
YAMNET_INSTRUMENT_CLASS_SLICE = slice(133, 211)
INSTRUMENTAL_STEM_CACHE_VERSION = 1

INSTRUMENTS = {
    "piano": 0,
    "bass": 1,
    "bass_drum": 2,
    "snare": 3,
    "hat": 4,
    "guitar": 5,
    "flute": 6,
    "bell": 7,
    "chime": 8,
    "xylophone": 9,
    "iron_xylophone": 10,
    "cow_bell": 11,
    "didgeridoo": 12,
    "bit": 13,
    "banjo": 14,
    "pling": 15,
}
BACKGROUND_ROLE_INSTRUMENTS = {
    "guitar": INSTRUMENTS["guitar"],
    "piano": INSTRUMENTS["piano"],
}


class ConversionError(RuntimeError):
    """A user-facing conversion error."""


@dataclass(frozen=True)
class NbsNote:
    tick: int
    layer: int
    instrument: int
    key: int
    velocity: int = 100
    panning: int = 0
    pitch: int = 0
    continuation: bool = False
    audio_dynamic: bool = False
    source_role: str | None = None
    # Analysis-only metadata.  These values are deliberately not serialized to
    # NBS, but retaining them until the final dynamics pass lets that pass
    # measure the actual source pitch at the actual source time.  Reconstructing
    # either value from a folded NBS key or a quantized tick loses information.
    source_midi: int | None = None
    source_time_seconds: float | None = None
    source_loudness: float | None = None
    source_snr_db: float | None = None
    source_onset_strength: float | None = None


@dataclass(frozen=True)
class ConversionConfig:
    bpm: float | None = None
    ticks_per_beat: int = 4
    # Twelve lanes cover full two-handed piano voicings and dense ensemble
    # attacks without imposing the old four-note loss before source validation.
    # Empty lanes are cheap in NBS, while a discarded chord tone is
    # unrecoverable after arrangement.
    max_chord_notes: int = 12
    sensitivity: float = 0.5
    retrigger_beats: float = 2.0
    instrument: int | None = None
    include_drums: bool = True
    # Minecraft accepts only NBS keys 33..57.  Keep every detected pitch by
    # octave-folding it into that range by default; otherwise downstream
    # Minecraft importers silently discard valid transcription events.
    minecraft_range: bool = True
    sample_rate: int = 22_050
    hop_length: int = 256
    time_signature: int = 4
    separation: str = "demucs"
    transcription: str = "ai"
    timing: str = "precise"
    vocals: str = "auto"


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    detected_bpm: float
    effective_bpm: float
    ticks_per_second: float
    duration_seconds: float
    vocal_notes: int
    bass_notes: int
    accompaniment_notes: int
    drum_notes: int
    layer_count: int
    timing: str = "precise"
    vocal_handling: str = "dedicated"
    maximum_timing_error_seconds: float = 0.0
    minecraft_range: bool = True
    octave_folded_notes: int = 0

    @property
    def melodic_notes(self) -> int:
        return self.vocal_notes + self.bass_notes + self.accompaniment_notes

    @property
    def total_notes(self) -> int:
        return self.melodic_notes + self.drum_notes


@dataclass(frozen=True)
class _VocalDetection:
    present: bool
    used_model: bool
    relative_loudness_db: float
    energy_active_ratio: float
    model_active_ratio: float = 0.0
    longest_vocal_seconds: float = 0.0
    score_p90: float = 0.0


@dataclass(frozen=True)
class _PitchEvent:
    tick: int
    midi: int
    velocity: int
    panning: int
    strength_db: float
    continuation: bool = False
    source_time_seconds: float | None = None


@dataclass(frozen=True)
class _TimedPitchEvent:
    start_seconds: float
    end_seconds: float
    midi: int
    amplitude: float
    source_role: str | None = None
    # `amplitude` is the arranged-candidate score used for backwards
    # compatibility by the planners below.  Keep the independent evidence
    # dimensions alongside it: Basic Pitch confidence is not source loudness.
    model_confidence: float | None = None
    pitch_loudness: float | None = None
    onset_strength: float | None = None
    pitch_snr_db: float | None = None


@dataclass(frozen=True)
class _QuantizedPitchEvent:
    start_tick: int
    end_tick: int
    midi: int
    amplitude: float
    phase_tick: float | None = None
    end_phase_tick: float | None = None
    source_role: str | None = None
    model_confidence: float | None = None
    pitch_loudness: float | None = None
    onset_strength: float | None = None
    pitch_snr_db: float | None = None
    source_time_seconds: float | None = None


@dataclass(frozen=True)
class _VoicedPitchEvent:
    """A quantized event assigned to a stable NBS accompaniment voice."""

    start_tick: int
    end_tick: int
    midi: int
    amplitude: float
    voice: int
    phase_tick: float | None = None
    end_phase_tick: float | None = None
    source_role: str | None = None
    model_confidence: float | None = None
    pitch_loudness: float | None = None
    onset_strength: float | None = None
    pitch_snr_db: float | None = None
    source_time_seconds: float | None = None


@dataclass(frozen=True)
class _PitchAnalysis:
    """Pitch-resolved source evidence on one canonical audio timeline."""

    magnitude: object
    flux: object
    hop_length: int
    onset_magnitude: object
    onset_hop_length: int
    onset_frequencies: object
    sample_rate: int
    global_reference: float


class _DemucsProgressParser:
    """Combine Demucs' repeated tqdm bars into one 0..1 progress value."""

    _PERCENT_PATTERN = re.compile(r"(?<!\d)(\d{1,3})%\|")

    def __init__(self, expected_passes: int):
        self.expected_passes = max(1, expected_passes)
        self.completed_passes = 0
        self.current_percent = -1

    def feed(self, text: str) -> float | None:
        if "seconds/s" not in text:
            return None
        result: float | None = None
        for match in self._PERCENT_PATTERN.finditer(text):
            percent = int(match.group(1))
            if not 0 <= percent <= 100:
                continue
            if self.current_percent >= 0 and percent < self.current_percent:
                # tqdm has moved from the completed bar to the next model/shift.
                self.current_percent = -1
            if percent == 100:
                # tqdm commonly writes the final 100% line twice. Count it once.
                if self.current_percent < 100:
                    self.completed_passes = min(
                        self.expected_passes, self.completed_passes + 1
                    )
                self.current_percent = 100
                result = self.completed_passes / self.expected_passes
            else:
                self.current_percent = percent
                result = (
                    self.completed_passes + percent / 100.0
                ) / self.expected_passes
        return min(1.0, result) if result is not None else None

    @property
    def active_pass(self) -> int:
        return min(self.expected_passes, self.completed_passes + 1)

    @property
    def displayed_pass(self) -> int:
        if self.current_percent >= 100:
            return self.completed_passes
        return self.active_pass


class _ConsoleProgress:
    """Render a single overall percentage bar with a continuously updated ETA."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.started_at = time.monotonic()
        self.fraction = 0.0
        self.status = "Preparing..."
        self.last_render_at = 0.0
        self.last_render_fraction = -1.0
        self.last_line_length = 0
        self.progress_samples: deque[tuple[float, float]] = deque(
            [(self.started_at, 0.0)], maxlen=200
        )
        self.dynamic = bool(
            getattr(self.stream, "isatty", lambda: False)()
        )
        self.line_open = False

    @staticmethod
    def _display_width(text: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            for character in text
        )

    @classmethod
    def _truncate_status(cls, text: str, maximum_width: int = 26) -> str:
        if cls._display_width(text) <= maximum_width:
            return text
        result: list[str] = []
        width = 0
        for character in text:
            character_width = (
                2
                if unicodedata.east_asian_width(character) in {"W", "F"}
                else 1
            )
            if width + character_width > maximum_width - 3:
                break
            result.append(character)
            width += character_width
        return "".join(result) + "..."

    def update(
        self,
        fraction: float,
        status: str | None = None,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        previous_fraction = self.fraction
        self.fraction = max(self.fraction, _clamp(fraction, 0.0, 1.0))
        if status is not None:
            self.status = status
        if self.fraction > previous_fraction:
            self.progress_samples.append((now, self.fraction))
        while (
            len(self.progress_samples) > 2
            and now - self.progress_samples[0][0] > 8.0
        ):
            self.progress_samples.popleft()

        if not force:
            if self.dynamic:
                if (
                    now - self.last_render_at < 0.08
                    and self.fraction - self.last_render_fraction < 0.002
                ):
                    return
            elif self.fraction - self.last_render_fraction < 0.05:
                return

        elapsed = max(0.0, now - self.started_at)
        if self.fraction >= 1.0:
            eta_text = "0s"
        elif elapsed >= 1.0 and self.fraction >= 0.01:
            sample_time, sample_fraction = self.progress_samples[0]
            sample_elapsed = now - sample_time
            sample_progress = self.fraction - sample_fraction
            if sample_elapsed >= 0.5 and sample_progress >= 0.005:
                rate = sample_progress / sample_elapsed
            else:
                rate = self.fraction / elapsed
            remaining = math.ceil((1.0 - self.fraction) / max(rate, 1e-9))
            eta_text = f"about {max(1, remaining)}s"
        else:
            eta_text = "calculating"

        width = 18
        filled = (
            width
            if self.fraction >= 1.0
            else min(width - 1, math.floor(width * self.fraction))
        )
        bar = "#" * filled + "-" * (width - filled)
        visible_status = self._truncate_status(self.status)
        line = (
            f"[{bar}] {self.fraction * 100:6.2f}% | "
            f"ETA {eta_text} | {visible_status}"
        )
        if self.dynamic:
            line_width = self._display_width(line)
            padding = " " * max(0, self.last_line_length - line_width)
            print(f"\r{line}{padding}", end="", file=self.stream, flush=True)
        else:
            print(line, file=self.stream, flush=True)
            line_width = self._display_width(line)
        self.last_line_length = line_width
        self.last_render_at = now
        self.last_render_fraction = self.fraction
        self.line_open = True

    def message(self, status: str) -> None:
        self.update(self.fraction, status, force=True)

    def complete(self) -> None:
        if self.fraction < 1.0 or self.status != "Conversion complete":
            self.update(1.0, "Conversion complete", force=True)
        self.close_line()

    def close_line(self) -> None:
        if self.dynamic and self.line_open:
            print(file=self.stream, flush=True)
        self.line_open = False


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _round_tick(value: float) -> int:
    """Round a timeline position deterministically, with half ticks away from zero."""

    if not math.isfinite(value):
        raise ValueError("A timeline position must be finite.")
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _round_tick_array(values, np):
    """Vectorized counterpart of :func:`_round_tick` for analysis frames."""

    values = np.asarray(values)
    rounded = np.where(
        values >= 0.0,
        np.floor(values + 0.5),
        np.ceil(values - 0.5),
    )
    return rounded.astype(np.int64)


def _seconds_to_tick(
    seconds: float,
    timeline_origin_seconds: float,
    tick_seconds: float,
) -> int:
    """Map an absolute source time to the single shared NBS clock."""

    return _round_tick((seconds - timeline_origin_seconds) / tick_seconds)


def _validate_source_timing(
    notes: Iterable[NbsNote],
    timeline_origin_seconds: float,
    ticks_per_second: float,
) -> float:
    """Assert that serialized AI notes remain within half a tick of the source."""

    if not math.isfinite(ticks_per_second) or ticks_per_second <= 0.0:
        raise ConversionError("The NBS timeline rate must be positive and finite.")
    maximum_error = 0.0
    tolerance = 0.5 / ticks_per_second + 1e-9
    for note in notes:
        if note.source_time_seconds is None:
            continue
        playback_seconds = timeline_origin_seconds + note.tick / ticks_per_second
        error = abs(playback_seconds - note.source_time_seconds)
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            raise ConversionError(
                "An internal timing check failed before writing the NBS file."
            )
    return maximum_error


def _validate_minecraft_key_range(notes: Iterable[NbsNote]) -> int:
    """Reject any note Minecraft would ignore and count octave-folded notes."""

    note_list = list(notes)
    out_of_range = [
        note.key
        for note in note_list
        if not MINECRAFT_KEY_MIN <= note.key <= MINECRAFT_KEY_MAX
    ]
    if out_of_range:
        raise ConversionError(
            "An internal Minecraft-range check failed before writing the NBS "
            f"file: {len(out_of_range)} note(s) use keys outside "
            f"{MINECRAFT_KEY_MIN}..{MINECRAFT_KEY_MAX} "
            f"(observed {min(out_of_range)}..{max(out_of_range)})."
        )

    return sum(
        1
        for note in note_list
        if note.source_midi is not None
        and note.key != note.source_midi - NBS_LOWEST_MIDI
    )


def _resolve_timing_grid(
    bpm: float,
    config: ConversionConfig,
    duration_seconds: float | None = None,
) -> tuple[float, float]:
    """Return the serialized NBS tick rate and the displayed song BPM.

    Precise mode is intended for native NBS playback and retains short attacks.
    Minecraft schematic/structure playback can represent only 10, 5, or 2.5
    NBS ticks per second.  Ten ticks per second gives the finest compatible
    grid and, unlike a beat-derived rate, does not accumulate timing drift when
    Open Note Block Studio exports the song to NBT.
    """

    if config.timing == "precise":
        ticks_per_second = PRECISE_TIMELINE_TPS
        if duration_seconds is not None and duration_seconds > 0.0:
            # NBS v5 stores tick positions in an unsigned 16-bit field.  Use
            # the finest exactly serializable clock instead of rejecting long
            # recordings outright.  Quantization remains absolute, so the
            # lower resolution increases bounded local error but never drift.
            length_limited_tps = (
                math.floor(MAX_NBS_TICK / duration_seconds * 100.0) / 100.0
            )
            ticks_per_second = min(ticks_per_second, length_limited_tps)
            if ticks_per_second < MINIMUM_NBS_TIMELINE_TPS:
                maximum_hours = (
                    MAX_NBS_TICK / MINIMUM_NBS_TIMELINE_TPS / 3600.0
                )
                raise ConversionError(
                    "The recording is longer than the NBS v5 timeline can "
                    f"represent (about {maximum_hours:.1f} hours)."
                )
        return ticks_per_second, bpm
    if config.timing == "minecraft":
        return MINECRAFT_TIMELINE_TPS, bpm
    if config.timing == "beat":
        requested_tps = bpm * config.ticks_per_beat / 60.0
        ticks_per_second = _round_tick(requested_tps * 100.0) / 100.0
        return (
            ticks_per_second,
            ticks_per_second * 60.0 / config.ticks_per_beat,
        )
    raise ConversionError(f"Unsupported timing mode: {config.timing}")


def _retrigger_interval_ticks_exact(config: ConversionConfig) -> float:
    """Return the unrounded repeat interval on the resolved NBS timeline."""

    if config.retrigger_beats <= 0.0:
        return 0.0
    ticks_per_beat = config.ticks_per_beat
    if config.timing in {"precise", "minecraft"} and config.bpm is not None:
        timeline_tps = (
            PRECISE_TIMELINE_TPS
            if config.timing == "precise"
            else MINECRAFT_TIMELINE_TPS
        )
        ticks_per_beat = timeline_tps * 60.0 / config.bpm
    interval = config.retrigger_beats * ticks_per_beat
    if not math.isfinite(interval) or interval <= 0.0:
        return 0.0
    # Multiple attacks cannot occupy fractions of one NBS tick.  Clamping here
    # also prevents extremely small user intervals from producing huge loops.
    return max(1.0, interval)


def _retrigger_interval_ticks(config: ConversionConfig) -> int:
    """Return a whole-tick approximation for window sizes, never scheduling."""

    interval = _retrigger_interval_ticks_exact(config)
    return _round_tick(interval) if interval > 0.0 else 0


def _iter_phase_locked_retrigger_ticks(
    start_tick: int,
    end_tick: int,
    config: ConversionConfig,
    *,
    phase_tick: float | None = None,
    end_phase_tick: float | None = None,
) -> Iterable[int]:
    """Yield repeats anchored to the exact onset without cumulative drift."""

    for tick, _position in _iter_phase_locked_retrigger_positions(
        start_tick,
        end_tick,
        config,
        phase_tick=phase_tick,
        end_phase_tick=end_phase_tick,
    ):
        yield tick


def _iter_phase_locked_retrigger_positions(
    start_tick: int,
    end_tick: int,
    config: ConversionConfig,
    *,
    phase_tick: float | None = None,
    end_phase_tick: float | None = None,
) -> Iterable[tuple[int, float]]:
    """Yield each serialized repeat and its unrounded canonical position."""

    interval = _retrigger_interval_ticks_exact(config)
    if interval <= 0.0 or end_tick <= start_tick + 1:
        return

    phase = float(start_tick) if phase_tick is None else phase_tick
    phase_end = float(end_tick) if end_phase_tick is None else end_phase_tick
    repeat_index = 1
    previous_tick = start_tick
    while True:
        position = phase + repeat_index * interval
        if position >= phase_end:
            return
        tick = _round_tick(position)
        if tick >= end_tick:
            return
        if tick > previous_tick:
            yield tick, position
            previous_tick = tick
        repeat_index += 1


def _advance_retrigger_deadline(
    deadline: float,
    emitted_tick: int,
    interval: float,
) -> float:
    """Advance a fractional repeat phase beyond an emitted integer tick."""

    while _round_tick(deadline) <= emitted_tick:
        deadline += interval
    return deadline


def _estimate_stem_delay_seconds(
    source_audio,
    stems: Iterable,
    sample_rate: int,
    np,
) -> float:
    """Estimate a common decoder/separator delay; positive means stems are late."""

    stem_list = list(stems)
    if not stem_list or source_audio.size < sample_rate // 2:
        return 0.0
    try:
        from scipy import signal
    except ImportError:
        return 0.0

    reconstructed = np.sum(np.stack(stem_list, axis=0), axis=0)
    sample_count = min(len(source_audio), len(reconstructed))
    if sample_count < sample_rate // 2:
        return 0.0

    # Use the loudest of several positions so a quiet intro or breakdown cannot
    # make the correlation select an arbitrary lag. Twenty seconds is long
    # enough to distinguish the waveform while keeping the FFT inexpensive.
    window_size = min(
        sample_count, max(sample_rate, _round_tick(sample_rate * 20.0))
    )
    available = sample_count - window_size
    starts = sorted(
        {
            _round_tick(available * fraction)
            for fraction in ((0.0,) if available <= 0 else (0.15, 0.40, 0.65, 0.85))
        }
    )
    start = max(
        starts,
        key=lambda index: float(
            np.mean(np.square(source_audio[index : index + window_size]))
        ),
    )
    source_window = np.asarray(
        source_audio[start : start + window_size], dtype=np.float32
    ).copy()
    stem_window = np.asarray(
        reconstructed[start : start + window_size], dtype=np.float32
    ).copy()
    source_window = np.nan_to_num(source_window, copy=False)
    stem_window = np.nan_to_num(stem_window, copy=False)
    source_window -= float(np.mean(source_window))
    stem_window -= float(np.mean(stem_window))
    if (
        float(np.max(np.abs(source_window))) < 1e-7
        or float(np.max(np.abs(stem_window))) < 1e-7
    ):
        return 0.0

    maximum_lag = min(_round_tick(sample_rate * 0.100), window_size // 8)
    correlation = signal.correlate(
        stem_window, source_window, mode="full", method="fft"
    )
    center = len(source_window) - 1
    relevant = correlation[center - maximum_lag : center + maximum_lag + 1]
    lag_samples = int(np.argmax(relevant)) - maximum_lag

    if lag_samples >= 0:
        aligned_stem = stem_window[lag_samples:]
        aligned_source = source_window[: len(aligned_stem)]
    else:
        aligned_source = source_window[-lag_samples:]
        aligned_stem = stem_window[: len(aligned_source)]
    denominator = math.sqrt(
        float(np.dot(aligned_source, aligned_source))
        * float(np.dot(aligned_stem, aligned_stem))
    )
    confidence = (
        float(np.dot(aligned_source, aligned_stem)) / denominator
        if denominator > 1e-12
        else 0.0
    )
    if not math.isfinite(confidence) or confidence < 0.25:
        return 0.0
    return lag_samples / sample_rate


def _shift_audio_to_timeline(audio, delay_seconds: float, sample_rate: int, np):
    """Remove a measured stem delay without changing the array duration."""

    delay_samples = _round_tick(delay_seconds * sample_rate)
    if delay_samples == 0:
        return audio
    aligned = np.zeros_like(audio)
    if delay_samples > 0 and delay_samples < len(audio):
        aligned[:-delay_samples] = audio[delay_samples:]
    elif delay_samples < 0 and -delay_samples < len(audio):
        aligned[-delay_samples:] = audio[:delay_samples]
    return aligned


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_yamnet_model(
    report: Callable[[str], None] | None = None,
    progress_update: Callable[[float, str], None] | None = None,
) -> Path:
    """Download the pinned, checksum-verified ONNX vocal detector once."""

    if importlib.util.find_spec("onnxruntime") is None:
        raise ConversionError(
            "ONNX Runtime is required for automatic vocal detection. "
            "Run setup.bat again."
        )
    notify = report or (lambda _message: None)
    cache_directory = Path(__file__).resolve().parent / ".model_cache"
    model_path = cache_directory / f"yamnet-{YAMNET_MODEL_SHA256[:12]}.onnx"
    if model_path.is_file() and _sha256_file(model_path) == YAMNET_MODEL_SHA256:
        if progress_update is not None:
            progress_update(1.0, "Vocal detection model ready")
        return model_path

    cache_directory.mkdir(parents=True, exist_ok=True)
    notify("Downloading the vocal detection model (about 16 MB, first run only)...")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".yamnet-",
            suffix=".tmp",
            dir=cache_directory,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            request = urllib.request.Request(
                YAMNET_MODEL_URL,
                headers={"User-Agent": "mp3-to-nbs/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                total_size = int(response.headers.get("Content-Length") or 0)
                received = 0
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
                    received += len(chunk)
                    if progress_update is not None and total_size > 0:
                        progress_update(
                            min(0.99, received / total_size),
                            "Downloading the vocal detection model...",
                        )
        if _sha256_file(temporary_path) != YAMNET_MODEL_SHA256:
            raise ConversionError(
                "The downloaded vocal detection model failed its integrity check."
            )
        os.replace(temporary_path, model_path)
        temporary_path = None
    except ConversionError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise ConversionError("Could not download the vocal detection model.") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if progress_update is not None:
        progress_update(1.0, "Vocal detection model ready")
    return model_path


def _classify_vocal_scores(
    scores,
    *,
    relative_loudness_db: float,
    energy_active_ratio: float,
    np,
) -> _VocalDetection:
    """Classify YAMNet frames while rejecting isolated instrumental leakage."""

    if scores.ndim != 2 or scores.shape[1] != 521 or scores.shape[0] == 0:
        raise ValueError("Unexpected YAMNet score shape")
    vocal_scores = np.max(scores[:, YAMNET_VOCAL_CLASS_INDICES], axis=1)
    instrument_scores = np.max(scores[:, YAMNET_INSTRUMENT_CLASS_SLICE], axis=1)
    active = (vocal_scores >= 0.15) & (
        vocal_scores >= instrument_scores * 1.05
    )
    active_ratio = float(np.mean(active))
    longest_run = 0
    current_run = 0
    for is_active in active:
        current_run = current_run + 1 if bool(is_active) else 0
        longest_run = max(longest_run, current_run)
    longest_seconds = (
        0.96 + max(0, longest_run - 1) * 0.48 if longest_run else 0.0
    )
    score_p90 = float(np.quantile(vocal_scores, 0.90))
    sustained_voice = (
        active_ratio >= 0.10
        and score_p90 >= 0.15
        and longest_seconds >= 3.5
    ) or longest_seconds >= 6.0
    present = (
        relative_loudness_db >= -30.0
        and energy_active_ratio >= 0.04
        and sustained_voice
    )
    return _VocalDetection(
        present=present,
        used_model=True,
        relative_loudness_db=relative_loudness_db,
        energy_active_ratio=energy_active_ratio,
        model_active_ratio=active_ratio,
        longest_vocal_seconds=longest_seconds,
        score_p90=score_p90,
    )


def _detect_vocal_presence(
    vocal_audio,
    mix_audio,
    sample_rate: int,
    librosa,
    np,
    *,
    report: Callable[[str], None] | None = None,
    progress_update: Callable[[float, str], None] | None = None,
    model_path: Path | None = None,
) -> _VocalDetection:
    """Detect sustained human voice in a Demucs vocal stem."""

    notify = report or (lambda _message: None)
    vocal_rms = math.sqrt(float(np.mean(np.square(vocal_audio))))
    mix_rms = math.sqrt(float(np.mean(np.square(mix_audio))))
    relative_loudness_db = 20.0 * math.log10(
        max(vocal_rms, 1e-12) / max(mix_rms, 1e-12)
    )
    frame_rms = librosa.feature.rms(
        y=vocal_audio, frame_length=2048, hop_length=512
    )[0]
    energy_threshold = max(mix_rms, 1e-12) * (10.0 ** (-30.0 / 20.0))
    energy_active_ratio = float(np.mean(frame_rms >= energy_threshold))

    try:
        if model_path is None:
            model_path = _ensure_yamnet_model(
                notify,
                (
                    (lambda fraction, status: progress_update(
                        fraction * 0.55, status
                    ))
                    if progress_update is not None
                    else None
                ),
            )
        import onnxruntime as ort

        if sample_rate != YAMNET_SAMPLE_RATE:
            detector_audio = librosa.resample(
                vocal_audio,
                orig_sr=sample_rate,
                target_sr=YAMNET_SAMPLE_RATE,
                res_type="kaiser_best",
            )
        else:
            detector_audio = vocal_audio
        detector_audio = np.clip(
            np.asarray(detector_audio, dtype=np.float32), -1.0, 1.0
        )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        score_chunks = []
        chunk_size = YAMNET_SAMPLE_RATE * 60
        total_chunks = max(1, math.ceil(len(detector_audio) / chunk_size))
        for chunk_number, start in enumerate(
            range(0, max(1, len(detector_audio)), chunk_size), start=1
        ):
            chunk = detector_audio[start : start + chunk_size]
            if len(chunk) == 0:
                chunk = np.zeros(YAMNET_SAMPLE_RATE, dtype=np.float32)
            outputs = session.run(None, {"waveform": chunk})
            scores = next(
                (
                    output
                    for output in outputs
                    if output.ndim == 2 and output.shape[1] == 521
                ),
                None,
            )
            if scores is None:
                raise ValueError("YAMNet scores were not returned")
            score_chunks.append(scores)
            if progress_update is not None:
                progress_update(
                    0.55 + 0.45 * chunk_number / total_chunks,
                    "Detecting vocals with AI...",
                )
        return _classify_vocal_scores(
            np.concatenate(score_chunks, axis=0),
            relative_loudness_db=relative_loudness_db,
            energy_active_ratio=energy_active_ratio,
            np=np,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        # Automatic mode remains usable offline.  The conservative fallback
        # avoids interpreting a weak Demucs residual as a dedicated singer.
        notify(
            "Warning: AI vocal detection is unavailable; using separated-stem "
            f"loudness ({type(exc).__name__})."
        )
        present = relative_loudness_db >= -16.0 and energy_active_ratio >= 0.20
        if progress_update is not None:
            progress_update(1.0, "Vocal loudness check complete")
        return _VocalDetection(
            present=present,
            used_model=False,
            relative_loudness_db=relative_loudness_db,
            energy_active_ratio=energy_active_ratio,
        )


def _write_instrumental_accompaniment_stem(
    cache_directory: Path,
    audio,
    sample_rate: int,
    amplitude_scale: float,
    np,
) -> Path:
    """Cache an aligned other+vocal stem for instrumental transcription."""

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ConversionError(
            "SoundFile is required to save the accompaniment stem."
        ) from exc
    target = cache_directory / (
        f"instrumental_other_aligned_v{INSTRUMENTAL_STEM_CACHE_VERSION}.wav"
    )
    if target.is_file():
        try:
            info = sf.info(str(target))
            if info.samplerate == sample_rate and info.frames == len(audio):
                return target
        except (OSError, RuntimeError):
            pass

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}-",
            suffix=".tmp.wav",
            dir=cache_directory,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        sf.write(
            str(temporary_path),
            np.asarray(audio * amplitude_scale, dtype=np.float32),
            sample_rate,
            subtype="FLOAT",
            format="WAV",
        )
        os.replace(temporary_path, target)
        temporary_path = None
    except (OSError, RuntimeError) as exc:
        raise ConversionError(
            "Could not save the merged instrumental accompaniment stem."
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def _locally_normalize_for_transcription(
    audio,
    sample_rate: int,
    librosa,
    np,
    *,
    maximum_gain: float = 12.0,
):
    """Compress only the model input so quiet phrases remain detectable.

    This signal is never used for velocity or source validation.  A smooth gain
    envelope exposes a locally quiet instrument to Basic Pitch without moving
    attacks; every returned candidate is subsequently measured against the
    untouched stem.
    """

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or sample_rate <= 0:
        return samples.copy()
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-7:
        return samples.copy()

    frame_length = 1 << max(
        7, int(round(math.log2(max(128.0, sample_rate * 0.050))))
    )
    hop_length = max(64, _round_tick(sample_rate * 0.012))
    frame_rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float64, copy=False)
    positive = frame_rms[frame_rms > 1e-9]
    if positive.size == 0:
        return samples.copy()
    target = max(float(np.quantile(positive, 0.82)), 1e-7)
    activity_floor = max(1e-7, target * 0.0025)
    gain = np.ones_like(frame_rms, dtype=np.float64)
    active = frame_rms >= activity_floor
    gain[active] = np.clip(
        target / np.maximum(frame_rms[active], 1e-9),
        1.0,
        max(1.0, maximum_gain),
    )

    # Smooth in log space so gain changes cannot create artificial onsets.
    smoothing_frames = max(3, _round_tick(0.40 * sample_rate / hop_length))
    if smoothing_frames % 2 == 0:
        smoothing_frames += 1
    padding = smoothing_frames // 2
    padded = np.pad(np.log(gain), (padding, padding), mode="edge")
    kernel = np.full(smoothing_frames, 1.0 / smoothing_frames, dtype=np.float64)
    gain = np.exp(np.convolve(padded, kernel, mode="valid"))
    frame_samples = np.minimum(
        np.arange(gain.size, dtype=np.float64) * hop_length,
        samples.size - 1,
    )
    sample_gain = np.interp(
        np.arange(samples.size, dtype=np.float64),
        frame_samples,
        gain,
        left=float(gain[0]),
        right=float(gain[-1]),
    )
    normalized = samples.astype(np.float64) * sample_gain
    normalized_peak = float(np.max(np.abs(normalized)))
    if normalized_peak > 0.98:
        normalized *= 0.98 / normalized_peak
    return normalized.astype(np.float32, copy=False)


def _write_adaptive_transcription_stem(
    cache_directory: Path,
    audio,
    sample_rate: int,
    role: str,
    librosa,
    np,
) -> Path:
    """Cache an aligned, locally normalized model-input stem atomically."""

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ConversionError(
            "SoundFile is required to save an adaptive transcription stem."
        ) from exc
    safe_role = re.sub(r"[^a-z0-9_-]+", "_", role.lower()).strip("_") or "stem"
    audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
    content_key = hashlib.sha256(audio_array.tobytes()).hexdigest()[:12]
    target = cache_directory / (
        f"adaptive_{safe_role}_{content_key}_v"
        f"{ADAPTIVE_TRANSCRIPTION_CACHE_VERSION}.wav"
    )
    if target.is_file():
        try:
            info = sf.info(str(target))
            if info.samplerate == sample_rate and info.frames == len(audio):
                return target
        except (OSError, RuntimeError):
            pass

    normalized = _locally_normalize_for_transcription(
        audio_array, sample_rate, librosa, np
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}-",
            suffix=".tmp.wav",
            dir=cache_directory,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        sf.write(
            str(temporary_path),
            normalized,
            sample_rate,
            subtype="FLOAT",
            format="WAV",
        )
        os.replace(temporary_path, target)
        temporary_path = None
    except (OSError, RuntimeError) as exc:
        raise ConversionError(
            f"Could not save the adaptive {role} transcription stem."
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def _select_background_stem_roles(
    stems: dict[str, object],
    reference_audio,
    np,
    *,
    maximum_roles: int,
) -> list[str]:
    """Choose audible instrument stems for dedicated background lanes.

    The six-source separator is used only as an additional instrument cue. A
    very weak residual is not promoted into a note layer, because independently
    normalizing separator bleed would make an inaudible artifact sound like a
    real instrument.
    """

    if maximum_roles <= 0 or reference_audio is None or reference_audio.size == 0:
        return []
    reference_rms = math.sqrt(
        max(float(np.mean(np.square(reference_audio))), 1e-12)
    )
    ranked: list[tuple[float, str]] = []
    for role in BACKGROUND_ROLE_INSTRUMENTS:
        audio = stems.get(role)
        if audio is None or audio.size == 0:
            continue
        role_rms = math.sqrt(max(float(np.mean(np.square(audio))), 1e-12))
        relative_db = 20.0 * math.log10(role_rms / reference_rms)
        if relative_db < -32.0:
            continue

        frame_size = 4096
        frame_count = len(audio) // frame_size
        if frame_count:
            frames = np.asarray(audio[: frame_count * frame_size]).reshape(
                frame_count, frame_size
            )
            frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
            active_ratio = float(
                np.mean(frame_rms >= max(reference_rms * 0.012, 1e-7))
            )
        else:
            active_ratio = 1.0
        if active_ratio < 0.025:
            continue
        score = relative_db + 5.0 * min(0.60, active_ratio)
        ranked.append((score, role))
    ranked.sort(reverse=True)
    return [role for _score, role in ranked[:maximum_roles]]


def _demucs_layer_layout(
    use_vocals: bool,
    max_chord_notes: int,
    background_roles: Sequence[str] = (),
) -> tuple[int, int, int, list[str]]:
    bass_layer = 1 if use_vocals else 0
    accompaniment_layer_offset = bass_layer + 1
    drum_layer_offset = accompaniment_layer_offset + max_chord_notes
    # Background instruments can receive more than one lane when their
    # measured chord polyphony requires it, so role-specific names assigned
    # before transcription would be misleading.  Keep functional lane names;
    # the serialized instrument field carries the eventual timbre.
    _ = background_roles
    accompaniment_names = ["Accompaniment lead"] if max_chord_notes else []
    if max_chord_notes >= 2:
        accompaniment_names.append("Accompaniment low anchor")
    if max_chord_notes >= 3:
        accompaniment_names.extend(
            f"Accompaniment voice {number}"
            for number in range(3, max_chord_notes + 1)
        )
    layer_names = (
        (["Vocal melody"] if use_vocals else [])
        + ["Bass"]
        + accompaniment_names
    )
    return (
        bass_layer,
        accompaniment_layer_offset,
        drum_layer_offset,
        layer_names,
    )


def fold_key_to_minecraft_range(key: int) -> int:
    """Fold an NBS key by octaves into Minecraft's playable 33..57 range."""

    if not NBS_KEY_MIN <= key <= NBS_KEY_MAX:
        raise ValueError(f"NBS key must be between 0 and 87: {key}")
    while key < MINECRAFT_KEY_MIN:
        key += 12
    while key > MINECRAFT_KEY_MAX:
        key -= 12
    return key


def choose_instrument(midi: int, requested: int | None) -> int:
    """Select a vanilla instrument, using source pitch when auto mode is active."""

    if requested is not None:
        return requested
    if midi < 48:
        return INSTRUMENTS["bass"]
    if midi < 60:
        return INSTRUMENTS["guitar"]
    if midi < 84:
        return INSTRUMENTS["piano"]
    return INSTRUMENTS["flute"]


def _write_u8(file_obj, value: int) -> None:
    file_obj.write(struct.pack("<B", value))


def _write_u16(file_obj, value: int) -> None:
    file_obj.write(struct.pack("<H", value))


def _write_i16(file_obj, value: int) -> None:
    file_obj.write(struct.pack("<h", value))


def _write_u32(file_obj, value: int) -> None:
    file_obj.write(struct.pack("<I", value))


def _write_string(file_obj, value: str) -> None:
    # NBS v5 retains the original one-byte string representation.  Open Note
    # Block Studio itself cannot save multi-byte metadata in this format.
    encoded = value.encode("cp1252", errors="replace")
    _write_u32(file_obj, len(encoded))
    file_obj.write(encoded)


def _validated_notes(notes: Iterable[NbsNote], layer_count: int) -> list[NbsNote]:
    result = sorted(notes, key=lambda note: (note.tick, note.layer))
    occupied: set[tuple[int, int]] = set()

    for note in result:
        if not 0 <= note.tick <= MAX_NBS_TICK:
            raise ConversionError(
                f"The song exceeds the maximum NBS length (tick={note.tick})."
            )
        if not 0 <= note.layer < layer_count:
            raise ConversionError(f"Invalid layer number: {note.layer}")
        if not 0 <= note.instrument < VANILLA_INSTRUMENT_COUNT:
            raise ConversionError(f"Invalid vanilla instrument number: {note.instrument}")
        if not NBS_KEY_MIN <= note.key <= NBS_KEY_MAX:
            raise ConversionError(f"Invalid NBS key: {note.key}")
        if not 0 <= note.velocity <= 100:
            raise ConversionError(f"Invalid velocity: {note.velocity}")
        if not -100 <= note.panning <= 100:
            raise ConversionError(f"Invalid panning value: {note.panning}")
        if not -1200 <= note.pitch <= 1200:
            raise ConversionError(f"Invalid fine-pitch value: {note.pitch}")

        position = (note.tick, note.layer)
        if position in occupied:
            raise ConversionError(
                f"Multiple notes occupy the same tick and layer: {position}"
            )
        occupied.add(position)

    return result


def write_nbs(
    output_path: Path,
    notes: Iterable[NbsNote],
    layer_names: Sequence[str],
    ticks_per_second: float,
    *,
    title: str,
    author: str,
    source_name: str,
    time_signature: int = 4,
) -> None:
    """Write an Open Note Block Studio v5 file atomically."""

    if not layer_names:
        raise ConversionError("An NBS song must contain at least one layer.")
    if len(layer_names) > 65_535:
        raise ConversionError("The layer count exceeds the NBS limit.")
    if not 0 < ticks_per_second <= 655.35:
        raise ConversionError("NBS tempo must be greater than 0 and at most 655.35.")
    if not 2 <= time_signature <= 8:
        raise ConversionError("The time-signature numerator must be between 2 and 8.")

    sorted_notes = _validated_notes(notes, len(layer_names))
    song_length = sorted_notes[-1].tick if sorted_notes else 0
    grouped: dict[int, list[NbsNote]] = defaultdict(list)
    for note in sorted_notes:
        grouped[note.tick].append(note)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as file_obj:
            temporary_name = file_obj.name

            # Header (Open Note Block Studio format version 5).
            _write_u16(file_obj, 0)
            _write_u8(file_obj, NBS_VERSION)
            _write_u8(file_obj, VANILLA_INSTRUMENT_COUNT)
            _write_u16(file_obj, song_length)
            _write_u16(file_obj, len(layer_names))
            _write_string(file_obj, title)
            _write_string(file_obj, author)
            _write_string(file_obj, "")  # Original author
            _write_string(
                file_obj,
                "Audio converted by mp3_to_nbs.py (approximate transcription)",
            )
            _write_u16(file_obj, _round_tick(ticks_per_second * 100.0))
            _write_u8(file_obj, 0)  # Auto-save disabled
            _write_u8(file_obj, 10)
            _write_u8(file_obj, time_signature)
            _write_u32(file_obj, 0)  # Minutes spent
            _write_u32(file_obj, 0)  # Left clicks
            _write_u32(file_obj, 0)  # Right clicks
            _write_u32(file_obj, len(sorted_notes))
            _write_u32(file_obj, 0)  # Blocks removed
            _write_string(file_obj, source_name)
            _write_u8(file_obj, 0)  # Loop disabled
            _write_u8(file_obj, 0)
            _write_u16(file_obj, 0)

            # Notes are delta-encoded from tick -1 and layer -1.
            current_tick = -1
            for tick, chord in grouped.items():
                tick_jump = tick - current_tick
                if tick_jump > 65_535:
                    raise ConversionError("A silent interval is too long for the NBS format.")
                _write_u16(file_obj, tick_jump)
                current_tick = tick
                current_layer = -1
                for note in chord:
                    _write_u16(file_obj, note.layer - current_layer)
                    current_layer = note.layer
                    _write_u8(file_obj, note.instrument)
                    _write_u8(file_obj, note.key)
                    _write_u8(file_obj, note.velocity)
                    _write_u8(file_obj, note.panning + 100)
                    _write_i16(file_obj, note.pitch)
                _write_u16(file_obj, 0)
            _write_u16(file_obj, 0)

            # Layers.
            for name in layer_names:
                _write_string(file_obj, name)
                _write_u8(file_obj, 0)  # Unlocked
                _write_u8(file_obj, 100)
                _write_u8(file_obj, 100)  # Center

            _write_u8(file_obj, 0)  # No custom instruments
            file_obj.flush()
            os.fsync(file_obj.fileno())

        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _load_audio_dependencies():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ConversionError(
            "Audio analysis dependencies are missing. Run "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return librosa, np


def _load_basic_pitch_model():
    """Load Basic Pitch through its lightweight ONNX backend."""

    required_modules = ("basic_pitch", "onnxruntime", "pretty_midi", "resampy")
    if any(importlib.util.find_spec(name) is None for name in required_modules):
        raise ConversionError(
            "Basic Pitch is required for AI transcription. Run setup.bat first."
        )

    previous_logging_level = logging.root.manager.disable
    try:
        # Basic Pitch advertises every optional backend it cannot find.  ONNX is
        # the intended backend here, so those warnings would only confuse users.
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            logging.disable(logging.WARNING)
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model, predict

            model = Model(ICASSP_2022_MODEL_PATH)
    except Exception as exc:
        raise ConversionError(
            "Could not load the Basic Pitch model. Run setup.bat again."
        ) from exc
    finally:
        logging.disable(previous_logging_level)
    return model, predict


def _midi_to_frequency(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _build_pitch_analysis(
    audio,
    sample_rate: int,
    librosa,
    np,
    *,
    hop_length: int = 128,
) -> _PitchAnalysis | None:
    """Build one pitch-resolved loudness and attack representation.

    A broadband RMS envelope cannot tell whether a particular candidate pitch
    is audible: one loud chord tone raises the RMS for every candidate at that
    time.  This representation retains one magnitude and positive spectral-flux
    track per semitone, all on the already delay-corrected source timeline.
    """

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if (
        samples.size == 0
        or sample_rate <= 0
        or float(np.max(np.abs(samples))) < 1e-7
    ):
        return None

    # Keep roughly 6-12 ms timing resolution while retaining a CQT-compatible
    # power-of-two hop at unusually low sample rates used by small unit tests.
    if sample_rate < 8_000:
        target = max(16, int(round(sample_rate * 0.008)))
        hop_length = 1 << max(4, int(round(math.log2(target))))
    else:
        hop_length = min(max(64, hop_length), 256)
        hop_length = 1 << int(round(math.log2(hop_length)))

    nyquist = sample_rate * 0.5 * 0.94
    if nyquist <= _midi_to_frequency(NBS_LOWEST_MIDI):
        return None
    maximum_midi = min(
        NBS_LOWEST_MIDI + NBS_KEY_MAX,
        int(math.floor(69.0 + 12.0 * math.log2(nyquist / 440.0))),
    )
    n_bins = maximum_midi - NBS_LOWEST_MIDI + 1
    if n_bins <= 0:
        return None

    try:
        magnitude = np.abs(
            librosa.cqt(
                samples,
                sr=sample_rate,
                hop_length=hop_length,
                fmin=_midi_to_frequency(NBS_LOWEST_MIDI),
                n_bins=n_bins,
                bins_per_octave=12,
            )
        ).astype(np.float32, copy=False)
    except Exception:
        return None
    magnitude = np.nan_to_num(magnitude, copy=False)
    positive = magnitude[magnitude > 0.0]
    if magnitude.size == 0 or positive.size == 0:
        return None

    # Use the extreme but robust upper tail as 0 dB.  A 99.5th-percentile
    # reference is below many ordinary note attacks in a sparse CQT and causes
    # most measured notes to saturate at the same loudness.
    global_reference = max(float(np.quantile(positive, 0.999)), 1e-12)
    floor = global_reference * 1e-5
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, floor) / global_reference)
    flux = np.maximum(
        0.0,
        np.diff(
            magnitude_db,
            axis=1,
            prepend=magnitude_db[:, :1],
        ),
    ).astype(np.float32, copy=False)
    if flux.shape[1]:
        flux[:, 0] = 0.0

    onset_fft = 512 if sample_rate >= 8_000 else 128
    onset_hop_length = onset_fft // 8
    try:
        onset_magnitude = np.abs(
            librosa.stft(
                samples,
                n_fft=onset_fft,
                hop_length=onset_hop_length,
                win_length=onset_fft,
                window="hann",
                center=True,
            )
        ).astype(np.float32, copy=False)
        onset_magnitude = np.nan_to_num(onset_magnitude, copy=False)
        onset_frequencies = np.asarray(
            librosa.fft_frequencies(sr=sample_rate, n_fft=onset_fft),
            dtype=np.float32,
        )
    except Exception:
        onset_magnitude = np.empty((0, 0), dtype=np.float32)
        onset_frequencies = np.empty(0, dtype=np.float32)

    return _PitchAnalysis(
        magnitude=magnitude,
        flux=flux,
        hop_length=hop_length,
        onset_magnitude=onset_magnitude,
        onset_hop_length=onset_hop_length,
        onset_frequencies=onset_frequencies,
        sample_rate=sample_rate,
        global_reference=global_reference,
    )


def _pitch_analysis_tracks(
    analysis: _PitchAnalysis,
    midi: int,
    np,
):
    """Return conservative detuning-tolerant magnitude and flux tracks."""

    pitch_index = midi - NBS_LOWEST_MIDI
    if not 0 <= pitch_index < analysis.magnitude.shape[0]:
        return None
    magnitude_track = np.asarray(analysis.magnitude[pitch_index], dtype=np.float32)
    flux_track = np.asarray(analysis.flux[pitch_index], dtype=np.float32)

    # Adjacent semitone bins may contain a detuned source, but they must not make
    # an unsupported neighboring MIDI note look fully present.  A small spill
    # allowance handles tuning error without reproducing the old broadband gate.
    adjacent_magnitude = []
    adjacent_flux = []
    if pitch_index > 0:
        adjacent_magnitude.append(analysis.magnitude[pitch_index - 1])
        adjacent_flux.append(analysis.flux[pitch_index - 1])
    if pitch_index + 1 < analysis.magnitude.shape[0]:
        adjacent_magnitude.append(analysis.magnitude[pitch_index + 1])
        adjacent_flux.append(analysis.flux[pitch_index + 1])
    if adjacent_magnitude:
        magnitude_track = np.maximum(
            magnitude_track,
            0.18 * np.maximum.reduce(adjacent_magnitude),
        )
        flux_track = np.maximum(
            flux_track,
            0.18 * np.maximum.reduce(adjacent_flux),
        )
    return pitch_index, magnitude_track, flux_track


def _time_to_analysis_frame(seconds: float, analysis: _PitchAnalysis) -> int:
    return _round_tick(seconds * analysis.sample_rate / analysis.hop_length)


def _pitch_onset_flux_track(
    analysis: _PitchAnalysis,
    midi: int,
    fallback_flux,
    np,
):
    """Return a short-window harmonic attack track for one source pitch."""

    if (
        analysis.onset_magnitude.size == 0
        or analysis.onset_frequencies.size < 2
    ):
        return fallback_flux, analysis.hop_length
    fundamental = _midi_to_frequency(midi)
    nyquist = analysis.sample_rate * 0.5
    harmonic_tracks = []
    harmonic_weights = []
    bin_width = float(
        analysis.onset_frequencies[1] - analysis.onset_frequencies[0]
    )
    for harmonic in range(1, 9):
        frequency = fundamental * harmonic
        if frequency >= nyquist * 0.96:
            break
        center_bin = int(round(frequency / max(bin_width, 1e-9)))
        if not 0 <= center_bin < analysis.onset_magnitude.shape[0]:
            continue
        # One neighboring FFT bin handles ordinary tuning error.  The CQT gate,
        # not this attack-only track, remains responsible for pitch identity.
        low = max(0, center_bin - 1)
        high = min(analysis.onset_magnitude.shape[0], center_bin + 2)
        harmonic_tracks.append(
            np.max(analysis.onset_magnitude[low:high], axis=0)
        )
        harmonic_weights.append(1.0 / math.sqrt(harmonic))
    if not harmonic_tracks:
        return fallback_flux, analysis.hop_length
    stacked = np.stack(harmonic_tracks, axis=0)
    weights = np.asarray(harmonic_weights, dtype=np.float32)[:, None]
    energy = np.sum(stacked * weights, axis=0) / max(
        float(np.sum(weights)), 1e-9
    )
    positive = energy[energy > 0.0]
    scale = max(float(np.quantile(positive, 0.50)), 1e-9) if positive.size else 1.0
    log_energy = np.log1p(energy / scale)
    flux = np.maximum(
        0.0,
        np.diff(log_energy, prepend=log_energy[:1]),
    ).astype(np.float32, copy=False)
    if flux.size:
        flux[0] = 0.0
    return flux, analysis.onset_hop_length


def _measure_timed_pitch_event(
    event: _TimedPitchEvent,
    analysis: _PitchAnalysis,
    sensitivity: float,
    np,
    *,
    onset_tolerance_seconds: float,
    tracks=None,
    onset_data=None,
) -> _TimedPitchEvent:
    """Refine one onset once and attach independent source evidence."""

    if tracks is None:
        tracks = _pitch_analysis_tracks(analysis, event.midi, np)
    model_confidence = _clamp(
        event.model_confidence
        if event.model_confidence is not None
        else event.amplitude,
        0.0,
        1.0,
    )
    if tracks is None:
        return replace(event, model_confidence=model_confidence)
    pitch_index, magnitude_track, cqt_flux_track = tracks
    magnitude_frame_count = len(magnitude_track)
    if magnitude_frame_count == 0:
        return replace(event, model_confidence=model_confidence)

    original_start = max(0.0, event.start_seconds)
    if onset_data is None:
        flux_track, onset_hop_length = _pitch_onset_flux_track(
            analysis, event.midi, cqt_flux_track, np
        )
        positive_flux = flux_track[flux_track > 0.0]
        flux_reference = (
            max(float(np.quantile(positive_flux, 0.95)), 1e-6)
            if positive_flux.size
            else 1.0
        )
    else:
        flux_track, onset_hop_length, flux_reference = onset_data
    onset_frame_count = len(flux_track)
    center_frame = int(
        _clamp(
            _round_tick(
                original_start * analysis.sample_rate / onset_hop_length
            ),
            0,
            max(0, onset_frame_count - 1),
        )
    )
    radius = max(
        1,
        math.ceil(
            onset_tolerance_seconds
            * analysis.sample_rate
            / onset_hop_length
        ),
    )
    search_start = max(0, center_frame - radius)
    search_end = min(onset_frame_count, center_frame + radius + 1)
    onset_window = flux_track[search_start:search_end]
    onset_frame = center_frame
    onset_strength = 0.0
    if onset_window.size:
        local_baseline = float(np.median(onset_window))
        candidate_frames: list[int] = []
        for local_index, value in enumerate(onset_window):
            left = (
                float(onset_window[local_index - 1])
                if local_index > 0
                else -math.inf
            )
            right = (
                float(onset_window[local_index + 1])
                if local_index + 1 < len(onset_window)
                else -math.inf
            )
            if float(value) >= left and float(value) > right:
                candidate_frames.append(search_start + local_index)
        if candidate_frames:
            onset_frame = max(
                candidate_frames,
                key=lambda frame: (
                    _clamp(float(flux_track[frame]) / flux_reference, 0.0, 1.5)
                    - 0.28 * abs(frame - center_frame) / max(1, radius),
                    -abs(frame - center_frame),
                ),
            )
            onset_strength = _clamp(
                float(flux_track[onset_frame]) / flux_reference, 0.0, 1.0
            )
            minimum_onset_support = 0.24 - 0.14 * sensitivity
            if (
                onset_strength < minimum_onset_support
                or float(flux_track[onset_frame])
                < max(0.02, local_baseline * 1.35)
            ):
                onset_frame = center_frame
                onset_strength = _clamp(
                    float(flux_track[center_frame]) / flux_reference, 0.0, 1.0
                )

    refined_start = onset_frame * onset_hop_length / analysis.sample_rate
    if abs(refined_start - original_start) > onset_tolerance_seconds + 1e-9:
        refined_start = original_start
    shift = refined_start - original_start
    refined_end = max(refined_start + 1e-4, event.end_seconds + shift)

    attack_start = max(
        0,
        _time_to_analysis_frame(refined_start - 0.02, analysis),
    )
    attack_end = min(
        magnitude_frame_count,
        _time_to_analysis_frame(refined_start + 0.11, analysis) + 1,
    )
    body_end_seconds = min(refined_end, refined_start + 0.45)
    body_start = max(0, _time_to_analysis_frame(refined_start, analysis))
    body_end = min(
        magnitude_frame_count,
        max(body_start + 1, _time_to_analysis_frame(body_end_seconds, analysis) + 1),
    )
    attack_values = magnitude_track[attack_start:attack_end]
    body_values = magnitude_track[body_start:body_end]
    signal = max(
        float(np.quantile(attack_values, 0.80)) if attack_values.size else 0.0,
        float(np.quantile(body_values, 0.65)) if body_values.size else 0.0,
        1e-12,
    )
    absolute_db = 20.0 * math.log10(signal / analysis.global_reference)
    pitch_loudness = _clamp((absolute_db + 50.0) / 50.0, 0.0, 1.0)

    neighborhood = []
    for distance in range(2, 7):
        for neighbor in (pitch_index - distance, pitch_index + distance):
            if 0 <= neighbor < analysis.magnitude.shape[0]:
                values = analysis.magnitude[neighbor, body_start:body_end]
                if values.size:
                    neighborhood.append(float(np.quantile(values, 0.65)))
    spectral_noise = float(np.median(neighborhood)) if neighborhood else 0.0
    pre_start = max(0, _time_to_analysis_frame(refined_start - 0.35, analysis))
    pre_end = max(pre_start, _time_to_analysis_frame(refined_start - 0.06, analysis))
    pre_values = magnitude_track[pre_start:min(pre_end, magnitude_frame_count)]
    temporal_noise = (
        float(np.quantile(pre_values, 0.65)) if pre_values.size else 0.0
    )
    noise = max(
        analysis.global_reference * 1e-5,
        spectral_noise,
        temporal_noise * 0.65,
    )
    pitch_snr_db = 20.0 * math.log10(signal / noise)
    snr_support = _clamp((pitch_snr_db - 1.0) / 20.0, 0.0, 1.0)
    duration_support = _clamp((refined_end - refined_start) / 0.30, 0.0, 1.0)
    candidate_score = _clamp(
        0.34 * model_confidence
        + 0.34 * pitch_loudness
        + 0.20 * snr_support
        + 0.08 * onset_strength
        + 0.04 * duration_support,
        0.0,
        1.0,
    )
    return replace(
        event,
        start_seconds=refined_start,
        end_seconds=refined_end,
        amplitude=candidate_score,
        model_confidence=model_confidence,
        pitch_loudness=pitch_loudness,
        onset_strength=onset_strength,
        pitch_snr_db=pitch_snr_db,
    )


def _timed_pitch_event_is_supported(
    event: _TimedPitchEvent,
    role: str,
    sensitivity: float,
) -> bool:
    """Reject a neural candidate only when its own source pitch is unsupported."""

    if event.pitch_loudness is None or event.pitch_snr_db is None:
        return True
    model_confidence = (
        event.model_confidence
        if event.model_confidence is not None
        else event.amplitude
    )
    duration = max(0.0, event.end_seconds - event.start_seconds)
    loudness_floor = 0.10 - 0.055 * sensitivity
    score_floor = (
        0.30 if role in {"vocals", "bass"} else 0.34
    ) - 0.06 * sensitivity
    quiet_but_clear = (
        event.pitch_snr_db >= 8.0
        and (event.onset_strength or 0.0) >= 0.20
        and event.pitch_loudness >= 0.015
    )
    if (
        event.pitch_loudness < loudness_floor
        and model_confidence < 0.84
        and not quiet_but_clear
    ):
        return False
    if (
        event.pitch_snr_db < -1.0
        and event.pitch_loudness < 0.30
        and model_confidence < 0.80
    ):
        return False
    if event.amplitude < score_floor:
        return False
    if (
        duration < 0.12
        and (event.onset_strength or 0.0) < 0.06
        and model_confidence < 0.72
    ):
        return False
    return True


def _analyze_timed_pitch_events(
    events: Iterable[_TimedPitchEvent],
    audio,
    sample_rate: int,
    config: ConversionConfig,
    librosa,
    np,
    *,
    role: str,
    reject_unsupported: bool = True,
) -> list[_TimedPitchEvent]:
    """Measure pitch-specific evidence and establish one canonical onset."""

    event_list = list(events)
    if not event_list:
        return []
    analysis = _build_pitch_analysis(audio, sample_rate, librosa, np)
    if analysis is None:
        return [
            replace(
                event,
                model_confidence=(
                    event.model_confidence
                    if event.model_confidence is not None
                    else event.amplitude
                ),
            )
            for event in event_list
        ]
    onset_tolerances = {
        "vocals": 0.070,
        "bass": 0.060,
        "other": 0.065,
        "other_transient": 0.050,
        "guitar": 0.060,
        "piano": 0.060,
    }
    role_tolerance = onset_tolerances.get(role, 0.060)
    starts_by_midi: dict[int, list[float]] = defaultdict(list)
    for event in event_list:
        starts_by_midi[event.midi].append(event.start_seconds)
    local_tolerances: list[float] = []
    for event in event_list:
        neighboring_distances = [
            abs(other_start - event.start_seconds)
            for other_start in starts_by_midi[event.midi]
            if abs(other_start - event.start_seconds) > 1e-9
        ]
        if neighboring_distances:
            # Two rapid strikes of one pitch must not both refine to the
            # stronger attack in their overlapping search windows.  Keep each
            # search inside slightly less than half the nearest model-onset
            # distance; a lone note retains the wider decoder-correction range.
            role_local_tolerance = min(
                role_tolerance,
                max(0.006, 0.45 * min(neighboring_distances)),
            )
        else:
            role_local_tolerance = role_tolerance
        local_tolerances.append(role_local_tolerance)
    tracks_by_midi = {
        midi: _pitch_analysis_tracks(analysis, midi, np)
        for midi in {event.midi for event in event_list}
    }
    onset_by_midi = {}
    for midi, tracks in tracks_by_midi.items():
        if tracks is None:
            continue
        _pitch_index, _magnitude_track, cqt_flux_track = tracks
        flux_track, onset_hop_length = _pitch_onset_flux_track(
            analysis, midi, cqt_flux_track, np
        )
        positive_flux = flux_track[flux_track > 0.0]
        flux_reference = (
            max(float(np.quantile(positive_flux, 0.95)), 1e-6)
            if positive_flux.size
            else 1.0
        )
        onset_by_midi[midi] = (
            flux_track,
            onset_hop_length,
            flux_reference,
        )
    measured = [
        _measure_timed_pitch_event(
            event,
            analysis,
            config.sensitivity,
            np,
            onset_tolerance_seconds=local_tolerance,
            tracks=tracks_by_midi.get(event.midi),
            onset_data=onset_by_midi.get(event.midi),
        )
        for event, local_tolerance in zip(event_list, local_tolerances)
    ]
    if reject_unsupported:
        measured = [
            event
            for event in measured
            if _timed_pitch_event_is_supported(event, role, config.sensitivity)
        ]
    return sorted(measured, key=lambda event: (event.start_seconds, event.midi))


def _predict_timed_pitch_events(
    stem_path: Path,
    model,
    predict,
    *,
    role: str,
    sensitivity: float,
    duration_seconds: float | None = None,
    timing_offset_seconds: float = 0.0,
    adaptive_recovery: bool = False,
    progress_update: Callable[[float], None] | None = None,
) -> list[_TimedPitchEvent]:
    """Transcribe a separated stem into confidence-weighted note events."""

    full_low = NBS_LOWEST_MIDI
    full_high = NBS_LOWEST_MIDI + NBS_KEY_MAX
    settings = {
        # onset, frame, minimum length (ms), minimum MIDI, maximum MIDI.
        # Stem names are useful priors for arrangement, not safe hard pitch
        # limits: piano left hands leak into ``other``, piccolo and soprano can
        # exceed C7, and extended basses reach below E1.  Basic Pitch and NBS
        # both natively span A0..C8, so every pass retains that complete range
        # and lets source-resolved evidence reject unsupported harmonics.
        "vocals": (0.52, 0.34, 35.0, full_low, full_high),
        "bass": (0.54, 0.37, 40.0, full_low, full_high),
        "other": (0.64, 0.44, 80.0, full_low, full_high),
        "other_transient": (0.53, 0.35, 30.0, full_low, full_high),
        "guitar": (0.56, 0.38, 40.0, full_low, full_high),
        "piano": (0.58, 0.40, 35.0, full_low, full_high),
    }
    if role not in settings:
        raise ValueError(f"Unknown transcription role: {role}")
    base_onset, base_frame, base_length, minimum_midi, maximum_midi = settings[role]
    onset_threshold = _clamp(
        base_onset + (0.5 - sensitivity) * 0.24, 0.30, 0.85
    )
    frame_threshold = _clamp(
        base_frame + (0.5 - sensitivity) * 0.20, 0.25, 0.75
    )
    minimum_note_length = base_length * (1.25 - 0.50 * sensitivity)
    if adaptive_recovery:
        # The model input has already been compressed locally.  Modestly lower
        # neural gates to expose the newly audible weak phrase, then rely on the
        # untouched source stem for the strict acceptance decision below.
        onset_threshold = max(0.28, onset_threshold - 0.07)
        frame_threshold = max(0.22, frame_threshold - 0.06)
        minimum_note_length *= 0.80

    prediction_model = model
    if progress_update is not None:
        try:
            from basic_pitch.constants import (
                AUDIO_N_SAMPLES,
                AUDIO_SAMPLE_RATE,
                FFT_HOP,
            )

            overlap_length = 30 * FFT_HOP
            hop_size = AUDIO_N_SAMPLES - overlap_length
            audio_samples = math.ceil(
                max(0.0, duration_seconds or 0.0) * AUDIO_SAMPLE_RATE
            ) + overlap_length // 2
            total_windows = max(1, math.ceil(audio_samples / hop_size))
            model_class = type(model)

            class _ProgressBasicPitchModel(model_class):
                def __init__(self, wrapped_model):
                    self.wrapped_model = wrapped_model
                    self.completed_windows = 0

                def predict(self, audio_window):
                    result = self.wrapped_model.predict(audio_window)
                    self.completed_windows += 1
                    progress_update(
                        min(1.0, self.completed_windows / total_windows)
                    )
                    return result

            prediction_model = _ProgressBasicPitchModel(model)
            progress_update(0.0)
        except (ImportError, TypeError):
            # Progress remains stage-based if a future Basic Pitch release
            # changes its internal model/window API.
            prediction_model = model

    try:
        # The library prints a separate English status line for every stem.
        # Progress is already reported by the caller.
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Consume Basic Pitch's seconds-stamped note events directly.  Do
            # not reconstruct times from its concatenated raw frame matrices:
            # window/frame strides are not guaranteed to form one exact clock
            # over a long recording.
            _, _, raw_events = predict(
                stem_path,
                prediction_model,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=minimum_note_length,
                minimum_frequency=_midi_to_frequency(minimum_midi),
                maximum_frequency=_midi_to_frequency(maximum_midi),
                multiple_pitch_bends=False,
                # This recovery pass is valuable for a smooth solo line, but it
                # tends to invent extra notes in a polyphonic accompaniment.
                melodia_trick=role not in {"other", "other_transient", "guitar", "piano"},
            )
    except Exception as exc:
        raise ConversionError(
            f"AI transcription failed for the {role} stem."
        ) from exc
    if progress_update is not None:
        progress_update(1.0)

    events: list[_TimedPitchEvent] = []
    for raw_event in raw_events:
        if len(raw_event) < 4:
            continue
        start_seconds = float(raw_event[0]) + timing_offset_seconds
        end_seconds = float(raw_event[1]) + timing_offset_seconds
        midi = int(raw_event[2])
        amplitude = float(raw_event[3])
        if not all(
            math.isfinite(value)
            for value in (start_seconds, end_seconds, amplitude)
        ):
            continue
        if (
            end_seconds <= max(0.0, start_seconds)
            or not minimum_midi <= midi <= maximum_midi
        ):
            continue
        events.append(
            _TimedPitchEvent(
                start_seconds=max(0.0, start_seconds),
                end_seconds=end_seconds,
                midi=midi,
                amplitude=_clamp(amplitude, 0.0, 1.0),
                source_role=(
                    role if role in BACKGROUND_ROLE_INSTRUMENTS else None
                ),
                model_confidence=_clamp(amplitude, 0.0, 1.0),
            )
        )
    return sorted(events, key=lambda event: (event.start_seconds, event.midi))


def _coalesce_polyphonic_onsets(
    events: Iterable[_TimedPitchEvent],
    *,
    tolerance_seconds: float = 0.020,
    maximum_shift_seconds: float = 0.0125,
) -> list[_TimedPitchEvent]:
    """Coalesce only sub-grid model jitter, preserving audible strums.

    Pitch-resolved attack refinement has already put every event on the source
    timeline.  Moving chord tones by the former 55 ms tolerance could therefore
    turn a real strum into a block chord or audibly delay its first tone.  This
    pass now removes only smaller-than-one-precise-tick decoder jitter, and no
    event is permitted to move by more than half a 40 TPS tick.
    """

    event_list = sorted(events, key=lambda event: (event.start_seconds, event.midi))
    if len(event_list) < 2:
        return event_list

    def overlaps_as_chord(first: _TimedPitchEvent, second: _TimedPitchEvent) -> bool:
        if first.midi == second.midi:
            return False
        first_duration = max(0.0, first.end_seconds - first.start_seconds)
        second_duration = max(0.0, second.end_seconds - second.start_seconds)
        shorter_duration = min(first_duration, second_duration)
        overlap = min(first.end_seconds, second.end_seconds) - max(
            first.start_seconds, second.start_seconds
        )
        required_overlap = min(0.10, shorter_duration * 0.55)
        return shorter_duration > 0.0 and overlap >= required_overlap

    groups: list[list[_TimedPitchEvent]] = []
    for event in event_list:
        group = groups[-1] if groups else None
        if (
            group
            and event.start_seconds - group[0].start_seconds <= tolerance_seconds
            and all(overlaps_as_chord(existing, event) for existing in group)
        ):
            group.append(event)
        else:
            groups.append([event])

    result: list[_TimedPitchEvent] = []
    for group in groups:
        if len(group) < 2:
            result.extend(group)
            continue
        ordered_starts = sorted(event.start_seconds for event in group)
        anchor = ordered_starts[(len(ordered_starts) - 1) // 2]
        if any(
            abs(event.start_seconds - anchor) > maximum_shift_seconds
            for event in group
        ):
            result.extend(group)
            continue
        for event in group:
            shift = anchor - event.start_seconds
            result.append(
                replace(
                    event,
                    start_seconds=anchor,
                    end_seconds=max(anchor + 1e-4, event.end_seconds + shift),
                )
            )
    return sorted(result, key=lambda event: (event.start_seconds, event.midi))


def _timed_events_are_duplicate(
    first: _TimedPitchEvent,
    second: _TimedPitchEvent,
    *,
    onset_tolerance_seconds: float = 0.08,
) -> bool:
    """Return true only for two events that look like the same leaked note."""

    if first.midi != second.midi:
        return False
    first_duration = max(0.0, first.end_seconds - first.start_seconds)
    second_duration = max(0.0, second.end_seconds - second.start_seconds)
    shorter = min(first_duration, second_duration)
    longer = max(first_duration, second_duration)
    if shorter <= 0.0 or longer <= 0.0:
        return False
    overlap = max(
        0.0,
        min(first.end_seconds, second.end_seconds)
        - max(first.start_seconds, second.start_seconds),
    )
    return (
        abs(first.start_seconds - second.start_seconds)
        <= onset_tolerance_seconds
        and overlap / shorter >= 0.70
        and shorter / longer >= 0.45
    )


def _merge_accompaniment_passes(
    primary: Iterable[_TimedPitchEvent],
    transient_recovery: Iterable[_TimedPitchEvent],
    duration_seconds: float,
) -> list[_TimedPitchEvent]:
    """Add credible short attacks from a sensitive second transcription pass."""

    retained = list(primary)
    by_midi: dict[int, list[_TimedPitchEvent]] = defaultdict(list)
    for event in retained:
        by_midi[event.midi].append(event)

    for event in transient_recovery:
        event_duration = event.end_seconds - event.start_seconds
        if event_duration <= 0.0 or event.amplitude < 0.30:
            continue
        if any(
            _timed_events_are_duplicate(event, existing)
            for existing in by_midi.get(event.midi, ())
        ):
            continue
        retained.append(event)
        by_midi[event.midi].append(event)

    return sorted(retained, key=lambda event: (event.start_seconds, event.midi))


def _fuse_accompaniment_events(
    primary: Iterable[_TimedPitchEvent],
    transient_recovery: Iterable[_TimedPitchEvent],
    audio,
    sample_rate: int,
    config: ConversionConfig,
    librosa,
    np,
    *,
    role: str = "other",
) -> list[_TimedPitchEvent]:
    """Fuse passes using independent model, loudness, S/N and onset evidence."""

    primary_events = list(primary)
    recovery_events = list(transient_recovery)
    if not primary_events and not recovery_events:
        return []

    measured_primary = _analyze_timed_pitch_events(
        primary_events,
        audio,
        sample_rate,
        config,
        librosa,
        np,
        role=role,
        reject_unsupported=False,
    )
    recovery_role = "other_transient" if role == "other" else role
    measured_recovery = _analyze_timed_pitch_events(
        recovery_events,
        audio,
        sample_rate,
        config,
        librosa,
        np,
        role=recovery_role,
        reject_unsupported=False,
    )
    if all(
        event.pitch_loudness is None
        for event in measured_primary + measured_recovery
    ):
        return _merge_accompaniment_passes(
            primary_events,
            recovery_events,
            len(audio) / max(1, sample_rate),
        )

    recovery_by_midi: dict[int, list[_TimedPitchEvent]] = defaultdict(list)
    for event in measured_recovery:
        recovery_by_midi[event.midi].append(event)

    retained: list[_TimedPitchEvent] = []
    retained_by_midi: dict[int, list[_TimedPitchEvent]] = defaultdict(list)
    primary_floor = 0.34 - 0.06 * config.sensitivity
    for event in measured_primary:
        consensus = any(
            _timed_events_are_duplicate(event, other)
            for other in recovery_by_midi.get(event.midi, ())
        )
        score = _clamp(event.amplitude + (0.06 if consensus else 0.0), 0.0, 1.0)
        calibrated = replace(event, amplitude=score)
        if score < primary_floor or not _timed_pitch_event_is_supported(
            calibrated, role, config.sensitivity
        ):
            continue
        retained.append(calibrated)
        retained_by_midi[event.midi].append(calibrated)

    duration_seconds = len(audio) / max(1, sample_rate)
    recovery_floor = 0.40 - 0.05 * config.sensitivity
    for event in measured_recovery:
        duration = event.end_seconds - event.start_seconds
        model_confidence = (
            event.model_confidence
            if event.model_confidence is not None
            else event.amplitude
        )
        if duration <= 0.0 or model_confidence < 0.30:
            continue
        if any(
            _timed_events_are_duplicate(event, existing)
            for existing in retained_by_midi.get(event.midi, ())
        ):
            continue
        if event.amplitude < recovery_floor or not _timed_pitch_event_is_supported(
            event, recovery_role, config.sensitivity
        ):
            continue
        if (event.onset_strength or 0.0) < 0.10 and duration < 0.18:
            continue
        retained.append(event)
        retained_by_midi[event.midi].append(event)

    return sorted(retained, key=lambda event: (event.start_seconds, event.midi))


def _merge_adaptive_recovery_events(
    primary: Iterable[_TimedPitchEvent],
    adaptive: Iterable[_TimedPitchEvent],
    duration_seconds: float,
    sensitivity: float,
    *,
    role: str,
) -> list[_TimedPitchEvent]:
    """Restore locally quiet phrases only when the untouched stem supports them."""

    retained = list(primary)
    candidates = list(adaptive)
    if not candidates:
        return sorted(retained, key=lambda event: (event.start_seconds, event.midi))
    evidence_role = "other_transient" if role == "other" else role

    retained_by_midi: dict[int, list[int]] = defaultdict(list)
    for index, event in enumerate(retained):
        retained_by_midi[event.midi].append(index)

    # Agreement with the ordinary pass is valuable evidence and must not create
    # a doubled attack.  Preserve its canonical source time while combining the
    # strongest independent measurements.
    unmatched: list[_TimedPitchEvent] = []
    for event in candidates:
        matching_index = next(
            (
                index
                for index in retained_by_midi.get(event.midi, ())
                if abs(retained[index].start_seconds - event.start_seconds) <= 0.09
            ),
            None,
        )
        if matching_index is None:
            unmatched.append(event)
            continue
        existing = retained[matching_index]

        def maximum_optional(first: float | None, second: float | None):
            values = [value for value in (first, second) if value is not None]
            return max(values) if values else None

        retained[matching_index] = replace(
            existing,
            end_seconds=max(existing.end_seconds, event.end_seconds),
            amplitude=_clamp(max(existing.amplitude, event.amplitude) + 0.025, 0.0, 1.0),
            model_confidence=maximum_optional(
                existing.model_confidence, event.model_confidence
            ),
            pitch_loudness=maximum_optional(
                existing.pitch_loudness, event.pitch_loudness
            ),
            onset_strength=maximum_optional(
                existing.onset_strength, event.onset_strength
            ),
            pitch_snr_db=maximum_optional(existing.pitch_snr_db, event.pitch_snr_db),
            source_role=existing.source_role or event.source_role,
        )

    occurrences: dict[int, int] = defaultdict(int)
    sections: dict[int, set[int]] = defaultdict(set)
    section_seconds = _clamp(max(0.0, duration_seconds) / 20.0, 3.0, 8.0)
    for event in unmatched:
        occurrences[event.midi] += 1
        sections[event.midi].add(int(event.start_seconds / section_seconds))
    all_phrase_events = sorted(
        retained + unmatched, key=lambda event: (event.start_seconds, event.midi)
    )

    ranked: list[tuple[float, _TimedPitchEvent]] = []
    for event in unmatched:
        if not _timed_pitch_event_is_supported(event, evidence_role, sensitivity):
            continue
        model_confidence = _clamp(
            event.model_confidence
            if event.model_confidence is not None
            else event.amplitude,
            0.0,
            1.0,
        )
        loudness = event.pitch_loudness or 0.0
        snr_db = event.pitch_snr_db if event.pitch_snr_db is not None else -math.inf
        onset = event.onset_strength or 0.0
        event_duration = max(0.0, event.end_seconds - event.start_seconds)
        if model_confidence < 0.28 or event_duration < 0.045:
            continue
        if loudness < 0.025 and snr_db < 12.0:
            continue
        if snr_db < 2.5 and loudness < 0.22:
            continue

        recurring = occurrences[event.midi] >= 2
        cross_section = len(sections[event.midi]) >= 2
        phrase_neighbor = any(
            other is not event
            and 0.04
            < abs(other.start_seconds - event.start_seconds)
            <= 1.35
            and (
                other.midi == event.midi
                or (
                    role in {"vocals", "bass"}
                    and abs(other.midi - event.midi) <= 7
                )
            )
            for other in all_phrase_events
        )
        physical_attack = onset >= 0.16
        sustained_support = event_duration >= 0.16 and snr_db >= 6.0
        exceptional = (
            model_confidence >= 0.34
            and snr_db >= 8.0
            and onset >= 0.30
        )
        if not (physical_attack or sustained_support):
            continue
        if role in {"other", "guitar", "piano"}:
            if not (exceptional or recurring or (cross_section and phrase_neighbor)):
                continue
        elif not (exceptional or phrase_neighbor or recurring):
            continue

        priority = (
            event.amplitude
            + 0.10 * _clamp((snr_db - 2.5) / 17.5, 0.0, 1.0)
            + 0.08 * onset
            + (0.05 if recurring else 0.0)
            + (0.04 if phrase_neighbor else 0.0)
        )
        ranked.append((priority, event))

    # Evidence gates above already operate per pitch and per physical attack.
    # A low whole-song notes-per-second quota erased legitimate trills and fast
    # quiet passages, so retain a safety ceiling that is well above playable
    # musical rates and serves only as a pathological-model guard.
    maximum_additions = max(
        64,
        math.ceil(max(0.0, duration_seconds) * 64.0),
    )
    if len(ranked) > maximum_additions:
        ranked = sorted(ranked, key=lambda item: item[0], reverse=True)[
            :maximum_additions
        ]
    retained.extend(event for _priority, event in ranked)
    return sorted(retained, key=lambda event: (event.start_seconds, event.midi))


def _merge_instrument_background_events(
    primary: Iterable[_TimedPitchEvent],
    background_by_role: dict[str, Iterable[_TimedPitchEvent]],
    duration_seconds: float,
) -> list[_TimedPitchEvent]:
    """Restore recurring instrument-stem notes without doubling the core part.

    A piano or guitar pass can expose notes masked in the combined accompaniment
    stem, but it can also contain separator bleed. The core transcription always
    wins a matching attack. A genuinely new background attack must recur across
    the song, or carry exceptional source-supported confidence, and each role
    receives a strict whole-song onset budget.
    """

    retained = list(primary)
    retained_by_midi: dict[int, list[_TimedPitchEvent]] = defaultdict(list)
    for event in retained:
        retained_by_midi[event.midi].append(event)

    section_seconds = _clamp(duration_seconds / 18.0, 4.0, 10.0)
    maximum_events_per_role = max(
        64, math.ceil(max(0.0, duration_seconds) * 64.0)
    )
    for role, role_material in background_by_role.items():
        if role not in BACKGROUND_ROLE_INSTRUMENTS:
            continue
        role_events = sorted(
            role_material, key=lambda event: (event.start_seconds, event.midi)
        )
        occurrences: dict[int, int] = defaultdict(int)
        sections: dict[int, set[int]] = defaultdict(set)
        for event in role_events:
            occurrences[event.midi] += 1
            sections[event.midi].add(int(event.start_seconds / section_seconds))

        candidates: list[tuple[float, _TimedPitchEvent]] = []
        for event in role_events:
            loudness = event.pitch_loudness or 0.0
            snr_db = (
                event.pitch_snr_db
                if event.pitch_snr_db is not None
                else -math.inf
            )
            onset = event.onset_strength or 0.0
            source_supported = (
                loudness >= 0.025 and snr_db >= 5.0 and onset >= 0.14
            )
            if event.amplitude < (0.34 if source_supported else 0.52):
                continue
            recurring = (
                occurrences[event.midi] >= 3
                and len(sections[event.midi]) >= 2
            )
            duration = event.end_seconds - event.start_seconds
            exceptional = (
                source_supported
                and duration >= 0.045
                and (
                    event.amplitude >= 0.54
                    or (
                        event.amplitude >= 0.34
                        and snr_db >= 10.0
                        and onset >= 0.30
                    )
                )
            )
            if not recurring and not exceptional:
                continue
            if any(
                abs(event.start_seconds - existing.start_seconds) <= 0.09
                for existing in retained_by_midi.get(event.midi, ())
            ):
                continue
            priority = (
                event.amplitude
                + 0.025 * min(8, occurrences[event.midi])
                + 0.04 * min(3, len(sections[event.midi]))
            )
            candidates.append((priority, replace(event, source_role=role)))

        if len(candidates) > maximum_events_per_role:
            candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[
                :maximum_events_per_role
            ]
        for _priority, event in sorted(
            candidates, key=lambda item: (item[1].start_seconds, item[1].midi)
        ):
            if any(
                abs(event.start_seconds - existing.start_seconds) <= 0.09
                for existing in retained_by_midi.get(event.midi, ())
            ):
                continue
            retained.append(event)
            retained_by_midi[event.midi].append(event)

    return sorted(retained, key=lambda event: (event.start_seconds, event.midi))


def _quantized_events_share_attack(
    previous: _QuantizedPitchEvent,
    current: _QuantizedPitchEvent,
    tick_seconds: float,
) -> bool:
    """Return whether two same-pitch candidates describe one source attack.

    Duration overlap alone is not sufficient: transcription models commonly
    let a held release overlap the next repeated strike.  Combining those
    events erased repeated piano, guitar, and percussion-like pitched notes.
    Only candidates whose canonical onsets are indistinguishable on the target
    grid are deduplicated.
    """

    previous_time = previous.source_time_seconds
    current_time = current.source_time_seconds
    if previous_time is None or current_time is None:
        return previous.start_tick == current.start_tick
    onset_distance = abs(current_time - previous_time)
    same_attack_tolerance = min(0.020, max(0.006, tick_seconds * 0.50))
    return onset_distance <= same_attack_tolerance


def _quantize_timed_pitch_events(
    events: Iterable[_TimedPitchEvent],
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    *,
    join_gap_ticks: int = -1,
) -> list[_QuantizedPitchEvent]:
    """Snap AI note boundaries to the NBS grid and merge overlapping duplicates."""

    by_pitch: dict[int, list[_QuantizedPitchEvent]] = defaultdict(list)
    for event in events:
        phase_tick = (
            event.start_seconds - timeline_origin_seconds
        ) / tick_seconds
        end_phase_tick = (
            event.end_seconds - timeline_origin_seconds
        ) / tick_seconds
        start_tick = _seconds_to_tick(
            event.start_seconds,
            timeline_origin_seconds,
            tick_seconds,
        )
        end_tick = math.ceil(
            (event.end_seconds - timeline_origin_seconds) / tick_seconds
        )
        if end_tick <= 0 or start_tick >= tick_count:
            continue
        start_tick = max(0, start_tick)
        end_tick = min(tick_count, max(start_tick + 1, end_tick))
        by_pitch[event.midi].append(
            _QuantizedPitchEvent(
                start_tick=start_tick,
                end_tick=end_tick,
                midi=event.midi,
                amplitude=event.amplitude,
                phase_tick=phase_tick,
                end_phase_tick=end_phase_tick,
                source_role=event.source_role,
                model_confidence=event.model_confidence,
                pitch_loudness=event.pitch_loudness,
                onset_strength=event.onset_strength,
                pitch_snr_db=event.pitch_snr_db,
                source_time_seconds=event.start_seconds,
            )
        )

    merged: list[_QuantizedPitchEvent] = []
    for midi, pitch_events in by_pitch.items():
        pitch_events.sort(key=lambda event: (event.start_tick, event.end_tick))
        for event in pitch_events:
            if merged_for_pitch := (
                merged[-1]
                if merged and merged[-1].midi == midi
                else None
            ):
                if (
                    event.start_tick
                    <= merged_for_pitch.end_tick + join_gap_ticks
                    and _quantized_events_share_attack(
                        merged_for_pitch, event, tick_seconds
                    )
                ):
                    merged[-1] = _QuantizedPitchEvent(
                        start_tick=merged_for_pitch.start_tick,
                        end_tick=max(merged_for_pitch.end_tick, event.end_tick),
                        midi=midi,
                        amplitude=max(
                            merged_for_pitch.amplitude, event.amplitude
                        ),
                        phase_tick=merged_for_pitch.phase_tick,
                        end_phase_tick=max(
                            merged_for_pitch.end_phase_tick
                            if merged_for_pitch.end_phase_tick is not None
                            else float(merged_for_pitch.end_tick),
                            event.end_phase_tick
                            if event.end_phase_tick is not None
                            else float(event.end_tick),
                        ),
                        source_role=(
                            None
                            if (
                                merged_for_pitch.source_role is None
                                or event.source_role is None
                            )
                            else (
                                event.source_role
                                if event.amplitude > merged_for_pitch.amplitude
                                else merged_for_pitch.source_role
                            )
                        ),
                        model_confidence=max(
                            merged_for_pitch.model_confidence
                            if merged_for_pitch.model_confidence is not None
                            else merged_for_pitch.amplitude,
                            event.model_confidence
                            if event.model_confidence is not None
                            else event.amplitude,
                        ),
                        pitch_loudness=(
                            max(
                                value
                                for value in (
                                    merged_for_pitch.pitch_loudness,
                                    event.pitch_loudness,
                                )
                                if value is not None
                            )
                            if (
                                merged_for_pitch.pitch_loudness is not None
                                or event.pitch_loudness is not None
                            )
                            else None
                        ),
                        onset_strength=merged_for_pitch.onset_strength,
                        pitch_snr_db=(
                            max(
                                value
                                for value in (
                                    merged_for_pitch.pitch_snr_db,
                                    event.pitch_snr_db,
                                )
                                if value is not None
                            )
                            if (
                                merged_for_pitch.pitch_snr_db is not None
                                or event.pitch_snr_db is not None
                            )
                            else None
                        ),
                        source_time_seconds=merged_for_pitch.source_time_seconds,
                    )
                    continue
            merged.append(event)

    return sorted(merged, key=lambda event: (event.start_tick, event.midi))


def _remove_overlapping_timed_duplicates(
    background: Iterable[_TimedPitchEvent],
    foreground: Iterable[_TimedPitchEvent],
) -> list[_TimedPitchEvent]:
    """Remove only near-identical lead leakage from an accompaniment stem.

    Pitch-class-only matching used to erase a guitar chord whenever its note
    name happened to match a sustained bass note, even in another octave.  That
    is especially destructive for syncopated funk arrangements.
    """

    foreground_by_midi: dict[int, list[_TimedPitchEvent]] = defaultdict(list)
    for event in foreground:
        foreground_by_midi[event.midi].append(event)

    retained: list[_TimedPitchEvent] = []
    for event in background:
        if event.source_role in BACKGROUND_ROLE_INSTRUMENTS:
            # This note was independently measured in an isolated instrument
            # stem.  A guitar or piano intentionally doubling the melody is
            # musical unison, not separator leakage.
            retained.append(event)
            continue
        conflicts = foreground_by_midi.get(event.midi, ())
        leaked_duplicate = any(
            _timed_events_are_duplicate(event, other)
            and event.amplitude <= other.amplitude * 1.20
            for other in conflicts
        )
        if not leaked_duplicate:
            retained.append(event)
    return retained


def _select_monophonic_events(
    events: Iterable[_QuantizedPitchEvent], *, prefer_low: bool
) -> list[_QuantizedPitchEvent]:
    """Find a globally coherent vocal or bass path through neural candidates.

    A greedy choice can jump to an octave harmonic for one event and then use
    that mistake as context for the next event.  Dynamic programming evaluates
    the complete phrase, rewarding supported notes while penalizing implausible
    leaps.  Long rests weaken the continuity term so a new phrase can start in
    another register naturally.
    """

    by_start: dict[int, list[_QuantizedPitchEvent]] = defaultdict(list)
    for event in events:
        by_start[event.start_tick].append(event)
    groups = [
        sorted(by_start[tick], key=lambda event: (event.midi, -event.amplitude))
        for tick in sorted(by_start)
    ]
    if not groups:
        return []

    scores: list[list[float]] = []
    parents: list[list[int]] = []
    for group_index, candidates in enumerate(groups):
        group_scores: list[float] = []
        group_parents: list[int] = []
        for candidate in candidates:
            duration = candidate.end_tick - candidate.start_tick
            local_score = candidate.amplitude + min(duration, 12) * 0.006
            if prefer_low:
                local_score += max(0, 60 - candidate.midi) * 0.0025

            if group_index == 0:
                group_scores.append(local_score)
                group_parents.append(-1)
                continue

            best_score = -math.inf
            best_parent = 0
            for parent_index, previous in enumerate(groups[group_index - 1]):
                gap = max(0, candidate.start_tick - previous.start_tick)
                continuity_weight = math.exp(-gap / 18.0)
                distance = abs(candidate.midi - previous.midi)
                transition = 0.0
                if distance == 0:
                    transition += 0.13 * continuity_weight
                else:
                    transition -= min(0.34, distance * 0.018) * continuity_weight
                    if distance >= 12:
                        transition -= 0.08 * continuity_weight
                if candidate.start_tick < previous.end_tick:
                    transition -= 0.04 * continuity_weight
                total = scores[-1][parent_index] + transition + local_score
                if total > best_score:
                    best_score = total
                    best_parent = parent_index
            group_scores.append(best_score)
            group_parents.append(best_parent)
        scores.append(group_scores)
        parents.append(group_parents)

    chosen_indices = [0] * len(groups)
    chosen_indices[-1] = max(
        range(len(groups[-1])), key=lambda index: scores[-1][index]
    )
    for group_index in range(len(groups) - 1, 0, -1):
        chosen_indices[group_index - 1] = parents[group_index][
            chosen_indices[group_index]
        ]

    selected: list[_QuantizedPitchEvent] = []
    for group, candidate_index in zip(groups, chosen_indices):
        candidate = group[candidate_index]
        if selected and candidate.start_tick < selected[-1].end_tick:
            active = selected[-1]
            if (
                candidate.midi != active.midi
                and candidate.amplitude < active.amplitude * 0.58
            ):
                continue
            selected[-1] = replace(
                active,
                end_tick=max(active.start_tick + 1, candidate.start_tick),
            )
        selected.append(candidate)
    return selected


def _select_accompaniment_focus_path(
    groups: Sequence[tuple[int, Sequence[_QuantizedPitchEvent]]],
) -> dict[int, _QuantizedPitchEvent]:
    """Track one persistent song-level focus while allowing support-only hits.

    Each MIDI pitch is a hidden state. A state can emit a candidate or carry
    silently across an accompaniment onset. Large register changes therefore
    need repeated evidence before they can replace the current focus; isolated
    chord tones and harmonics remain supporting notes instead of briefly taking
    over the lead layer.
    """

    if not groups:
        return {}

    all_events = [event for _tick, candidates in groups for event in candidates]
    pitch_weights: dict[int, float] = defaultdict(float)
    pitch_class_weights: dict[int, float] = defaultdict(float)
    for event in all_events:
        weight = event.amplitude * min(12, event.end_tick - event.start_tick + 1)
        pitch_weights[event.midi] += weight
        pitch_class_weights[event.midi % 12] += weight
    total_pitch_weight = sum(pitch_weights.values())
    focus_target = 60
    cumulative = 0.0
    for midi, weight in sorted(pitch_weights.items()):
        cumulative += weight
        if cumulative >= total_pitch_weight * 0.70:
            focus_target = midi
            break
    maximum_pitch_class_weight = max(pitch_class_weights.values(), default=1.0)

    def emission_score(
        event: _QuantizedPitchEvent,
        candidates: Sequence[_QuantizedPitchEvent],
    ) -> float:
        low_midi = min(candidate.midi for candidate in candidates)
        high_midi = max(candidate.midi for candidate in candidates)
        span = max(1, high_midi - low_midi)
        register_rank = (event.midi - low_midi) / span
        target_support = max(0.0, 1.0 - abs(event.midi - focus_target) / 24.0)
        pitch_class_support = (
            pitch_class_weights[event.midi % 12] / maximum_pitch_class_weight
        )
        duration_support = min(1.0, (event.end_tick - event.start_tick) / 8.0)
        return (
            0.07
            + 0.30 * event.amplitude
            + 0.07 * register_rank
            + 0.04 * target_support
            + 0.03 * pitch_class_support
            + 0.03 * duration_support
        )

    def transition_score(previous_midi: int, midi: int, gap: int) -> float:
        distance = abs(midi - previous_midi)
        if distance == 0:
            score = 0.16
        elif distance <= 2:
            score = 0.11
        elif distance <= 4:
            score = 0.04
        elif distance == 5:
            score = -0.10
        elif distance <= 7:
            score = -0.50
        elif distance <= 11:
            score = -0.72
        else:
            score = -1.08 - min(0.30, (distance - 12) * 0.015)
        # A genuine new phrase after a long rest may begin in another register.
        return score * math.exp(-max(0, gap - 1) / 32.0)

    candidate_maps = [
        {event.midi: event for event in candidates}
        for _tick, candidates in groups
    ]
    first_candidates = candidate_maps[0]
    scores = {
        midi: emission_score(event, groups[0][1])
        for midi, event in first_candidates.items()
    }
    history: list[dict[int, tuple[int | None, bool]]] = [
        {midi: (None, True) for midi in scores}
    ]
    carry_penalty = 0.055

    for group_index in range(1, len(groups)):
        tick, candidates = groups[group_index]
        previous_tick = groups[group_index - 1][0]
        gap = max(1, tick - previous_tick)
        next_scores = {
            midi: score - carry_penalty for midi, score in scores.items()
        }
        pointers: dict[int, tuple[int | None, bool]] = {
            midi: (midi, False) for midi in scores
        }
        for midi, event in candidate_maps[group_index].items():
            emission = emission_score(event, candidates)
            previous_midi, best_score = max(
                (
                    (state_midi, state_score + transition_score(state_midi, midi, gap))
                    for state_midi, state_score in scores.items()
                ),
                key=lambda item: item[1],
            )
            emitted_score = best_score + emission
            if emitted_score > next_scores.get(midi, -math.inf):
                next_scores[midi] = emitted_score
                pointers[midi] = (previous_midi, True)
        scores = next_scores
        history.append(pointers)

    state: int | None = max(scores, key=scores.get)
    selected: dict[int, _QuantizedPitchEvent] = {}
    for group_index in range(len(groups) - 1, -1, -1):
        if state is None:
            break
        previous_state, emitted = history[group_index][state]
        if emitted:
            event = candidate_maps[group_index].get(state)
            if event is not None:
                selected[groups[group_index][0]] = event
        state = previous_state
    return selected


def _arrange_polyphonic_events(
    events: Iterable[_QuantizedPitchEvent],
    max_voices: int,
) -> list[_VoicedPitchEvent]:
    """Track stable core roles and reserve lanes for proven instruments."""

    if max_voices <= 0:
        return []
    by_start: dict[int, dict[int, _QuantizedPitchEvent]] = defaultdict(dict)
    for event in events:
        previous = by_start[event.start_tick].get(event.midi)
        event_priority = (
            event.source_role is None,
            event.amplitude,
            event.end_tick - event.start_tick,
        )
        previous_priority = (
            previous.source_role is None,
            previous.amplitude,
            previous.end_tick - previous.start_tick,
        ) if previous is not None else None
        if previous is None or event_priority > previous_priority:
            by_start[event.start_tick][event.midi] = event

    prepared_groups: list[tuple[int, list[_QuantizedPitchEvent]]] = []
    for start_tick in sorted(by_start):
        candidates = list(by_start[start_tick].values())
        # Every incoming candidate has already passed pitch-resolved source
        # validation.  A floor relative to the loudest chord tone used to erase
        # legitimate quiet inner voices whenever the melody was strongly
        # accented, so arrangement now applies only a conservative absolute
        # floor.
        confidence_floor = 0.27
        candidates = [
            event
            for event in candidates
            if (
                event.amplitude >= confidence_floor
                if event.source_role is None
                else event.amplitude >= 0.30
            )
        ]
        if candidates:
            prepared_groups.append((start_tick, candidates))

    # An auxiliary guitar or piano pass may reveal a masked background attack,
    # but it must never steal the song-level focus from the authoritative mixed
    # accompaniment transcription.
    core_groups = [
        (
            start_tick,
            [event for event in candidates if event.source_role is None],
        )
        for start_tick, candidates in prepared_groups
    ]
    focus_by_tick = _select_accompaniment_focus_path(
        [(tick, candidates) for tick, candidates in core_groups if candidates]
    )

    role_weights: dict[str, float] = defaultdict(float)
    role_demands: dict[str, int] = defaultdict(int)
    for _start_tick, candidates in prepared_groups:
        group_role_counts: dict[str, int] = defaultdict(int)
        for event in candidates:
            if event.source_role in BACKGROUND_ROLE_INSTRUMENTS:
                role_weights[event.source_role] += (
                    event.amplitude * min(12, event.end_tick - event.start_tick + 1)
                )
                group_role_counts[event.source_role] += 1
        for role, count in group_role_counts.items():
            role_demands[role] = max(role_demands[role], count)

    ranked_background_roles = [
        role
        for role, _weight in sorted(
            role_weights.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    maximum_background_voices = min(
        sum(role_demands.values()),
        max(0, max_voices - 3),
        max(len(ranked_background_roles), max_voices // 3),
    )
    background_voice_counts = {
        role: 0 for role in ranked_background_roles
    }
    # Give every audible role one lane first, then distribute remaining lanes
    # according to measured simultaneous demand.  This preserves a quiet piano
    # chord instead of reducing the entire isolated piano stem to one note.
    for _slot in range(maximum_background_voices):
        eligible = [
            role
            for role in ranked_background_roles
            if background_voice_counts[role] < role_demands[role]
        ]
        if not eligible:
            break
        role = max(
            eligible,
            key=lambda candidate: (
                background_voice_counts[candidate] == 0,
                role_weights[candidate]
                / (background_voice_counts[candidate] + 1),
                role_demands[candidate] - background_voice_counts[candidate],
            ),
        )
        background_voice_counts[role] += 1

    background_voice_count = sum(background_voice_counts.values())
    core_voice_limit = max(1, max_voices - background_voice_count)
    background_voices_by_role: dict[str, list[int]] = {}
    next_background_voice = core_voice_limit
    for role in ranked_background_roles:
        count = background_voice_counts[role]
        background_voices_by_role[role] = list(
            range(next_background_voice, next_background_voice + count)
        )
        next_background_voice += count

    result: list[_VoicedPitchEvent] = []
    previous_voice_pitch: dict[int, int] = {}
    consonant_intervals = {0, 3, 4, 5, 7, 8, 9}

    for start_tick, candidates in prepared_groups:
        lead = focus_by_tick.get(start_tick)
        core_candidates = [
            event for event in candidates if event.source_role is None
        ]
        selected = [lead] if lead is not None else []
        anchor: _QuantizedPitchEvent | None = None

        if core_voice_limit > 1:
            remaining = [
                event
                for event in core_candidates
                if event is not lead
            ]
            if remaining:
                supported_anchors = [
                    event
                    for event in remaining
                    if event.amplitude >= 0.30
                ]
                if supported_anchors:
                    anchor = min(
                        supported_anchors,
                        key=lambda event: (event.midi, -event.amplitude),
                    )
                    selected.append(anchor)

        remaining = [
            event
            for event in core_candidates
            if event not in selected
        ]
        selection_limit = core_voice_limit
        while remaining and len(selected) < selection_limit:
            def fill_score(event: _QuantizedPitchEvent) -> float:
                consonance = sum(
                    1
                    for chosen in selected
                    if abs(chosen.midi - event.midi) % 12 in consonant_intervals
                )
                return event.amplitude + 0.025 * consonance

            fill = max(remaining, key=fill_score)
            if fill.amplitude < 0.30:
                break
            selected.append(fill)
            remaining = [
                event
                for event in remaining
                if event is not fill
            ]

        assignments: dict[int, _QuantizedPitchEvent] = {}
        if lead is not None:
            assignments[0] = lead
        if anchor is not None:
            assignments[1] = anchor
        free_voices = [
            voice
            for voice in range(core_voice_limit)
            if voice not in assignments
        ]
        for event in sorted(
            (
                item
                for item in selected
                if item is not lead and item is not anchor
            ),
            key=lambda item: item.midi,
        ):
            if not free_voices:
                break
            voice = min(
                free_voices,
                key=lambda candidate_voice: (
                    abs(
                        event.midi
                        - previous_voice_pitch.get(candidate_voice, event.midi)
                    ),
                    candidate_voice,
                ),
            )
            assignments[voice] = event
            free_voices.remove(voice)

        for role in ranked_background_roles:
            role_voices = background_voices_by_role[role]
            if not role_voices:
                continue
            role_candidates = [
                event
                for event in candidates
                if event.source_role == role
            ]
            if not role_candidates:
                continue

            def background_score(
                event: _QuantizedPitchEvent, voice: int
            ) -> float:
                previous_pitch = previous_voice_pitch.get(voice)
                continuity = (
                    max(0.0, 1.0 - abs(event.midi - previous_pitch) / 12.0)
                    if previous_pitch is not None
                    else 0.5
                )
                duration_support = min(
                    1.0, (event.end_tick - event.start_tick) / 8.0
                )
                return event.amplitude + 0.06 * continuity + 0.02 * duration_support

            available_voices = list(role_voices)
            remaining_role_candidates = list(role_candidates)
            while available_voices and remaining_role_candidates:
                voice, background = max(
                    (
                        (voice, event)
                        for voice in available_voices
                        for event in remaining_role_candidates
                    ),
                    key=lambda item: background_score(item[1], item[0]),
                )
                assignments[voice] = background
                available_voices.remove(voice)
                remaining_role_candidates.remove(background)

        # A reserved core lane that is silent at this onset can carry an
        # additional independently supported background chord tone.  This is
        # especially important when the combined accompaniment pass misses an
        # entire quiet piano attack; leaving those lanes empty would re-create
        # a fixed polyphony cap despite having verified candidates available.
        assigned_event_ids = {id(event) for event in assignments.values()}
        overflow_candidates = sorted(
            (
                event
                for event in candidates
                if event.source_role in BACKGROUND_ROLE_INSTRUMENTS
                and id(event) not in assigned_event_ids
            ),
            key=lambda event: (event.amplitude, -event.midi),
            reverse=True,
        )
        overflow_voices = [
            voice
            for voice in range(core_voice_limit)
            if voice not in assignments
        ]
        for event in overflow_candidates:
            if not overflow_voices:
                break
            voice = min(
                overflow_voices,
                key=lambda candidate_voice: (
                    abs(
                        event.midi
                        - previous_voice_pitch.get(
                            candidate_voice, event.midi
                        )
                    ),
                    candidate_voice,
                ),
            )
            assignments[voice] = event
            overflow_voices.remove(voice)

        for voice, event in sorted(assignments.items()):
            result.append(
                _VoicedPitchEvent(
                    start_tick=event.start_tick,
                    end_tick=event.end_tick,
                    midi=event.midi,
                    amplitude=event.amplitude,
                    voice=voice,
                    phase_tick=event.phase_tick,
                    end_phase_tick=event.end_phase_tick,
                    source_role=event.source_role,
                    model_confidence=event.model_confidence,
                    pitch_loudness=event.pitch_loudness,
                    onset_strength=event.onset_strength,
                    pitch_snr_db=event.pitch_snr_db,
                    source_time_seconds=event.source_time_seconds,
                )
            )
            previous_voice_pitch[voice] = event.midi
    return result


def _stable_voice_instruments(
    events: Iterable[_VoicedPitchEvent],
    requested_instrument: int | None,
    default_instrument: int | None,
) -> dict[int, int]:
    """Choose one timbre per voice from its weighted whole-song register."""

    by_voice: dict[int, list[_VoicedPitchEvent]] = defaultdict(list)
    for event in events:
        by_voice[event.voice].append(event)
    result: dict[int, int] = {}
    for voice, voice_events in by_voice.items():
        if requested_instrument is not None:
            result[voice] = requested_instrument
            continue
        if default_instrument is not None:
            result[voice] = default_instrument
            continue
        source_weights: dict[str, float] = defaultdict(float)
        total_weight = 0.0
        for event in voice_events:
            weight = max(0.01, event.amplitude) * max(
                1, event.end_tick - event.start_tick
            )
            total_weight += weight
            if event.source_role in BACKGROUND_ROLE_INSTRUMENTS:
                source_weights[event.source_role] += weight
        if source_weights:
            source_role, source_weight = max(
                source_weights.items(), key=lambda item: item[1]
            )
            if source_weight >= total_weight * 0.45:
                result[voice] = BACKGROUND_ROLE_INSTRUMENTS[source_role]
                continue
        weighted_pitches = sorted(
            (
                event.midi,
                max(0.01, event.amplitude)
                * max(1, event.end_tick - event.start_tick),
            )
            for event in voice_events
        )
        midpoint = sum(weight for _midi, weight in weighted_pitches) * 0.5
        cumulative = 0.0
        representative_midi = weighted_pitches[-1][0]
        for midi, weight in weighted_pitches:
            cumulative += weight
            if cumulative >= midpoint:
                representative_midi = midi
                break
        if voice == 0:
            if representative_midi < 60:
                result[voice] = INSTRUMENTS["guitar"]
            elif representative_midi < 84:
                result[voice] = INSTRUMENTS["piano"]
            else:
                result[voice] = INSTRUMENTS["flute"]
        elif voice == 1:
            result[voice] = (
                INSTRUMENTS["guitar"]
                if representative_midi < 64
                else INSTRUMENTS["piano"]
            )
        else:
            result[voice] = INSTRUMENTS["piano"]
    return result


def _ai_events_to_nbs(
    events: Iterable[_TimedPitchEvent],
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    panning_by_tick,
    config: ConversionConfig,
    *,
    layer_offset: int,
    max_notes: int,
    default_instrument: int | None,
    velocity_scale: float,
    monophonic: bool = False,
    prefer_low: bool = False,
) -> list[NbsNote]:
    """Convert duration-aware AI events into sparse, playable NBS notes."""

    quantized = _quantize_timed_pitch_events(
        events,
        timeline_origin_seconds,
        tick_seconds,
        tick_count,
        join_gap_ticks=1 if monophonic else -1,
    )
    if monophonic:
        quantized = _select_monophonic_events(quantized, prefer_low=prefer_low)
        arranged = [
            _VoicedPitchEvent(
                start_tick=event.start_tick,
                end_tick=event.end_tick,
                midi=event.midi,
                amplitude=event.amplitude,
                voice=0,
                phase_tick=event.phase_tick,
                end_phase_tick=event.end_phase_tick,
                source_role=event.source_role,
                model_confidence=event.model_confidence,
                pitch_loudness=event.pitch_loudness,
                onset_strength=event.onset_strength,
                pitch_snr_db=event.pitch_snr_db,
                source_time_seconds=event.source_time_seconds,
            )
            for event in quantized
        ]
    else:
        arranged = _arrange_polyphonic_events(quantized, max_notes)
    voice_instruments = _stable_voice_instruments(
        arranged, config.instrument, default_instrument
    )

    scheduled: dict[
        int, list[tuple[_VoicedPitchEvent, float, bool, float]]
    ] = defaultdict(list)
    for event in arranged:
        onset_phase = (
            event.phase_tick
            if event.phase_tick is not None
            else float(event.start_tick)
        )
        scheduled[event.start_tick].append(
            (event, event.amplitude, True, onset_phase)
        )
        for retrigger_tick, retrigger_phase in _iter_phase_locked_retrigger_positions(
            event.start_tick,
            event.end_tick,
            config,
            phase_tick=event.phase_tick,
            end_phase_tick=event.end_phase_tick,
        ):
            scheduled[retrigger_tick].append(
                (event, event.amplitude * 0.82, False, retrigger_phase)
            )

    notes: list[NbsNote] = []
    for tick in sorted(scheduled):
        # Prefer true onsets, then confidence.  Deduplicate only sounds that
        # actually collapse to the same output instrument and key; octave
        # doublings that remain distinct in NBS or Minecraft are musical data.
        candidates = sorted(
            scheduled[tick],
            key=lambda item: (item[2], item[1], item[0].end_tick - item[0].start_tick),
            reverse=True,
        )
        retained: list[tuple[_VoicedPitchEvent, float, bool, float]] = []
        occupied_voices: set[int] = set()
        occupied_sounds: set[tuple[int, int]] = set()
        for candidate in candidates:
            if candidate[0].voice in occupied_voices:
                continue
            raw_key = candidate[0].midi - NBS_LOWEST_MIDI
            output_key = (
                fold_key_to_minecraft_range(raw_key)
                if config.minecraft_range
                else raw_key
            )
            sound_identity = (
                voice_instruments[candidate[0].voice],
                output_key,
            )
            if sound_identity in occupied_sounds:
                continue
            retained.append(candidate)
            occupied_voices.add(candidate[0].voice)
            occupied_sounds.add(sound_identity)
            if len(retained) >= max_notes:
                break
        retained.sort(key=lambda item: item[0].voice)

        for event, amplitude, is_onset, canonical_phase in retained:
            raw_key = event.midi - NBS_LOWEST_MIDI
            key = (
                fold_key_to_minecraft_range(raw_key)
                if config.minecraft_range
                else raw_key
            )
            instrument = voice_instruments[event.voice]
            source_level = (
                event.pitch_loudness
                if event.pitch_loudness is not None
                else _clamp((amplitude - 0.25) / 0.65, 0.0, 1.0)
            )
            model_confidence = (
                event.model_confidence
                if event.model_confidence is not None
                else amplitude
            )
            velocity = (
                (8.0 + 92.0 * (source_level ** 0.70))
                * (0.70 + 0.30 * math.sqrt(_clamp(model_confidence, 0.0, 1.0)))
                * velocity_scale
            )
            if not is_onset:
                # NBS has no note duration.  A quiet continuation keeps a long
                # instrument audible without pretending that the source made
                # another full-strength attack.
                velocity *= 0.55
            notes.append(
                NbsNote(
                    tick=tick,
                    layer=layer_offset + event.voice,
                    instrument=instrument,
                    key=key,
                    velocity=int(_clamp(round(velocity), 1, 100)),
                    panning=int(panning_by_tick[tick]),
                    continuation=not is_onset,
                    source_role=event.source_role,
                    source_midi=event.midi,
                    source_time_seconds=(
                        event.source_time_seconds
                        if is_onset and event.source_time_seconds is not None
                        else timeline_origin_seconds
                        + canonical_phase * tick_seconds
                    ),
                    source_loudness=(source_level if is_onset else None),
                    source_snr_db=event.pitch_snr_db,
                    source_onset_strength=event.onset_strength,
                )
            )
    return notes


def _source_cache_key(
    input_path: Path,
    *,
    model: str = DEMUCS_MODEL,
    shifts: int = DEMUCS_SHIFTS,
    overlap: float = DEMUCS_OVERLAP,
    cache_version: int = SEPARATION_CACHE_VERSION,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        (
            f"demucs:{model}:shifts={shifts}:"
            f"overlap={overlap}:v{cache_version}\0"
        ).encode("ascii")
    )
    with input_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def _find_demucs_stems(
    cache_entry: Path,
    expected_stems: Sequence[str] = DEMUCS_STEMS,
) -> dict[str, Path] | None:
    result: dict[str, Path] = {}
    for stem in expected_stems:
        matches = sorted(cache_entry.rglob(f"{stem}.wav"))
        if not matches:
            return None
        result[stem] = matches[0]
    return result


def _separate_audio_stems(
    input_path: Path,
    report: Callable[[str], None],
    progress_update: Callable[[float, str | None], None] | None = None,
    *,
    progress_start: float = 0.03,
    progress_end: float = 0.72,
    model: str = DEMUCS_MODEL,
    model_count: int = DEMUCS_MODEL_COUNT,
    shifts: int = DEMUCS_SHIFTS,
    overlap: float = DEMUCS_OVERLAP,
    expected_stems: Sequence[str] = DEMUCS_STEMS,
    cache_version: int = SEPARATION_CACHE_VERSION,
    description: str = "vocals, bass, drums, and accompaniment",
) -> dict[str, Path]:
    """Separate an input mix with Demucs and reuse a content-addressed cache."""

    if importlib.util.find_spec("demucs") is None:
        raise ConversionError(
            "Demucs is required for AI source separation. Run "
            "`python -m pip install -r requirements.txt`."
        )

    cache_root = Path(__file__).resolve().parent / ".stem_cache"
    cache_entry = cache_root / _source_cache_key(
        input_path,
        model=model,
        shifts=shifts,
        overlap=overlap,
        cache_version=cache_version,
    )
    cached = (
        _find_demucs_stems(cache_entry, expected_stems)
        if cache_entry.is_dir()
        else None
    )
    if cached is not None:
        report("Reusing cached AI separation results...")
        if progress_update is not None:
            progress_update(progress_end, "AI separation complete (cached)")
        return cached

    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_entry.exists():
        resolved_root = cache_root.resolve()
        resolved_entry = cache_entry.resolve()
        if resolved_entry.parent != resolved_root:
            raise ConversionError("Could not validate the source-separation cache path.")
        shutil.rmtree(resolved_entry)
    cache_entry.mkdir(parents=True)

    try:
        import torch

        demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = (
            torch.cuda.get_device_name(0) if demucs_device == "cuda" else "CPU"
        )
    except (ImportError, RuntimeError):
        demucs_device = "cpu"
        device_name = "CPU"

    report(f"Separating {description} with AI...")
    report(
        f"Demucs {model} / {shifts} shifts / "
        f"device: {device_name}"
    )
    report("The first run may take several minutes while models are downloaded.")
    if progress_update is not None:
        progress_update(progress_start, "Preparing the AI separation model...")
    command = [
        sys.executable,
        "-m",
        "demucs",
        "--name",
        model,
        "--out",
        str(cache_entry),
        "--filename",
        "{stem}.{ext}",
        "--shifts",
        str(shifts),
        "--overlap",
        str(overlap),
        "--clip-mode",
        "clamp",
        "--jobs",
        "1",
        "--device",
        demucs_device,
        str(input_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    environment["PYTHONUTF8"] = "1"
    expected_passes = model_count * shifts
    parser = _DemucsProgressParser(expected_passes)
    output_tail: deque[str] = deque(maxlen=40)
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise OSError("Demucs output pipe could not be opened")

        separating_started = False
        last_prepare_update = 0.0
        for raw_line in process.stdout:
            clean_line = raw_line.strip()
            if clean_line:
                output_tail.append(clean_line)
            if "Separating track" in clean_line:
                separating_started = True
            fraction = parser.feed(clean_line) if separating_started else None
            if fraction is not None and progress_update is not None:
                overall = progress_start + (
                    progress_end - progress_start
                ) * fraction
                progress_update(
                    overall,
                    f"AI separation {parser.displayed_pass}/{expected_passes}",
                )
            elif (
                not separating_started
                and progress_update is not None
                and time.monotonic() - last_prepare_update >= 1.0
            ):
                progress_update(progress_start, "Preparing the AI separation model...")
                last_prepare_update = time.monotonic()
        return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    except OSError as exc:
        shutil.rmtree(cache_entry, ignore_errors=True)
        raise ConversionError("Could not start Demucs.") from exc
    if return_code != 0:
        shutil.rmtree(cache_entry, ignore_errors=True)
        detail = next(
            (
                line
                for line in reversed(output_tail)
                if any(word in line.lower() for word in ("error", "cuda", "memory"))
            ),
            "",
        )
        detail_suffix = f" Details: {detail}" if detail else ""
        raise ConversionError(
            "AI source separation failed. Check available disk space and the network connection."
            + detail_suffix
        )

    stems = _find_demucs_stems(cache_entry, expected_stems)
    if stems is None:
        shutil.rmtree(cache_entry, ignore_errors=True)
        raise ConversionError("Could not read the separated Demucs stems.")
    if progress_update is not None:
        progress_update(progress_end, "AI source separation complete")
    return stems


def _as_scalar_tempo(value, fallback: float = 120.0) -> float:
    try:
        size = int(value.size)
        tempo = float(value.reshape(-1)[0]) if size else fallback
    except AttributeError:
        tempo = float(value)
    if not math.isfinite(tempo) or tempo <= 0:
        return fallback
    return tempo


def _refine_tempo_from_beat_times(
    estimated_bpm: float,
    beat_times,
    np,
) -> float:
    """Fit one stable tempo to the complete tracked beat sequence.

    Librosa's scalar tempo candidates are quantized by the analysis hop.  A
    135 BPM song can therefore be reported as 136 BPM even when hundreds of
    tracked beats clearly establish the longer average interval.  Regressing
    absolute beat time against beat number averages away that frame rounding
    and prevents sustained-note repeats from drifting across the song.
    """

    if not math.isfinite(estimated_bpm) or estimated_bpm <= 0.0:
        return estimated_bpm
    times = np.asarray(beat_times, dtype=np.float64).reshape(-1)
    times = times[np.isfinite(times)]
    if times.size < 8:
        return estimated_bpm
    times = np.unique(times)
    if times.size < 8:
        return estimated_bpm

    nominal_interval = 60.0 / estimated_bpm
    deltas = np.diff(times)
    if (
        float(times[-1] - times[0]) < nominal_interval * 7.0
        or np.any(deltas <= 0.0)
    ):
        return estimated_bpm

    # Preserve the beat count across an occasional missed tracker event.  The
    # initial scalar estimate is sufficiently accurate for this integer-only
    # decision even when it is not accurate enough for long-term scheduling.
    beat_steps = np.maximum(
        1,
        np.floor(deltas / nominal_interval + 0.5).astype(np.int64),
    )
    beat_numbers = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(beat_steps, dtype=np.float64))
    )
    try:
        interval, intercept = np.polyfit(beat_numbers, times, 1)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return estimated_bpm
    if not math.isfinite(float(interval)) or interval <= 0.0:
        return estimated_bpm

    refined_bpm = 60.0 / float(interval)
    if not 0.85 <= refined_bpm / estimated_bpm <= 1.15:
        return estimated_bpm

    fitted = intercept + interval * beat_numbers
    residuals = times - fitted
    residual_p95 = float(np.quantile(np.abs(residuals), 0.95))
    if residual_p95 > max(0.20, nominal_interval * 0.60):
        return estimated_bpm
    return refined_bpm


def _normalize_onset_envelope(envelope, np):
    """Scale an onset envelope without letting one extreme hit dominate it."""

    values = np.asarray(envelope, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    values = np.maximum(np.nan_to_num(values, copy=False), 0.0)
    positive = values[values > 0.0]
    if positive.size == 0:
        return np.zeros_like(values)
    reference = max(float(np.quantile(positive, 0.95)), 1e-9)
    return np.clip(values / reference, 0.0, 2.0).astype(np.float32, copy=False)


def _tempo_pulse_score(
    onset_envelope,
    bpm: float,
    sample_rate: int,
    hop_length: int,
    np,
) -> float:
    """Measure normalized onset-envelope agreement at one tempo period."""

    if not math.isfinite(bpm) or bpm <= 0.0 or sample_rate <= 0 or hop_length <= 0:
        return 0.0
    values = _normalize_onset_envelope(onset_envelope, np).astype(
        np.float64, copy=False
    )
    if values.size < 4 or not np.any(values > 0.0):
        return 0.0
    values = np.maximum(0.0, values - float(np.quantile(values, 0.35)))
    exact_lag = sample_rate * 60.0 / (hop_length * bpm)
    if not 1.0 <= exact_lag < values.size - 1:
        return 0.0

    def score_at_lag(lag: int) -> float:
        lag = max(1, min(int(lag), values.size - 1))
        first = values[:-lag]
        second = values[lag:]
        denominator = math.sqrt(
            float(np.dot(first, first)) * float(np.dot(second, second))
        )
        return float(np.dot(first, second)) / max(denominator, 1e-12)

    lower = math.floor(exact_lag)
    upper = math.ceil(exact_lag)
    if lower == upper:
        return score_at_lag(lower)
    fraction = exact_lag - lower
    return (1.0 - fraction) * score_at_lag(lower) + fraction * score_at_lag(upper)


def _resolve_tempo_octave(
    estimated_bpm: float,
    onset_envelope,
    sample_rate: int,
    hop_length: int,
    np,
) -> float:
    """Resolve only clear half/double-tempo estimates to a useful beat unit.

    Beat period and metrical level are intrinsically ambiguous.  We therefore
    leave the broad 75-210 BPM range untouched and only canonicalize an extreme
    estimate when the neighboring octave has comparable pulse support.  This
    fixes the common 67.5-vs-135 BPM failure without arbitrarily doubling a
    normal 90 BPM song.
    """

    if not math.isfinite(estimated_bpm) or estimated_bpm <= 0.0:
        return estimated_bpm
    alternate: float | None = None
    if estimated_bpm < 75.0 and estimated_bpm * 2.0 <= 210.0:
        alternate = estimated_bpm * 2.0
    elif estimated_bpm > 210.0 and estimated_bpm * 0.5 >= 75.0:
        alternate = estimated_bpm * 0.5
    if alternate is None:
        return estimated_bpm

    original_score = _tempo_pulse_score(
        onset_envelope, estimated_bpm, sample_rate, hop_length, np
    )
    alternate_score = _tempo_pulse_score(
        onset_envelope, alternate, sample_rate, hop_length, np
    )
    if alternate_score + 0.02 >= original_score * 0.82:
        return alternate
    return estimated_bpm


def _estimate_tempo_and_beats(
    percussive,
    mono,
    sample_rate: int,
    hop_length: int,
    librosa,
    np,
) -> tuple[float, float, object]:
    """Track tempo from a drum/full-mix consensus at high time resolution."""

    analysis_hop = max(64, min(256, int(hop_length)))
    envelopes = []
    for audio in (percussive, mono):
        try:
            envelope = librosa.onset.onset_strength(
                y=np.asarray(audio, dtype=np.float32),
                sr=sample_rate,
                hop_length=analysis_hop,
                aggregate=np.median,
            )
        except (ValueError, RuntimeError):
            continue
        normalized = _normalize_onset_envelope(envelope, np)
        if normalized.size and np.any(normalized > 0.0):
            envelopes.append(normalized)
    if not envelopes:
        return 120.0, 120.0, np.empty(0, dtype=np.float64)

    common_length = min(len(envelope) for envelope in envelopes)
    if len(envelopes) == 1:
        consensus = envelopes[0][:common_length]
    else:
        # Percussion supplies the stable pulse; the complete mix restores beats
        # during drum dropouts and intros without overwhelming it.
        consensus = (
            0.68 * envelopes[0][:common_length]
            + 0.32 * envelopes[1][:common_length]
        ).astype(np.float32, copy=False)

    estimated_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=consensus,
        sr=sample_rate,
        hop_length=analysis_hop,
        trim=False,
    )
    raw_bpm = _as_scalar_tempo(estimated_tempo)
    resolved_bpm = _resolve_tempo_octave(
        raw_bpm, consensus, sample_rate, analysis_hop, np
    )
    if abs(resolved_bpm - raw_bpm) >= 0.05:
        try:
            _, beat_frames = librosa.beat.beat_track(
                onset_envelope=consensus,
                sr=sample_rate,
                hop_length=analysis_hop,
                bpm=resolved_bpm,
                trim=False,
            )
        except (TypeError, ValueError, RuntimeError):
            pass
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sample_rate,
        hop_length=analysis_hop,
    )
    refined_bpm = _refine_tempo_from_beat_times(
        resolved_bpm, beat_times, np
    )
    return raw_bpm, refined_bpm, beat_times


def _aggregate_by_tick(
    matrix,
    frame_ticks,
    tick_count: int,
    np,
    *,
    frame_times=None,
):
    aggregate = np.zeros((matrix.shape[0], tick_count), dtype=np.float32)
    source_times = (
        np.full((matrix.shape[0], tick_count), np.nan, dtype=np.float64)
        if frame_times is not None
        else None
    )
    valid_indices = np.flatnonzero(
        (frame_ticks >= 0) & (frame_ticks < tick_count)
    )
    for frame_index in valid_indices:
        tick = int(frame_ticks[frame_index])
        if source_times is not None:
            stronger = matrix[:, frame_index] > aggregate[:, tick]
            source_times[stronger, tick] = float(frame_times[frame_index])
        np.maximum(
            aggregate[:, tick], matrix[:, frame_index], out=aggregate[:, tick]
        )
    return (aggregate, source_times) if source_times is not None else aggregate


def _pitch_candidates(
    column,
    sensitivity: float,
    max_notes: int,
    np,
    *,
    unique_pitch_classes: bool = False,
) -> list[int]:
    """Return prominent CQT-bin indices, with common harmonics de-emphasized."""

    peak_db = float(np.max(column))
    global_floor = -24.0 - (36.0 * sensitivity)
    if peak_db < global_floor:
        return []
    threshold = max(global_floor, peak_db - 24.0)

    peaks = []
    for pitch in range(len(column)):
        value = float(column[pitch])
        if value < threshold:
            continue
        left = float(column[pitch - 1]) if pitch > 0 else -math.inf
        right = (
            float(column[pitch + 1]) if pitch + 1 < len(column) else -math.inf
        )
        if value >= left and value > right:
            peaks.append(pitch)

    scores: list[tuple[float, float, int]] = []
    for pitch in peaks:
        score = float(column[pitch])
        # Octaves and the 3rd/4th harmonics are frequent false fundamentals in CQT.
        for interval, penalty in ((12, 8.0), (19, 6.0), (24, 7.0), (28, 4.0)):
            lower = pitch - interval
            if lower < 0 or float(column[lower]) < threshold:
                continue
            difference = float(column[pitch] - column[lower])
            if difference <= 12.0:
                score -= penalty * (1.0 - max(0.0, difference) / 12.0)
        scores.append((score, float(column[pitch]), pitch))

    scores.sort(reverse=True)
    selected: list[int] = []
    selected_pitch_classes: set[int] = set()
    for _, _, pitch in scores:
        pitch_class = pitch % 12
        if unique_pitch_classes and pitch_class in selected_pitch_classes:
            continue
        selected.append(pitch)
        selected_pitch_classes.add(pitch_class)
        if len(selected) >= max_notes:
            break
    return selected


def _extract_pitch_events(
    tick_db,
    config: ConversionConfig,
    panning_by_tick,
    np,
    *,
    unique_pitch_classes: bool = False,
    source_times=None,
    timeline_origin_seconds: float = 0.0,
    tick_seconds: float | None = None,
) -> list[_PitchEvent]:
    events: list[_PitchEvent] = []
    last_selected = np.full(tick_db.shape[0], -10_000, dtype=np.int32)
    retrigger_interval = _retrigger_interval_ticks_exact(config)
    next_retrigger = np.full(tick_db.shape[0], math.inf, dtype=np.float64)
    global_floor = -24.0 - (36.0 * config.sensitivity)
    onset_delta = 6.0 - (3.0 * config.sensitivity)

    for tick in range(tick_db.shape[1]):
        column = tick_db[:, tick]
        selected = _pitch_candidates(
            column,
            config.sensitivity,
            config.max_chord_notes,
            np,
            unique_pitch_classes=unique_pitch_classes,
        )
        previous = tick_db[:, tick - 1] if tick > 0 else None

        for pitch in selected:
            newly_present = tick - int(last_selected[pitch]) > 1
            strengthened = previous is None or (
                float(column[pitch] - previous[pitch]) >= onset_delta
            )
            periodic = (
                retrigger_interval > 0.0
                and math.isfinite(float(next_retrigger[pitch]))
                and tick >= _round_tick(float(next_retrigger[pitch]))
            )
            if newly_present or strengthened or periodic:
                normalized = (float(column[pitch]) - global_floor) / -global_floor
                velocity = round(30.0 + 70.0 * math.sqrt(_clamp(normalized, 0.0, 1.0)))
                events.append(
                    _PitchEvent(
                        tick=tick,
                        midi=NBS_LOWEST_MIDI + pitch,
                        velocity=int(_clamp(velocity, 1, 100)),
                        panning=int(panning_by_tick[tick]),
                        strength_db=float(column[pitch]),
                        continuation=(
                            periodic and not newly_present and not strengthened
                        ),
                        source_time_seconds=(
                            float(source_times[pitch, tick])
                            if (
                                source_times is not None
                                and math.isfinite(
                                    float(source_times[pitch, tick])
                                )
                                and not (
                                    periodic
                                    and not newly_present
                                    and not strengthened
                                )
                            )
                            else (
                                timeline_origin_seconds + tick * tick_seconds
                                if tick_seconds is not None
                                else None
                            )
                        ),
                    )
                )
                if retrigger_interval > 0.0:
                    if newly_present or strengthened:
                        next_retrigger[pitch] = tick + retrigger_interval
                    else:
                        next_retrigger[pitch] = _advance_retrigger_deadline(
                            float(next_retrigger[pitch]),
                            tick,
                            retrigger_interval,
                        )
            last_selected[pitch] = tick

    return events


def _calculate_tick_panning(
    channels,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    np,
):
    if channels.ndim < 2 or channels.shape[0] < 2:
        return np.zeros(tick_count, dtype=np.int16)

    left = channels[0]
    right = channels[1]
    half_window = max(256, _round_tick(sample_rate * tick_seconds * 0.5))
    result = np.zeros(tick_count, dtype=np.int16)
    for tick in range(tick_count):
        center = _round_tick(
            (timeline_origin_seconds + tick * tick_seconds) * sample_rate
        )
        start = max(0, center - half_window)
        end = min(left.shape[-1], center + half_window + 1)
        if end <= start:
            continue
        left_rms = math.sqrt(float(np.mean(np.square(left[start:end]))))
        right_rms = math.sqrt(float(np.mean(np.square(right[start:end]))))
        total = left_rms + right_rms
        if total > 1e-9:
            # Keep some headroom: hard-panned note blocks are rarely pleasant.
            result[tick] = round(80.0 * (right_rms - left_rms) / total)
    return result


def _calculate_tick_loudness(
    audio,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    np,
):
    """Measure a stable 0..1 log-loudness envelope on the shared NBS clock."""

    if audio is None or audio.size == 0:
        return np.zeros(tick_count, dtype=np.float32)
    samples = np.nan_to_num(
        np.asarray(audio, dtype=np.float32), copy=True
    )
    squared = np.square(samples, dtype=np.float64)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(squared, dtype=np.float64))
    )

    # This broadband envelope is retained only as a compatibility fallback for
    # sources on which pitch analysis cannot run.  Keep its window narrow so a
    # neighboring attack cannot lend its volume to the current note.
    centers = (
        timeline_origin_seconds + np.arange(tick_count) * tick_seconds
    ) * sample_rate
    starts = np.floor(centers - 0.02 * sample_rate).astype(np.int64)
    ends = np.ceil(
        centers + min(0.08, max(0.05, tick_seconds * 0.60)) * sample_rate
    ).astype(np.int64)
    starts = np.clip(starts, 0, len(samples))
    ends = np.clip(ends, 0, len(samples))
    counts = np.maximum(1, ends - starts)
    rms = np.sqrt(np.maximum(0.0, cumulative[ends] - cumulative[starts]) / counts)

    peak = float(np.max(rms))
    if peak < 1e-9:
        return np.zeros(tick_count, dtype=np.float32)
    active = rms[rms >= peak * 0.01]
    reference = max(
        float(np.quantile(active, 0.95)) if active.size else peak,
        1e-9,
    )
    relative_db = 20.0 * np.log10(np.maximum(rms, reference * 1e-4) / reference)
    return np.clip((relative_db + 40.0) / 40.0, 0.0, 1.0).astype(
        np.float32
    )


def _apply_audio_loudness(
    notes: Iterable[NbsNote],
    audio,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    np,
) -> list[NbsNote]:
    """Replace confidence-like velocities with measured source dynamics."""

    note_list = list(notes)
    if not note_list:
        return []
    loudness = _calculate_tick_loudness(
        audio,
        sample_rate,
        timeline_origin_seconds,
        tick_seconds,
        tick_count,
        np,
    )
    result: list[NbsNote] = []
    for note in note_list:
        if not 0 <= note.tick < tick_count:
            continue
        if note.continuation:
            level = float(loudness[note.tick])
            # An overlong neural release must not keep striking after the stem
            # itself has faded into its noise floor.
            if level < 0.08:
                continue
        else:
            level = float(loudness[note.tick])

        confidence = _clamp(note.velocity / 100.0, 0.0, 1.0)
        dynamic = 0.04 + 0.96 * (level ** 0.72)
        confidence_trim = 0.55 + 0.45 * math.sqrt(confidence)
        velocity = 100.0 * dynamic * confidence_trim
        if note.continuation:
            velocity *= 0.72
        result.append(
            replace(
                note,
                velocity=int(_clamp(_round_tick(velocity), 1, 100)),
                audio_dynamic=True,
            )
        )
    return result


def _measure_nbs_note_pitch(
    note: NbsNote,
    analysis: _PitchAnalysis,
    fallback_seconds: float,
    np,
    *,
    tracks=None,
) -> tuple[float, float] | None:
    """Measure only this note's source pitch around its canonical onset."""

    midi = note.source_midi
    if midi is None:
        return None
    if tracks is None:
        tracks = _pitch_analysis_tracks(analysis, midi, np)
    if tracks is None:
        return None
    pitch_index, magnitude_track, _flux_track = tracks
    seconds = (
        note.source_time_seconds
        if note.source_time_seconds is not None
        else fallback_seconds
    )
    before = 0.045 if note.continuation else 0.020
    after = 0.055 if note.continuation else 0.100
    start_frame = max(0, _time_to_analysis_frame(seconds - before, analysis))
    end_frame = min(
        len(magnitude_track),
        _time_to_analysis_frame(seconds + after, analysis) + 1,
    )
    values = magnitude_track[start_frame:end_frame]
    if values.size == 0:
        return 0.0, -math.inf
    signal = max(float(np.quantile(values, 0.80)), 1e-12)
    absolute_db = 20.0 * math.log10(signal / analysis.global_reference)
    pitch_loudness = _clamp((absolute_db + 50.0) / 50.0, 0.0, 1.0)

    neighborhood = []
    for distance in range(2, 7):
        for neighbor in (pitch_index - distance, pitch_index + distance):
            if 0 <= neighbor < analysis.magnitude.shape[0]:
                neighbor_values = analysis.magnitude[
                    neighbor, start_frame:end_frame
                ]
                if neighbor_values.size:
                    neighborhood.append(
                        float(np.quantile(neighbor_values, 0.65))
                    )
    noise = max(
        analysis.global_reference * 1e-5,
        float(np.median(neighborhood)) if neighborhood else 0.0,
    )
    pitch_snr_db = 20.0 * math.log10(signal / noise)
    return pitch_loudness, pitch_snr_db


def _apply_pitch_loudness(
    notes: Iterable[NbsNote],
    audio,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    config: ConversionConfig,
    librosa,
    np,
) -> list[NbsNote]:
    """Set dynamics and reject silence from each note's own pitch band."""

    note_list = list(notes)
    if not note_list:
        return []
    analysis = _build_pitch_analysis(audio, sample_rate, librosa, np)
    if analysis is None:
        return _apply_audio_loudness(
            note_list,
            audio,
            sample_rate,
            timeline_origin_seconds,
            tick_seconds,
            tick_count,
            np,
        )

    result: list[NbsNote] = []
    tracks_by_midi = {
        midi: _pitch_analysis_tracks(analysis, midi, np)
        for midi in {
            note.source_midi
            for note in note_list
            if note.source_midi is not None
        }
    }
    for note in note_list:
        if not 0 <= note.tick < tick_count:
            continue
        fallback_seconds = timeline_origin_seconds + note.tick * tick_seconds
        measured = _measure_nbs_note_pitch(
            note,
            analysis,
            fallback_seconds,
            np,
            tracks=tracks_by_midi.get(note.source_midi),
        )
        if measured is None:
            # A legacy/internal note without its original MIDI cannot be checked
            # pitch-wise.  Do not pretend a broadband level is pitch evidence.
            result.append(note)
            continue
        pitch_loudness, pitch_snr_db = measured
        if note.continuation:
            if pitch_loudness < 0.020 or (
                pitch_loudness < 0.055 and pitch_snr_db < 6.0
            ):
                continue
        else:
            absolute_floor = 0.045 - 0.025 * config.sensitivity
            if pitch_loudness < 0.015 or (
                pitch_loudness < absolute_floor and pitch_snr_db < 8.0
            ):
                continue
            if pitch_snr_db < -4.0 and pitch_loudness < 0.18:
                continue

        confidence = _clamp(note.velocity / 100.0, 0.0, 1.0)
        velocity = (
            100.0
            * (0.03 + 0.97 * (pitch_loudness ** 0.72))
            * (0.55 + 0.45 * math.sqrt(confidence))
        )
        if note.continuation:
            velocity *= 0.65
        result.append(
            replace(
                note,
                velocity=int(_clamp(_round_tick(velocity), 1, 100)),
                audio_dynamic=True,
                source_loudness=pitch_loudness,
                source_snr_db=pitch_snr_db,
            )
        )
    return result


def _weighted_median(values, weights, np) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    midpoint = float(np.sum(sorted_weights)) * 0.5
    index = int(np.searchsorted(np.cumsum(sorted_weights), midpoint, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _extract_monophonic_notes(
    audio,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    panning_by_tick,
    config: ConversionConfig,
    *,
    layer: int,
    default_instrument: int,
    minimum_note: str,
    maximum_note: str,
    velocity_scale: float,
    librosa,
    np,
) -> list[NbsNote]:
    """Extract one stable pitch track, suitable for vocals or bass."""

    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-6:
        return []
    minimum_hz = float(librosa.note_to_hz(minimum_note))
    maximum_hz = min(
        float(librosa.note_to_hz(maximum_note)), sample_rate * 0.47
    )
    if minimum_hz >= maximum_hz:
        return []
    required_window = max(2048, math.ceil(3.0 * sample_rate / minimum_hz))
    frame_length = min(
        8192,
        1 << math.ceil(math.log2(required_window)),
    )
    try:
        f0, voiced, probability = librosa.pyin(
            audio,
            fmin=minimum_hz,
            fmax=maximum_hz,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=config.hop_length,
        )
    except Exception:
        return []

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=config.hop_length
    )[0]
    frame_count = min(len(f0), len(voiced), len(probability), len(rms))
    if frame_count == 0:
        return []
    f0 = f0[:frame_count]
    voiced = voiced[:frame_count]
    probability = probability[:frame_count]
    rms = rms[:frame_count]

    positive_rms = rms[rms > 0]
    if len(positive_rms) == 0:
        return []
    strength_reference = max(float(np.percentile(positive_rms, 95)), 1e-9)
    strength_floor = strength_reference * (0.13 - 0.08 * config.sensitivity)
    confidence_floor = 0.72 - 0.22 * config.sensitivity
    frame_times = librosa.frames_to_time(
        np.arange(frame_count), sr=sample_rate, hop_length=config.hop_length
    )
    frame_ticks = _round_tick_array(
        (frame_times - timeline_origin_seconds) / tick_seconds, np
    )

    tick_pitch = np.full(tick_count, np.nan, dtype=np.float32)
    tick_strength = np.zeros(tick_count, dtype=np.float32)
    for tick in np.unique(frame_ticks):
        tick_int = int(tick)
        if not 0 <= tick_int < tick_count:
            continue
        indices = np.flatnonzero(frame_ticks == tick_int)
        valid = indices[
            voiced[indices]
            & np.isfinite(f0[indices])
            & (probability[indices] >= confidence_floor)
            & (rms[indices] >= strength_floor)
        ]
        if len(valid) == 0:
            continue
        midi_values = librosa.hz_to_midi(f0[valid])
        weights = probability[valid] * np.maximum(rms[valid], 1e-9)
        tick_pitch[tick_int] = round(_weighted_median(midi_values, weights, np))
        tick_strength[tick_int] = float(np.max(rms[valid]))

    # Suppress one-tick vibrato and octave glitches without flattening phrases.
    smoothed = tick_pitch.copy()
    for tick in range(1, tick_count - 1):
        window = tick_pitch[tick - 1 : tick + 2]
        finite = window[np.isfinite(window)]
        if len(finite) >= 2 and float(np.max(finite) - np.min(finite)) <= 3.0:
            smoothed[tick] = round(float(np.median(finite)))

    instrument = (
        default_instrument if config.instrument is None else config.instrument
    )
    retrigger_interval = _retrigger_interval_ticks_exact(config)
    active_pitch: int | None = None
    last_voiced_tick = -10_000
    next_retrigger: float | None = None
    notes: list[NbsNote] = []
    for tick in range(tick_count):
        if not np.isfinite(smoothed[tick]):
            if tick - last_voiced_tick > 1:
                active_pitch = None
            continue

        midi = int(smoothed[tick])
        if not NBS_LOWEST_MIDI <= midi <= NBS_LOWEST_MIDI + NBS_KEY_MAX:
            continue
        newly_present = active_pitch != midi
        periodic = (
            retrigger_interval > 0.0
            and next_retrigger is not None
            and tick >= _round_tick(next_retrigger)
        )
        if newly_present or periodic:
            normalized = _clamp(
                float(tick_strength[tick]) / strength_reference, 0.0, 1.0
            )
            velocity = round((35.0 + 65.0 * math.sqrt(normalized)) * velocity_scale)
            raw_key = midi - NBS_LOWEST_MIDI
            key = (
                fold_key_to_minecraft_range(raw_key)
                if config.minecraft_range
                else raw_key
            )
            notes.append(
                NbsNote(
                    tick=tick,
                    layer=layer,
                    instrument=instrument,
                    key=key,
                    velocity=int(_clamp(velocity, 1, 100)),
                    panning=int(panning_by_tick[tick]),
                    continuation=periodic and not newly_present,
                    source_midi=midi,
                    source_time_seconds=(
                        timeline_origin_seconds + tick * tick_seconds
                    ),
                    source_loudness=normalized,
                )
            )
            if retrigger_interval > 0.0:
                if newly_present or next_retrigger is None:
                    next_retrigger = tick + retrigger_interval
                else:
                    next_retrigger = _advance_retrigger_deadline(
                        next_retrigger,
                        tick,
                        retrigger_interval,
                    )
            else:
                next_retrigger = None
        active_pitch = midi
        last_voiced_tick = tick
    return notes


def _select_drum_components(
    component_scores: Sequence[float], sensitivity: float
) -> list[int]:
    """Select independently supported kick, snare, and hi-hat components."""

    if len(component_scores) != 3:
        raise ValueError("Exactly three drum component scores are required")
    scores = [float(_clamp(score, 0.0, 1.0)) for score in component_scores]
    strongest = max(scores)
    if strongest < 0.30 - 0.10 * sensitivity:
        return []
    selected = [scores.index(strongest)]
    additional_floor = max(0.55 - 0.10 * sensitivity, strongest * 0.75)
    additional = [
        index
        for index, score in enumerate(scores)
        if index not in selected and score >= additional_floor
    ]
    selected.extend(
        sorted(additional, key=lambda index: scores[index], reverse=True)
    )
    return selected


def _extract_drum_notes(
    percussive,
    onset_envelope,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    panning_by_tick,
    config: ConversionConfig,
    layer_offset: int,
    velocity_scale: float,
    librosa,
    np,
) -> list[NbsNote]:
    if (
        not config.include_drums
        or percussive.size == 0
        or float(np.max(np.abs(percussive))) < 1e-7
    ):
        return []

    magnitude = np.abs(
        librosa.stft(
            percussive,
            n_fft=2048,
            hop_length=config.hop_length,
            center=True,
        )
    )
    if magnitude.shape[1] == 0:
        return []
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    band_masks = (
        (frequencies >= 25.0) & (frequencies < 220.0),
        (frequencies >= 220.0) & (frequencies < 3_500.0),
        frequencies >= 4_500.0,
    )
    positive_flux = np.maximum(magnitude[:, 1:] - magnitude[:, :-1], 0.0)
    flux = np.pad(positive_flux, ((0, 0), (1, 0)))
    band_flux = []
    for mask in band_masks:
        if np.any(mask):
            band_flux.append(np.mean(flux[mask, :], axis=0))
        else:
            band_flux.append(np.zeros(magnitude.shape[1], dtype=np.float32))

    candidate_frames: set[int] = set()
    envelopes = [np.asarray(onset_envelope, dtype=np.float32), *band_flux]
    for envelope in envelopes:
        if envelope.size == 0 or not np.any(envelope > 0):
            continue
        try:
            detected = librosa.onset.onset_detect(
                onset_envelope=envelope,
                sr=sample_rate,
                hop_length=config.hop_length,
                units="frames",
                backtrack=False,
            )
        except (ValueError, RuntimeError):
            continue
        candidate_frames.update(int(frame) for frame in detected)
    if not candidate_frames:
        return []

    band_references = []
    for envelope in band_flux:
        positive = envelope[envelope > 0]
        band_references.append(
            max(float(np.quantile(positive, 0.95)), 1e-12)
            if positive.size
            else 1.0
        )
    combined_flux = np.sum(np.stack(band_flux, axis=0), axis=0)
    positive_combined = combined_flux[combined_flux > 0]
    combined_reference = (
        max(float(np.quantile(positive_combined, 0.95)), 1e-12)
        if positive_combined.size
        else 1.0
    )
    minimum_combined = 0.18 - 0.10 * config.sensitivity

    instrument_by_component = (
        INSTRUMENTS["bass_drum"],
        INSTRUMENTS["snare"],
        INSTRUMENTS["hat"],
    )
    # Independent sub-band detectors often report neighboring frames for one
    # physical hit.  Form bounded (non-transitive) attack groups before drum
    # classification so one kick cannot become a two-tick flam merely because
    # its low and high-frequency edges peaked at different frames.
    grouping_radius = max(
        1, _round_tick(0.025 * sample_rate / config.hop_length)
    )
    onset_groups: list[list[int]] = []
    for frame in sorted(candidate_frames):
        if (
            onset_groups
            and frame - onset_groups[-1][0] <= grouping_radius
        ):
            onset_groups[-1].append(frame)
        else:
            onset_groups.append([frame])

    # Independent sub-band detection can preserve a kick and hi-hat that occur
    # together. Keep only the strongest event after multiple attacks quantize
    # to the same NBS slot.
    hits: dict[tuple[int, int], NbsNote] = {}
    for onset_group in onset_groups:
        onset_frame = max(
            onset_group,
            key=lambda frame: (
                float(combined_flux[frame])
                if 0 <= frame < len(combined_flux)
                else -math.inf,
                -frame,
            ),
        )
        if not 0 <= onset_frame < magnitude.shape[1]:
            continue
        combined_score = _clamp(
            float(combined_flux[onset_frame]) / combined_reference, 0.0, 1.0
        )
        if combined_score < minimum_combined:
            continue
        # Quadratic peak interpolation removes the last whole-hop bias while
        # retaining the absolute source clock.
        fractional_frame = float(onset_frame)
        if 0 < onset_frame and onset_frame + 1 < len(combined_flux):
            left = float(combined_flux[onset_frame - 1])
            center = float(combined_flux[onset_frame])
            right = float(combined_flux[onset_frame + 1])
            curvature = left - 2.0 * center + right
            if abs(curvature) > 1e-12:
                fractional_frame += _clamp(
                    0.5 * (left - right) / curvature, -0.5, 0.5
                )
        onset_time = (
            fractional_frame * config.hop_length / sample_rate
        )
        tick = _seconds_to_tick(
            onset_time, timeline_origin_seconds, tick_seconds
        )
        if not 0 <= tick < tick_count:
            continue

        start = max(0, min(onset_group) - 1)
        end = min(magnitude.shape[1], max(onset_group) + 2)
        raw_components = [
            float(np.max(envelope[start:end])) if end > start else 0.0
            for envelope in band_flux
        ]
        component_total = sum(raw_components)
        if component_total <= 1e-12:
            continue
        component_scores = [
            _clamp(
                0.75 * raw / reference
                + 0.25 * min(1.0, (raw / component_total) * 2.5),
                0.0,
                1.0,
            )
            for raw, reference in zip(raw_components, band_references)
        ]
        selected_components = _select_drum_components(
            component_scores, config.sensitivity
        )
        for drum_layer in selected_components:
            strength = math.sqrt(
                max(0.0, component_scores[drum_layer] * combined_score)
            )
            velocity = round((35.0 + 65.0 * strength) * velocity_scale)
            note = NbsNote(
                tick=tick,
                layer=layer_offset + drum_layer,
                instrument=instrument_by_component[drum_layer],
                key=45,
                velocity=int(_clamp(velocity, 1, 100)),
                panning=int(panning_by_tick[tick]),
                source_time_seconds=onset_time,
            )
            position = (tick, note.layer)
            previous = hits.get(position)
            if previous is None or note.velocity > previous.velocity:
                hits[position] = note

    return sorted(hits.values(), key=lambda note: (note.tick, note.layer))


def _pitch_events_to_nbs(
    events: Iterable[_PitchEvent],
    config: ConversionConfig,
    *,
    layer_offset: int = 0,
    velocity_scale: float = 1.0,
) -> list[NbsNote]:
    by_tick: dict[int, list[_PitchEvent]] = defaultdict(list)
    for event in events:
        by_tick[event.tick].append(event)

    result: list[NbsNote] = []
    for tick, chord in by_tick.items():
        # Folding distant octaves can create duplicate NBS notes. Retain the louder one.
        deduplicated: dict[tuple[int, int], _PitchEvent] = {}
        for event in chord:
            raw_key = event.midi - NBS_LOWEST_MIDI
            key = (
                fold_key_to_minecraft_range(raw_key)
                if config.minecraft_range
                else raw_key
            )
            instrument = choose_instrument(event.midi, config.instrument)
            identity = (instrument, key)
            previous = deduplicated.get(identity)
            if previous is None or event.velocity > previous.velocity:
                deduplicated[identity] = event

        ordered: list[tuple[_PitchEvent, int, int]] = []
        for (instrument, key), event in deduplicated.items():
            ordered.append((event, instrument, key))
        ordered.sort(key=lambda item: item[0].midi)

        for voice, (event, instrument, key) in enumerate(
            ordered[: config.max_chord_notes]
        ):
            result.append(
                NbsNote(
                    tick=tick,
                    layer=layer_offset + voice,
                    instrument=instrument,
                    key=key,
                    velocity=int(
                        _clamp(round(event.velocity * velocity_scale), 1, 100)
                    ),
                    panning=event.panning,
                    continuation=event.continuation,
                    source_midi=event.midi,
                    source_time_seconds=event.source_time_seconds,
                )
            )
    return sorted(result, key=lambda note: (note.tick, note.layer))


def _minimum_accompaniment_event_count(
    audio,
    duration_seconds: float,
    librosa,
    np,
) -> int:
    """Estimate a conservative note floor from the amount of active audio."""

    if audio.size == 0 or duration_seconds <= 0.0:
        return 0
    frame_rms = librosa.feature.rms(
        y=np.asarray(audio, dtype=np.float32),
        frame_length=2048,
        hop_length=512,
    )[0]
    if frame_rms.size == 0:
        return 0
    reference = float(np.quantile(frame_rms, 0.95))
    if reference < 1e-7:
        return 0
    active_ratio = float(np.mean(frame_rms >= max(1e-7, reference * 0.035)))
    active_seconds = duration_seconds * active_ratio
    return max(4, math.ceil(active_seconds * 0.18))


def _select_local_accompaniment_recovery(
    primary: Iterable[NbsNote],
    recovery: Iterable[NbsNote],
    audio,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    config: ConversionConfig,
    librosa,
    np,
    *,
    globally_sparse: bool = False,
) -> list[NbsNote]:
    """Keep a ranked, bounded set of CQT notes in measurable AI gaps."""

    primary_notes = list(primary)
    recovery_notes = list(recovery)
    if not recovery_notes or audio.size == 0:
        return []
    layer_base = min(note.layer for note in recovery_notes)

    sustained_covered = np.zeros(tick_count, dtype=bool)
    for note in primary_notes:
        sustained_covered[
            max(0, note.tick - 5) : min(tick_count, note.tick + 6)
        ] = True

    primary_by_midi: dict[int, list[int]] = defaultdict(list)
    primary_by_sound: dict[tuple[int, int], list[int]] = defaultdict(list)
    for note in primary_notes:
        if note.source_midi is not None:
            primary_by_midi[note.source_midi].append(note.tick)
        primary_by_sound[(note.instrument, note.key)].append(note.tick)

    def pitch_is_covered(note: NbsNote, radius: int = 2) -> bool:
        if note.source_midi is not None:
            ticks = primary_by_midi.get(note.source_midi, ())
        else:
            ticks = primary_by_sound.get((note.instrument, note.key), ())
        return any(abs(note.tick - tick) <= radius for tick in ticks)

    allowed_priorities: dict[int, float] = {}

    def allow_tick(tick: int, priority: float) -> None:
        if 0 <= tick < tick_count:
            allowed_priorities[tick] = max(
                allowed_priorities.get(tick, 0.0), priority
            )

    try:
        onset_hop = 256
        onset_envelope = librosa.onset.onset_strength(
            y=audio,
            sr=sample_rate,
            hop_length=onset_hop,
            aggregate=np.median,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=onset_hop,
            backtrack=False,
            units="frames",
        )
        if len(onset_frames):
            strengths = onset_envelope[onset_frames]
            quantile = _clamp(0.40 - 0.35 * config.sensitivity, 0.05, 0.40)
            strength_floor = float(np.quantile(strengths, quantile))
            strength_reference = max(float(np.quantile(strengths, 0.95)), 1e-9)
            onset_times = librosa.frames_to_time(
                onset_frames, sr=sample_rate, hop_length=onset_hop
            )
            for onset_time, strength in zip(onset_times, strengths):
                tick = _seconds_to_tick(
                    float(onset_time),
                    timeline_origin_seconds,
                    tick_seconds,
                )
                if (
                    float(strength) >= strength_floor
                    and 0 <= tick < tick_count
                ):
                    priority = 0.60 + 0.40 * _clamp(
                        float(strength) / strength_reference, 0.0, 1.0
                    )
                    allow_tick(tick - 1, priority * 0.85)
                    allow_tick(tick, priority)
                    allow_tick(tick + 1, priority * 0.85)

        # A sustained pad or held guitar may have no fresh onset.  Recover only
        # genuinely active runs that remain uncovered for at least 0.8 seconds.
        rms_hop = 512
        frame_rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=rms_hop,
        )[0]
        if frame_rms.size:
            active_floor = float(np.quantile(frame_rms, 0.90)) * 0.10
            frame_times = librosa.frames_to_time(
                np.arange(frame_rms.size), sr=sample_rate, hop_length=rms_hop
            )
            active_ticks = sorted(
                {
                    _seconds_to_tick(
                        float(frame_time),
                        timeline_origin_seconds,
                        tick_seconds,
                    )
                    for frame_time, rms in zip(frame_times, frame_rms)
                    if float(rms) >= active_floor
                }
            )
            uncovered_runs: list[list[int]] = []
            for tick in active_ticks:
                if not 0 <= tick < tick_count or bool(sustained_covered[tick]):
                    continue
                if not uncovered_runs or tick > uncovered_runs[-1][-1] + 1:
                    uncovered_runs.append([tick])
                else:
                    uncovered_runs[-1].append(tick)
            minimum_gap_ticks = max(2, math.ceil(0.8 / tick_seconds))
            for run in uncovered_runs:
                if len(run) < minimum_gap_ticks:
                    continue
                allow_tick(run[0], 0.45)
                for tick in _iter_phase_locked_retrigger_ticks(
                    run[0], run[-1] + 1, config
                ):
                    allow_tick(tick, 0.45)
    except (ValueError, RuntimeError):
        return []

    candidates_by_tick: dict[int, dict[tuple[int, int], NbsNote]] = defaultdict(dict)
    for note in recovery_notes:
        if note.tick not in allowed_priorities or pitch_is_covered(note):
            continue
        identity = (note.instrument, note.key)
        previous = candidates_by_tick[note.tick].get(identity)
        if previous is None or note.velocity > previous.velocity:
            candidates_by_tick[note.tick][identity] = note

    tick_groups: dict[int, list[NbsNote]] = {}
    tick_scores: dict[int, float] = {}
    for tick, candidates in candidates_by_tick.items():
        notes = sorted(
            candidates.values(), key=lambda note: note.velocity, reverse=True
        )
        recovery_voice_offset = (
            0 if globally_sparse or config.max_chord_notes == 1 else 1
        )
        notes = notes[: max(0, config.max_chord_notes - recovery_voice_offset)]
        if not notes:
            continue
        tick_groups[tick] = notes
        tick_scores[tick] = allowed_priorities[tick] * (
            max(note.velocity for note in notes) / 100.0
        )

    # A single detected attack permits tick-1/tick/tick+1 for quantization
    # tolerance. Select only the best of adjacent recovery ticks so that the
    # safety net cannot turn one attack into a three-note temporal smear.
    nonadjacent_ticks: list[int] = []
    for tick in sorted(tick_groups, key=lambda item: tick_scores[item], reverse=True):
        if any(abs(tick - selected_tick) <= 1 for selected_tick in nonadjacent_ticks):
            continue
        nonadjacent_ticks.append(tick)

    # Every retained position is tied to an independently detected physical
    # onset (or a long uncovered active run) and to an uncovered CQT pitch.
    # Limiting additions to 35% of the primary transcription prevented the
    # safety pass from repairing systematically omitted inner chord tones.
    # Keep only a pathological-density ceiling; ordinary detected onsets pass.
    duration_seconds = tick_count * tick_seconds
    maximum_onsets = max(8, math.ceil(duration_seconds * 20.0))
    selected_ticks = set(
        sorted(nonadjacent_ticks, key=lambda item: tick_scores[item], reverse=True)[
            :maximum_onsets
        ]
    )

    selected: list[NbsNote] = []
    recovery_voice_offset = (
        0 if globally_sparse or config.max_chord_notes == 1 else 1
    )
    for tick in sorted(selected_ticks):
        for voice, note in enumerate(
            sorted(tick_groups[tick], key=lambda item: (item.key, item.instrument))
        ):
            selected.append(
                replace(note, layer=layer_base + recovery_voice_offset + voice)
            )
    return selected


def _extract_cqt_accompaniment_notes(
    audio,
    sample_rate: int,
    timeline_origin_seconds: float,
    tick_seconds: float,
    tick_count: int,
    panning_by_tick,
    config: ConversionConfig,
    *,
    layer_offset: int,
    librosa,
    np,
    recovery: bool = False,
) -> list[NbsNote]:
    """Transcribe accompaniment with CQT, also serving as an AI safety net."""

    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-7:
        if recovery:
            return []
        raise ConversionError("No pitch could be detected.")
    try:
        cqt = np.abs(
            librosa.cqt(
                audio,
                sr=sample_rate,
                hop_length=config.hop_length,
                fmin=float(librosa.note_to_hz("A0")),
                n_bins=88,
                bins_per_octave=12,
            )
        )
    except Exception as exc:
        if recovery:
            return []
        raise ConversionError(
            "Pitch analysis failed; the input audio may be too short."
        ) from exc
    cqt = np.nan_to_num(cqt, copy=False)
    if not np.any(cqt > 0):
        if recovery:
            return []
        raise ConversionError("No pitch could be detected.")

    frame_times = librosa.frames_to_time(
        np.arange(cqt.shape[1]), sr=sample_rate, hop_length=config.hop_length
    )
    frame_ticks = _round_tick_array(
        (frame_times - timeline_origin_seconds) / tick_seconds, np
    )
    tick_energy, tick_source_times = _aggregate_by_tick(
        cqt,
        frame_ticks,
        tick_count,
        np,
        frame_times=frame_times,
    )
    reference = float(np.max(tick_energy))
    tick_db = librosa.amplitude_to_db(
        tick_energy, ref=max(reference, 1e-12), top_db=80.0
    )
    # Do not erase low accompaniment bins merely because a bass stem exists.
    # Demucs can route a piano left hand, low guitar, organ pedal, or orchestral
    # voice into ``other``; exact cross-stem duplicates are removed after both
    # sources have been transcribed.

    analysis_config = (
        replace(config, sensitivity=max(config.sensitivity, 0.68))
        if recovery
        else config
    )
    pitch_events = _extract_pitch_events(
        tick_db,
        analysis_config,
        panning_by_tick,
        np,
        unique_pitch_classes=False,
        source_times=tick_source_times,
        timeline_origin_seconds=timeline_origin_seconds,
        tick_seconds=tick_seconds,
    )
    return _pitch_events_to_nbs(
        pitch_events,
        config,
        layer_offset=layer_offset,
        velocity_scale=(
            0.68
            if recovery
            else (0.72 if config.separation == "demucs" else 1.0)
        ),
    )


def _merge_accompaniment_notes(
    primary: Iterable[NbsNote],
    recovery: Iterable[NbsNote],
    *,
    layer_offset: int,
    max_notes: int,
) -> list[NbsNote]:
    """Fill sparse AI chords from CQT without creating layer collisions."""

    primary_notes = list(primary)
    candidates: dict[int, list[tuple[NbsNote, bool]]] = defaultdict(list)
    for note in primary_notes:
        candidates[note.tick].append((note, True))
    for note in recovery:
        candidates[note.tick].append((note, False))

    reserved_background_layers = {
        note.layer for note in primary_notes if note.source_role is not None
    }
    merged: list[NbsNote] = []
    for tick in sorted(candidates):
        deduplicated: dict[tuple[int, int], tuple[NbsNote, bool]] = {}
        for note, is_primary in candidates[tick]:
            identity = (note.instrument, note.key)
            previous = deduplicated.get(identity)
            if previous is None or (is_primary, note.velocity) > (
                previous[1],
                previous[0].velocity,
            ):
                deduplicated[identity] = (note, is_primary)
        selected = sorted(
            deduplicated.values(),
            key=lambda item: (item[1], item[0].velocity),
            reverse=True,
        )[:max_notes]
        available_layers = set(range(layer_offset, layer_offset + max_notes))
        for note, is_primary in selected:
            preferred_layer = note.layer
            eligible_layers = available_layers
            if not is_primary and note.source_role is None:
                eligible_layers = available_layers - reserved_background_layers
            if preferred_layer not in eligible_layers:
                if not eligible_layers:
                    break
                preferred_layer = min(eligible_layers)
            merged.append(replace(note, layer=preferred_layer))
            available_layers.remove(preferred_layer)
    return merged


def _merge_essential_line_notes(
    primary: Iterable[NbsNote],
    fallback: Iterable[NbsNote],
    *,
    coverage_radius_ticks: int,
    neighbor_gap_ticks: int,
) -> list[NbsNote]:
    """Fill phrase-sized neural gaps from a conservative monophonic tracker.

    Neural notes always win collisions.  A fallback note is admitted only when
    it lies outside the neural coverage radius and is either strong on its own
    or belongs to a nearby fallback phrase.  This protects vocals and bass from
    complete dropouts without layering two competing pitch tracks.
    """

    primary_notes = sorted(primary, key=lambda note: (note.tick, note.layer))
    fallback_by_tick: dict[int, NbsNote] = {}
    for note in fallback:
        previous = fallback_by_tick.get(note.tick)
        if previous is None or note.velocity > previous.velocity:
            fallback_by_tick[note.tick] = note
    fallback_notes = [fallback_by_tick[tick] for tick in sorted(fallback_by_tick)]
    if not fallback_notes:
        return primary_notes
    if not primary_notes:
        return fallback_notes

    primary_ticks = [note.tick for note in primary_notes]
    fallback_ticks = [note.tick for note in fallback_notes]
    additions: list[NbsNote] = []
    for index, note in enumerate(fallback_notes):
        if any(
            abs(note.tick - primary_tick) <= coverage_radius_ticks
            for primary_tick in primary_ticks
        ):
            continue
        previous_gap = (
            note.tick - fallback_ticks[index - 1]
            if index > 0
            else neighbor_gap_ticks + 1
        )
        next_gap = (
            fallback_ticks[index + 1] - note.tick
            if index + 1 < len(fallback_ticks)
            else neighbor_gap_ticks + 1
        )
        phrase_supported = min(previous_gap, next_gap) <= neighbor_gap_ticks
        if note.velocity < 65 and not phrase_supported:
            continue
        additions.append(note)

    occupied = {(note.tick, note.layer) for note in primary_notes}
    return sorted(
        primary_notes
        + [note for note in additions if (note.tick, note.layer) not in occupied],
        key=lambda note: (note.tick, note.layer),
    )


def _remove_cross_stem_duplicates(
    background: Iterable[NbsNote], foreground: Iterable[NbsNote]
) -> list[NbsNote]:
    """Remove only an exact same-instrument duplicate at the same tick.

    At this stage octave information has already been folded for Minecraft, so
    treating nearby matching pitch classes as duplicates can erase real chords.
    Duration-aware leakage removal happens before this function.
    """

    blocked = {(note.tick, note.instrument, note.key) for note in foreground}
    return [
        note
        for note in background
        if (note.tick, note.instrument, note.key) not in blocked
    ]


def _stabilize_layer_instruments(
    notes: Iterable[NbsNote],
    *,
    layer_offset: int,
    layer_count: int,
    requested_instrument: int | None,
) -> list[NbsNote]:
    """Use one dominant timbre per accompaniment layer for the whole song."""

    note_list = list(notes)
    instrument_by_layer: dict[int, int] = {}
    for layer in range(layer_offset, layer_offset + layer_count):
        if requested_instrument is not None:
            instrument_by_layer[layer] = requested_instrument
            continue
        voice = layer - layer_offset
        if voice == 1:
            instrument_by_layer[layer] = INSTRUMENTS["guitar"]
            continue
        source_votes: dict[str, float] = defaultdict(float)
        for note in note_list:
            if (
                note.layer == layer
                and note.source_role in BACKGROUND_ROLE_INSTRUMENTS
            ):
                source_votes[note.source_role] += max(1, note.velocity)
        if voice >= 3 and source_votes:
            source_role = max(
                source_votes,
                key=lambda role: (source_votes[role], role),
            )
            instrument_by_layer[layer] = BACKGROUND_ROLE_INSTRUMENTS[source_role]
            continue
        if voice >= 2:
            instrument_by_layer[layer] = INSTRUMENTS["piano"]
            continue
        votes: dict[int, float] = defaultdict(float)
        for note in note_list:
            if note.layer == layer:
                votes[note.instrument] += max(1, note.velocity)
        if votes:
            instrument_by_layer[layer] = max(
                votes, key=lambda instrument: (votes[instrument], -instrument)
            )
    return [
        replace(
            note,
            instrument=instrument_by_layer.get(note.layer, note.instrument),
        )
        for note in note_list
    ]


def _normalize_role_dynamics(
    notes: Iterable[NbsNote],
    *,
    target_p90: float,
    minimum: int,
    maximum: int,
    maximum_step: int,
    phrase_gap_ticks: int,
    continuation_ratio: float = 0.55,
    audio_minimum: int | None = None,
) -> list[NbsNote]:
    """Normalize true attacks while keeping sustain continuations subordinate."""

    note_list = sorted(notes, key=lambda note: (note.tick, note.layer))
    if not note_list:
        return []
    attack_notes = [note for note in note_list if not note.continuation]
    velocities = sorted(
        note.velocity for note in (attack_notes if attack_notes else note_list)
    )
    percentile_index = min(len(velocities) - 1, math.ceil(len(velocities) * 0.90) - 1)
    reference = max(1, velocities[percentile_index])
    scale = _clamp(target_p90 / reference, 0.75, 1.35)
    previous_attack_by_layer: dict[int, tuple[int, int]] = {}
    result: list[NbsNote] = []
    for note in note_list:
        scaled_velocity = round(note.velocity * scale)
        previous = previous_attack_by_layer.get(note.layer)
        if note.continuation:
            anchor_velocity = previous[1] if previous is not None else maximum
            continuation_maximum = max(
                1, round(anchor_velocity * continuation_ratio)
            )
            velocity = int(
                _clamp(
                    scaled_velocity,
                    1,
                    min(maximum, continuation_maximum),
                )
            )
            result.append(replace(note, velocity=velocity))
            continue

        effective_minimum = (
            minimum
            if audio_minimum is None or not note.audio_dynamic
            else audio_minimum
        )
        velocity = int(_clamp(scaled_velocity, effective_minimum, maximum))
        if (
            previous is not None
            and not note.audio_dynamic
            and note.tick - previous[0] <= phrase_gap_ticks
        ):
            velocity = int(
                _clamp(
                    velocity,
                    previous[1] - maximum_step,
                    previous[1] + maximum_step,
                )
            )
            velocity = int(_clamp(velocity, effective_minimum, maximum))
        result.append(replace(note, velocity=velocity))
        previous_attack_by_layer[note.layer] = (note.tick, velocity)
    return result


def _balance_song_dynamics(
    vocal_notes: Iterable[NbsNote],
    bass_notes: Iterable[NbsNote],
    accompaniment_notes: Iterable[NbsNote],
    drum_notes: Iterable[NbsNote],
    *,
    use_vocals: bool,
    accompaniment_layer_offset: int,
    drum_layer_offset: int,
    max_chord_notes: int,
    tick_count: int,
    config: ConversionConfig,
) -> tuple[list[NbsNote], list[NbsNote], list[NbsNote], list[NbsNote]]:
    """Enforce a stable melody-first loudness hierarchy across the song."""

    retrigger_ticks = max(1, _retrigger_interval_ticks(config))
    phrase_gap = max(4, retrigger_ticks * 2)
    vocals = _normalize_role_dynamics(
        vocal_notes,
        target_p90=88.0,
        minimum=64,
        maximum=100,
        maximum_step=12,
        phrase_gap_ticks=phrase_gap,
        continuation_ratio=0.48,
        audio_minimum=1,
    )
    bass = _normalize_role_dynamics(
        bass_notes,
        target_p90=68.0,
        minimum=38,
        maximum=80,
        maximum_step=10,
        phrase_gap_ticks=phrase_gap,
        continuation_ratio=0.58,
        audio_minimum=1,
    )

    accompaniment_by_layer: dict[int, list[NbsNote]] = defaultdict(list)
    for note in accompaniment_notes:
        accompaniment_by_layer[note.layer].append(note)
    accompaniment: list[NbsNote] = []
    for voice in range(max_chord_notes):
        layer = accompaniment_layer_offset + voice
        if voice == 0:
            settings = (78.0, 45, 88, 10)
        elif voice == 1:
            settings = (58.0, 32, 68, 9)
        elif voice == 2:
            settings = (50.0, 26, 60, 8)
        else:
            # These lanes contain notes independently verified in a dedicated
            # instrument stem.  Keep them subordinate to the focus lane, but do
            # not bury a quiet main piano/guitar phrase below audibility.
            settings = (56.0, 20, 70, 8)
        accompaniment.extend(
            _normalize_role_dynamics(
                accompaniment_by_layer.get(layer, ()),
                target_p90=settings[0],
                minimum=settings[1],
                maximum=settings[2],
                maximum_step=settings[3],
                phrase_gap_ticks=phrase_gap,
                continuation_ratio=(
                    0.52 if voice == 0 else (0.46 if voice <= 2 else 0.42)
                ),
                audio_minimum=1,
            )
        )

    drums_by_layer: dict[int, list[NbsNote]] = defaultdict(list)
    for note in drum_notes:
        drums_by_layer[note.layer].append(note)
    drums: list[NbsNote] = []
    drum_settings = (
        (68.0, 36, 78),
        (64.0, 34, 74),
        (50.0, 28, 58),
    )
    for voice, settings in enumerate(drum_settings):
        drums.extend(
            _normalize_role_dynamics(
                drums_by_layer.get(drum_layer_offset + voice, ()),
                target_p90=settings[0],
                minimum=settings[1],
                maximum=settings[2],
                maximum_step=10,
                phrase_gap_ticks=phrase_gap,
            )
        )

    vocal_activity = [False] * tick_count
    vocal_ticks = sorted({note.tick for note in vocals})
    activity_radius = max(1, retrigger_ticks // 2)
    for tick in vocal_ticks:
        for active_tick in range(
            max(0, tick - activity_radius),
            min(tick_count, tick + activity_radius + 1),
        ):
            vocal_activity[active_tick] = True
    for first, second in zip(vocal_ticks, vocal_ticks[1:]):
        if second - first <= phrase_gap:
            for active_tick in range(first, min(tick_count, second + 1)):
                vocal_activity[active_tick] = True

    accompaniment_focus_ticks = {
        note.tick
        for note in accompaniment
        if note.layer == accompaniment_layer_offset
    }
    balanced_accompaniment: list[NbsNote] = []
    for note in accompaniment:
        velocity = note.velocity
        if (
            use_vocals
            and note.layer == accompaniment_layer_offset
            and 0 <= note.tick < tick_count
            and vocal_activity[note.tick]
        ):
            velocity = min(58, round(velocity * 0.70))
        elif (
            note.layer > accompaniment_layer_offset
            and note.tick in accompaniment_focus_ticks
        ):
            velocity = round(velocity * 0.86)
        balanced_accompaniment.append(
            replace(note, velocity=int(_clamp(velocity, 1, 100)))
        )

    balanced_bass = [
        replace(
            note,
            velocity=(
                min(62, round(note.velocity * 0.82))
                if use_vocals
                and 0 <= note.tick < tick_count
                and vocal_activity[note.tick]
                else note.velocity
            ),
        )
        for note in bass
    ]
    balanced_drums = [
        replace(
            note,
            velocity=(
                round(note.velocity * 0.78)
                if use_vocals
                and 0 <= note.tick < tick_count
                and vocal_activity[note.tick]
                else note.velocity
            ),
        )
        for note in drums
    ]
    return (
        sorted(vocals, key=lambda note: (note.tick, note.layer)),
        sorted(balanced_bass, key=lambda note: (note.tick, note.layer)),
        sorted(balanced_accompaniment, key=lambda note: (note.tick, note.layer)),
        sorted(balanced_drums, key=lambda note: (note.tick, note.layer)),
    )


def convert_audio_to_nbs(
    input_path: Path,
    output_path: Path,
    config: ConversionConfig,
    *,
    title: str,
    author: str,
    progress: Callable[[str], None] | None = None,
    progress_update: Callable[[float, str | None], None] | None = None,
) -> ConversionResult:
    """Analyze an audio file and save an approximate NBS transcription."""

    librosa, np = _load_audio_dependencies()
    report = progress or (lambda _message: None)

    def advance(fraction: float, status: str) -> None:
        if progress_update is not None:
            progress_update(fraction, status)

    advance(0.0, "Starting audio load")
    report("Loading audio...")
    try:
        channels, sample_rate = librosa.load(
            str(input_path), sr=config.sample_rate, mono=False, dtype=np.float32
        )
    except Exception as exc:
        raise ConversionError(
            "Could not read the audio file. Check that it is not damaged and "
            "that FFmpeg is available."
        ) from exc

    if channels.size == 0:
        raise ConversionError("The input audio is empty.")
    mono = channels if channels.ndim == 1 else np.mean(channels, axis=0)
    mono = np.asarray(mono, dtype=np.float32)
    mono -= float(np.mean(mono))
    normalization_peak = float(np.max(np.abs(mono)))
    if normalization_peak < 1e-6:
        raise ConversionError("The input audio contains no analyzable sound.")
    mono /= normalization_peak
    if channels.ndim == 1:
        channels = mono
    else:
        channels = np.asarray(channels, dtype=np.float32) / normalization_peak

    duration_seconds = float(len(mono) / sample_rate)
    advance(0.02, f"Audio loaded ({duration_seconds:.1f}s)")
    stem_delay_seconds = 0.0
    use_vocals = False
    vocal_handling = "unavailable"
    transcription_stem_paths: dict[str, Path] = {}
    transcription_timing_offsets: dict[str, float] = {}
    instrument_stems: dict[str, object] = {}
    selected_background_roles: list[str] = []
    if config.separation == "demucs":
        background_slot_count = (
            max(0, config.max_chord_notes - 3)
            if config.transcription == "ai"
            else 0
        )
        primary_separation_end = 0.62 if background_slot_count else 0.72
        stem_paths = _separate_audio_stems(
            input_path,
            report,
            progress_update,
            progress_start=0.03,
            progress_end=primary_separation_end,
        )
        instrument_stem_paths: dict[str, Path] = {}
        if background_slot_count:
            try:
                instrument_stem_paths = _separate_audio_stems(
                    input_path,
                    report,
                    progress_update,
                    progress_start=primary_separation_end,
                    progress_end=0.72,
                    model=DEMUCS_INSTRUMENT_MODEL,
                    model_count=DEMUCS_INSTRUMENT_MODEL_COUNT,
                    shifts=DEMUCS_INSTRUMENT_SHIFTS,
                    overlap=DEMUCS_OVERLAP,
                    expected_stems=DEMUCS_INSTRUMENT_STEMS,
                    cache_version=INSTRUMENT_SEPARATION_CACHE_VERSION,
                    description="piano, guitar, and residual instrument cues",
                )
            except ConversionError as exc:
                report(
                    "Warning: instrument-aware background separation is "
                    f"unavailable; continuing with the core arrangement ({exc})."
                )
                advance(0.72, "Background instrument cues unavailable")
        stems = {}
        report("Loading separated stems...")
        for stem_name, stem_path in stem_paths.items():
            stem_audio, _ = librosa.load(
                str(stem_path), sr=sample_rate, mono=True, dtype=np.float32
            )
            stem_audio = librosa.util.fix_length(stem_audio, size=len(mono))
            stems[stem_name] = np.asarray(stem_audio, dtype=np.float32) / max(
                normalization_peak, 1e-9
            )
        stem_delay_seconds = _estimate_stem_delay_seconds(
            mono, stems.values(), sample_rate, np
        )
        if abs(stem_delay_seconds) >= 1.0 / sample_rate:
            stems = {
                name: _shift_audio_to_timeline(
                    stem_audio, stem_delay_seconds, sample_rate, np
                )
                for name, stem_audio in stems.items()
            }
            report(
                "Corrected a shared stem delay of "
                f"{stem_delay_seconds * 1000.0:+.1f} ms."
            )
        instrument_stem_delay_seconds = 0.0
        if instrument_stem_paths:
            for stem_name, stem_path in instrument_stem_paths.items():
                stem_audio, _ = librosa.load(
                    str(stem_path), sr=sample_rate, mono=True, dtype=np.float32
                )
                stem_audio = librosa.util.fix_length(stem_audio, size=len(mono))
                instrument_stems[stem_name] = np.asarray(
                    stem_audio, dtype=np.float32
                ) / max(normalization_peak, 1e-9)
            instrument_stem_delay_seconds = _estimate_stem_delay_seconds(
                mono, instrument_stems.values(), sample_rate, np
            )
            if abs(instrument_stem_delay_seconds) >= 1.0 / sample_rate:
                instrument_stems = {
                    name: _shift_audio_to_timeline(
                        stem_audio,
                        instrument_stem_delay_seconds,
                        sample_rate,
                        np,
                    )
                    for name, stem_audio in instrument_stems.items()
                }
                report(
                    "Corrected an instrument-cue delay of "
                    f"{instrument_stem_delay_seconds * 1000.0:+.1f} ms."
                )
        advance(0.74, "Separated stems loaded")

        if config.vocals == "on":
            use_vocals = True
            vocal_handling = "forced_on"
            report("Forcing the dedicated vocal layer on.")
        elif config.vocals == "off":
            use_vocals = False
            vocal_handling = "forced_off"
            report("Disabling the dedicated vocal layer.")
        elif config.vocals == "auto":
            report("Detecting whether the track contains vocals...")

            def update_vocal_stage(fraction: float, status: str) -> None:
                advance(0.74 + 0.02 * fraction, status)

            reconstructed_mix = np.sum(
                np.stack(list(stems.values()), axis=0), axis=0
            )
            vocal_detection = _detect_vocal_presence(
                stems["vocals"],
                reconstructed_mix,
                sample_rate,
                librosa,
                np,
                report=report,
                progress_update=update_vocal_stage,
            )
            use_vocals = vocal_detection.present
            vocal_handling = "detected" if use_vocals else "instrumental"
            if use_vocals:
                report(
                    "Vocals detected "
                    f"({vocal_detection.longest_vocal_seconds:.1f}s continuous)."
                )
            else:
                report(
                    "No sustained vocals detected; merging the vocal residual "
                    "back into the accompaniment."
                )
        else:
            raise ConversionError(f"Unsupported vocal mode: {config.vocals}")

        # `--vocals off` means "do not create a vocal layer", not "throw the
        # entire Demucs vocal stem away".  That stem often contains a lead
        # synth, guitar or brass, so both forced-off and auto-instrumental modes
        # must return it to the accompaniment.
        merge_vocal_residual = not use_vocals
        accompaniment_source = stems["other"]
        if merge_vocal_residual:
            accompaniment_source = accompaniment_source + stems["vocals"]

        selected_background_roles = _select_background_stem_roles(
            instrument_stems,
            accompaniment_source,
            np,
            maximum_roles=background_slot_count,
        )
        if selected_background_roles:
            report(
                "Protecting independently separated background instruments: "
                + ", ".join(selected_background_roles)
                + "."
            )

        transcription_stem_paths = dict(stem_paths)
        transcription_timing_offsets = {
            role: -stem_delay_seconds for role in stem_paths
        }
        if merge_vocal_residual and config.transcription == "ai":
            transcription_stem_paths["other"] = (
                _write_instrumental_accompaniment_stem(
                    stem_paths["other"].parent,
                    accompaniment_source,
                    sample_rate,
                    normalization_peak,
                    np,
                )
            )
            # The merged cache is written from arrays already aligned to the
            # source timeline, unlike the original Demucs WAV files.
            transcription_timing_offsets["other"] = 0.0
        for role in selected_background_roles:
            transcription_stem_paths[role] = instrument_stem_paths[role]
            transcription_timing_offsets[role] = (
                -instrument_stem_delay_seconds
            )
        vocal_audio = stems["vocals"]
        bass_audio = stems["bass"]
        percussive = stems["drums"]
        accompaniment, _ = librosa.effects.hpss(
            accompaniment_source, margin=(2.0, 2.0)
        )
        advance(0.76, "Vocal and stem analysis complete")
    elif config.separation == "basic":
        advance(0.08, "Running basic source separation...")
        accompaniment, percussive = librosa.effects.hpss(
            mono, margin=(1.0, 2.0)
        )
        vocal_audio = None
        bass_audio = None
        advance(0.35, "Basic source separation complete")
    else:
        raise ConversionError(f"Unsupported separation mode: {config.separation}")

    report("Analyzing tempo and percussion...")
    advance(
        0.765 if config.separation == "demucs" else 0.38,
        "Analyzing tempo and beat positions...",
    )
    onset_envelope = librosa.onset.onset_strength(
        y=percussive,
        sr=sample_rate,
        hop_length=config.hop_length,
        aggregate=np.median,
    )
    if not np.any(onset_envelope > 0):
        fallback_percussive = librosa.effects.percussive(mono)
        onset_envelope = librosa.onset.onset_strength(
            y=fallback_percussive,
            sr=sample_rate,
            hop_length=config.hop_length,
            aggregate=np.median,
        )
    scalar_tempo, detected_bpm, beat_times = _estimate_tempo_and_beats(
        percussive,
        mono,
        sample_rate,
        config.hop_length,
        librosa,
        np,
    )
    if abs(detected_bpm - scalar_tempo) >= 0.05:
        report(
            "Whole-song beat tracking refined the tempo from "
            f"{scalar_tempo:.2f} to {detected_bpm:.2f} BPM."
        )
    bpm = config.bpm if config.bpm is not None else detected_bpm
    if not math.isfinite(bpm) or not 20.0 <= bpm <= 400.0:
        raise ConversionError("BPM must be between 20 and 400.")
    # Downstream musical intervals use the resolved value even when BPM was
    # auto-detected.
    config = replace(config, bpm=bpm)
    ticks_per_second, effective_bpm = _resolve_timing_grid(
        bpm, config, duration_seconds
    )
    if ticks_per_second <= 0 or ticks_per_second > 655.35:
        raise ConversionError("The selected BPM and resolution exceed NBS tempo limits.")
    tick_seconds = 1.0 / ticks_per_second

    # Tick zero is always absolute source time zero.  Anchoring it to the first
    # detected beat used to remove the intro offset from every note and made
    # synchronization depend on a sometimes unstable beat estimate.
    timeline_origin_seconds = 0.0
    tick_count = max(1, math.ceil(duration_seconds / tick_seconds) + 1)
    if tick_count - 1 > MAX_NBS_TICK:
        maximum_minutes = MAX_NBS_TICK / ticks_per_second / 60.0
        raise ConversionError(
            f"The song is too long; this configuration supports about "
            f"{maximum_minutes:.1f} minutes."
        )

    panning_by_tick = _calculate_tick_panning(
        channels,
        sample_rate,
        timeline_origin_seconds,
        tick_seconds,
        tick_count,
        np,
    )
    advance(
        0.79 if config.separation == "demucs" else 0.45,
        "Tempo and panning analysis complete",
    )

    ai_events: dict[str, list[_TimedPitchEvent]] = {}
    if config.separation == "demucs":
        (
            bass_layer,
            accompaniment_layer_offset,
            drum_layer_offset,
            layer_names,
        ) = _demucs_layer_layout(
            use_vocals,
            config.max_chord_notes,
            selected_background_roles,
        )

        if config.transcription == "ai":
            report("Loading the Basic Pitch transcription model...")
            advance(0.80, "Loading the AI transcription model...")
            basic_pitch_model, basic_pitch_predict = _load_basic_pitch_model()
            advance(0.81, "AI transcription model loaded")
            transcription_roles = []
            if use_vocals:
                transcription_roles.append(("vocals", "Vocals"))
            transcription_roles.extend(
                (
                    ("bass", "Bass"),
                    (
                        "other",
                        (
                            "Accompaniment"
                            if use_vocals
                            else "Instrumental accompaniment"
                        ),
                    ),
                    ("other_transient", "Short accompaniment notes"),
                )
            )
            transcription_roles.extend(
                (role, f"{role.title()} background")
                for role in selected_background_roles
            )
            report("Preparing locally normalized weak-phrase model inputs...")
            adaptive_audio_by_role = {
                "bass": bass_audio,
                "other": accompaniment_source,
                **(
                    {"vocals": vocal_audio}
                    if use_vocals and vocal_audio is not None
                    else {}
                ),
                **{
                    role: instrument_stems[role]
                    for role in selected_background_roles
                },
            }
            adaptive_paths = {
                role: _write_adaptive_transcription_stem(
                    transcription_stem_paths[role].parent,
                    audio,
                    sample_rate,
                    role,
                    librosa,
                    np,
                )
                for role, audio in adaptive_audio_by_role.items()
            }
            transcription_jobs = [
                (
                    role,
                    role,
                    display_name,
                    transcription_stem_paths[
                        "other" if role == "other_transient" else role
                    ],
                    transcription_timing_offsets[
                        "other" if role == "other_transient" else role
                    ],
                    False,
                )
                for role, display_name in transcription_roles
            ]
            adaptive_display_names = {
                "vocals": "Quiet vocal phrases",
                "bass": "Quiet bass phrases",
                "other": "Quiet accompaniment phrases",
                "guitar": "Quiet guitar phrases",
                "piano": "Quiet piano phrases",
            }
            for role in adaptive_audio_by_role:
                transcription_jobs.append(
                    (
                        f"{role}_adaptive",
                        "other_transient" if role == "other" else role,
                        adaptive_display_names.get(
                            role, f"Quiet {role} phrases"
                        ),
                        adaptive_paths[role],
                        0.0,
                        True,
                    )
                )
            transcription_stages = [
                (
                    event_key,
                    prediction_role,
                    display_name,
                    stem_path,
                    timing_offset,
                    adaptive_recovery,
                    0.81 + 0.15 * index / len(transcription_jobs),
                    0.81 + 0.15 * (index + 1) / len(transcription_jobs),
                )
                for index, (
                    event_key,
                    prediction_role,
                    display_name,
                    stem_path,
                    timing_offset,
                    adaptive_recovery,
                ) in enumerate(transcription_jobs)
            ]
            for (
                event_key,
                prediction_role,
                display_name,
                stem_path,
                timing_offset,
                adaptive_recovery,
                stage_start,
                stage_end,
            ) in transcription_stages:
                report(f"Transcribing {display_name.lower()} with the neural model...")
                advance(stage_start, f"Transcribing {display_name.lower()}...")

                def update_ai_stage(
                    fraction: float,
                    start: float = stage_start,
                    end: float = stage_end,
                    name: str = display_name,
                ) -> None:
                    advance(
                        start + (end - start) * fraction,
                        f"Transcribing {name.lower()}...",
                    )

                ai_events[event_key] = _predict_timed_pitch_events(
                    stem_path,
                    basic_pitch_model,
                    basic_pitch_predict,
                    role=prediction_role,
                    sensitivity=config.sensitivity,
                    duration_seconds=duration_seconds,
                    timing_offset_seconds=timing_offset,
                    adaptive_recovery=adaptive_recovery,
                    progress_update=update_ai_stage,
                )
                advance(stage_end, f"{display_name} transcription complete")

            report("Validating accompaniment notes against the separated audio...")
            ai_events["other"] = _fuse_accompaniment_events(
                ai_events["other"],
                ai_events.pop("other_transient"),
                accompaniment_source,
                sample_rate,
                config,
                librosa,
                np,
            )
            adaptive_other = _analyze_timed_pitch_events(
                ai_events.pop("other_adaptive"),
                accompaniment_source,
                sample_rate,
                config,
                librosa,
                np,
                role="other_transient",
                reject_unsupported=False,
            )
            ai_events["other"] = _merge_adaptive_recovery_events(
                ai_events["other"],
                adaptive_other,
                duration_seconds,
                config.sensitivity,
                role="other",
            )
            for role in selected_background_roles:
                ai_events[role] = _fuse_accompaniment_events(
                    ai_events[role],
                    (),
                    instrument_stems[role],
                    sample_rate,
                    config,
                    librosa,
                    np,
                    role=role,
                )
                adaptive_role_events = _analyze_timed_pitch_events(
                    ai_events.pop(f"{role}_adaptive"),
                    instrument_stems[role],
                    sample_rate,
                    config,
                    librosa,
                    np,
                    role=role,
                    reject_unsupported=False,
                )
                ai_events[role] = _merge_adaptive_recovery_events(
                    ai_events[role],
                    adaptive_role_events,
                    duration_seconds,
                    config.sensitivity,
                    role=role,
                )

            report("Measuring pitch-specific loudness and physical attacks...")
            if use_vocals:
                ai_events["vocals"] = _analyze_timed_pitch_events(
                    ai_events["vocals"],
                    vocal_audio,
                    sample_rate,
                    config,
                    librosa,
                    np,
                    role="vocals",
                )
                adaptive_vocals = _analyze_timed_pitch_events(
                    ai_events.pop("vocals_adaptive"),
                    vocal_audio,
                    sample_rate,
                    config,
                    librosa,
                    np,
                    role="vocals",
                    reject_unsupported=False,
                )
                ai_events["vocals"] = _merge_adaptive_recovery_events(
                    ai_events["vocals"],
                    adaptive_vocals,
                    duration_seconds,
                    config.sensitivity,
                    role="vocals",
                )
            ai_events["bass"] = _analyze_timed_pitch_events(
                ai_events["bass"],
                bass_audio,
                sample_rate,
                config,
                librosa,
                np,
                role="bass",
            )
            adaptive_bass = _analyze_timed_pitch_events(
                ai_events.pop("bass_adaptive"),
                bass_audio,
                sample_rate,
                config,
                librosa,
                np,
                role="bass",
                reject_unsupported=False,
            )
            ai_events["bass"] = _merge_adaptive_recovery_events(
                ai_events["bass"],
                adaptive_bass,
                duration_seconds,
                config.sensitivity,
                role="bass",
            )
            ai_events["other"] = _coalesce_polyphonic_onsets(
                ai_events["other"],
                maximum_shift_seconds=min(0.0125, 0.5 * tick_seconds),
            )
            for role in selected_background_roles:
                ai_events[role] = _coalesce_polyphonic_onsets(
                    ai_events[role],
                    maximum_shift_seconds=min(0.0125, 0.5 * tick_seconds),
                )
            if selected_background_roles:
                ai_events["other"] = _merge_instrument_background_events(
                    ai_events["other"],
                    {
                        role: ai_events.pop(role)
                        for role in selected_background_roles
                    },
                    duration_seconds,
                )

            vocal_notes = (
                _ai_events_to_nbs(
                    ai_events["vocals"],
                    timeline_origin_seconds,
                    tick_seconds,
                    tick_count,
                    panning_by_tick,
                    config,
                    layer_offset=0,
                    max_notes=1,
                    default_instrument=INSTRUMENTS["flute"],
                    velocity_scale=0.95,
                    monophonic=True,
                )
                if use_vocals
                else []
            )
            bass_notes = _ai_events_to_nbs(
                ai_events["bass"],
                timeline_origin_seconds,
                tick_seconds,
                tick_count,
                panning_by_tick,
                config,
                layer_offset=bass_layer,
                max_notes=1,
                default_instrument=INSTRUMENTS["bass"],
                velocity_scale=0.88,
                monophonic=True,
                prefer_low=True,
            )

            # Basic Pitch and pYIN fail in different ways.  Use the neural path
            # as the authority, then admit conservative pYIN notes only inside
            # phrase-sized neural gaps.  This protects essential lead and bass
            # lines without creating doubled melodies.
            report("Cross-checking essential vocal and bass phrases...")
            coverage_radius = max(1, _round_tick(0.20 / tick_seconds))
            neighbor_gap = max(3, _round_tick(0.75 / tick_seconds))
            if use_vocals:
                fallback_vocal_notes = _extract_monophonic_notes(
                    vocal_audio,
                    sample_rate,
                    timeline_origin_seconds,
                    tick_seconds,
                    tick_count,
                    panning_by_tick,
                    config,
                    layer=0,
                    default_instrument=INSTRUMENTS["flute"],
                    minimum_note="A0",
                    maximum_note="C8",
                    velocity_scale=0.90,
                    librosa=librosa,
                    np=np,
                )
                vocal_notes = _merge_essential_line_notes(
                    vocal_notes,
                    fallback_vocal_notes,
                    coverage_radius_ticks=coverage_radius,
                    neighbor_gap_ticks=neighbor_gap,
                )
            fallback_bass_notes = _extract_monophonic_notes(
                bass_audio,
                sample_rate,
                timeline_origin_seconds,
                tick_seconds,
                tick_count,
                panning_by_tick,
                config,
                layer=bass_layer,
                default_instrument=INSTRUMENTS["bass"],
                minimum_note="A0",
                maximum_note="C8",
                velocity_scale=0.85,
                librosa=librosa,
                np=np,
            )
            bass_notes = _merge_essential_line_notes(
                bass_notes,
                fallback_bass_notes,
                coverage_radius_ticks=coverage_radius,
                neighbor_gap_ticks=neighbor_gap,
            )
        elif config.transcription == "dsp":
            if use_vocals:
                report("Extracting a monophonic vocal line...")
                advance(0.81, "Analyzing vocal pitch...")
                vocal_notes = _extract_monophonic_notes(
                    vocal_audio,
                    sample_rate,
                    timeline_origin_seconds,
                    tick_seconds,
                    tick_count,
                    panning_by_tick,
                    config,
                    layer=0,
                    default_instrument=INSTRUMENTS["flute"],
                    minimum_note="A0",
                    maximum_note="C8",
                    velocity_scale=0.90,
                    librosa=librosa,
                    np=np,
                )
                advance(0.86, "Vocal pitch analysis complete")
            else:
                vocal_notes = []
                advance(0.81, "No vocals: skipped vocal pitch analysis")
            report("Extracting the bass line...")
            advance(0.86 if use_vocals else 0.81, "Analyzing bass pitch...")
            bass_notes = _extract_monophonic_notes(
                bass_audio,
                sample_rate,
                timeline_origin_seconds,
                tick_seconds,
                tick_count,
                panning_by_tick,
                config,
                layer=bass_layer,
                default_instrument=INSTRUMENTS["bass"],
                minimum_note="A0",
                maximum_note="C8",
                velocity_scale=0.85,
                librosa=librosa,
                np=np,
            )
            advance(0.91, "Bass pitch analysis complete")
        else:
            raise ConversionError(f"Unsupported transcription mode: {config.transcription}")
    else:
        vocal_notes = []
        bass_notes = []
        accompaniment_layer_offset = 0
        drum_layer_offset = config.max_chord_notes
        layer_names = [
            f"Pitch {number + 1}" for number in range(config.max_chord_notes)
        ]

    if config.separation == "demucs" and config.transcription == "ai":
        report("Tracking one persistent focus across the complete song...")
        advance(0.96, "Planning the song-level focus and arrangement...")
        accompaniment_events = _remove_overlapping_timed_duplicates(
            ai_events["other"],
            ai_events.get("vocals", []) + ai_events["bass"],
        )
        accompaniment_notes = _ai_events_to_nbs(
            accompaniment_events,
            timeline_origin_seconds,
            tick_seconds,
            tick_count,
            panning_by_tick,
            config,
            layer_offset=accompaniment_layer_offset,
            max_notes=config.max_chord_notes,
            default_instrument=None,
            velocity_scale=0.68,
        )
        minimum_event_count = _minimum_accompaniment_event_count(
            accompaniment_source,
            duration_seconds,
            librosa,
            np,
        )
        globally_sparse = len(accompaniment_events) < minimum_event_count
        if globally_sparse:
            report(
                "The accompaniment is sparse; recovering missing parts with spectral analysis..."
            )
        else:
            report("Checking strong local gaps with independent spectral analysis...")
        advance(0.965, "Checking for missing essential notes...")
        recovery_notes = _extract_cqt_accompaniment_notes(
            accompaniment_source,
            sample_rate,
            timeline_origin_seconds,
            tick_seconds,
            tick_count,
            panning_by_tick,
            config,
            layer_offset=accompaniment_layer_offset,
            librosa=librosa,
            np=np,
            recovery=True,
        )
        recovery_notes = _select_local_accompaniment_recovery(
            accompaniment_notes,
            recovery_notes,
            accompaniment_source,
            sample_rate,
            timeline_origin_seconds,
            tick_seconds,
            tick_count,
            config,
            librosa,
            np,
            globally_sparse=globally_sparse,
        )
        if recovery_notes:
            accompaniment_notes = _merge_accompaniment_notes(
                accompaniment_notes,
                recovery_notes,
                layer_offset=accompaniment_layer_offset,
                max_notes=config.max_chord_notes,
            )
        advance(0.97, "Accompaniment arrangement complete")
    else:
        report("Extracting accompaniment chord candidates...")
        cqt_start = 0.91 if config.separation == "demucs" else 0.45
        advance(cqt_start, "Analyzing accompaniment frequencies...")
        accompaniment_notes = _extract_cqt_accompaniment_notes(
            accompaniment,
            sample_rate,
            timeline_origin_seconds,
            tick_seconds,
            tick_count,
            panning_by_tick,
            config,
            layer_offset=accompaniment_layer_offset,
            librosa=librosa,
            np=np,
        )
        advance(0.97, "Accompaniment pitch analysis complete")
    if config.separation == "demucs":
        accompaniment_notes = _remove_cross_stem_duplicates(
            accompaniment_notes, vocal_notes + bass_notes
        )
    accompaniment_notes = _stabilize_layer_instruments(
        accompaniment_notes,
        layer_offset=accompaniment_layer_offset,
        layer_count=config.max_chord_notes,
        requested_instrument=config.instrument,
    )

    report("Extracting independent kick, snare, and hi-hat events...")
    advance(0.975, "Analyzing percussion...")
    drum_notes = _extract_drum_notes(
        percussive,
        onset_envelope,
        sample_rate,
        timeline_origin_seconds,
        tick_seconds,
        tick_count,
        panning_by_tick,
        config,
        drum_layer_offset,
        0.85 if config.separation == "demucs" else 1.0,
        librosa,
        np,
    )
    advance(0.99, "Percussion analysis complete")

    report("Measuring pitch-specific note dynamics from the source audio...")
    if config.separation == "demucs":
        if vocal_notes:
            vocal_notes = _apply_pitch_loudness(
                vocal_notes,
                vocal_audio,
                sample_rate,
                timeline_origin_seconds,
                tick_seconds,
                tick_count,
                config,
                librosa,
                np,
            )
        bass_notes = _apply_pitch_loudness(
            bass_notes,
            bass_audio,
            sample_rate,
            timeline_origin_seconds,
            tick_seconds,
            tick_count,
            config,
            librosa,
            np,
        )
        dynamics_accompaniment = accompaniment_source
    else:
        dynamics_accompaniment = accompaniment
    accompaniment_by_source: dict[str | None, list[NbsNote]] = defaultdict(list)
    for note in accompaniment_notes:
        accompaniment_by_source[note.source_role].append(note)
    accompaniment_notes = []
    for source_role, source_notes in accompaniment_by_source.items():
        source_audio = (
            instrument_stems.get(source_role, dynamics_accompaniment)
            if source_role is not None
            else dynamics_accompaniment
        )
        accompaniment_notes.extend(
            _apply_pitch_loudness(
                source_notes,
                source_audio,
                sample_rate,
                timeline_origin_seconds,
                tick_seconds,
                tick_count,
                config,
                librosa,
                np,
            )
        )
    advance(0.992, "Source dynamics measured")

    (
        vocal_notes,
        bass_notes,
        accompaniment_notes,
        drum_notes,
    ) = _balance_song_dynamics(
        vocal_notes,
        bass_notes,
        accompaniment_notes,
        drum_notes,
        use_vocals=use_vocals,
        accompaniment_layer_offset=accompaniment_layer_offset,
        drum_layer_offset=drum_layer_offset,
        max_chord_notes=config.max_chord_notes,
        tick_count=tick_count,
        config=config,
    )
    all_notes = sorted(
        vocal_notes + bass_notes + accompaniment_notes + drum_notes,
        key=lambda note: (note.tick, note.layer),
    )
    if not all_notes:
        raise ConversionError(
            "No notes were detected. Increase --sensitivity and try again."
        )

    if config.include_drums:
        layer_names.extend(["Kick", "Snare", "Hi-hat"])

    octave_folded_notes = (
        _validate_minecraft_key_range(all_notes) if config.minecraft_range else 0
    )
    maximum_timing_error_seconds = _validate_source_timing(
        all_notes,
        timeline_origin_seconds,
        ticks_per_second,
    )
    report("Writing the NBS file...")
    advance(0.995, "Writing the NBS file...")
    write_nbs(
        output_path,
        all_notes,
        layer_names,
        ticks_per_second,
        title=title,
        author=author,
        source_name=input_path.name,
        time_signature=config.time_signature,
    )
    advance(1.0, "Conversion complete")

    return ConversionResult(
        output_path=output_path,
        detected_bpm=detected_bpm,
        effective_bpm=effective_bpm,
        ticks_per_second=ticks_per_second,
        duration_seconds=duration_seconds,
        vocal_notes=len(vocal_notes),
        bass_notes=len(bass_notes),
        accompaniment_notes=len(accompaniment_notes),
        drum_notes=len(drum_notes),
        layer_count=len(layer_names),
        timing=config.timing,
        vocal_handling=vocal_handling,
        maximum_timing_error_seconds=maximum_timing_error_seconds,
        minecraft_range=config.minecraft_range,
        octave_folded_notes=octave_folded_notes,
    )


def _bounded_float(minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Enter a number.") from exc
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"Enter a value between {minimum:g} and {maximum:g}."
            )
        return number

    return parse


def _bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Enter an integer.") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"Enter a value between {minimum} and {maximum}."
            )
        return number

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert audio into an arranged Open Note Block Studio v5 file."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="input audio file (prompted when omitted)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output path (default: input name.nbs)"
    )
    parser.add_argument(
        "--bpm",
        type=_bounded_float(20.0, 400.0),
        help="override BPM (default: automatic detection)",
    )
    parser.add_argument(
        "--ticks-per-beat",
        type=int,
        choices=(1, 2, 3, 4, 6, 8),
        default=4,
        help="subdivisions per beat in beat timing mode (default: 4)",
    )
    parser.add_argument(
        "--timing",
        choices=("precise", "minecraft", "beat"),
        default="precise",
        help=(
            "timeline mode: precise preserves attacks at 40 ticks/s; "
            "minecraft uses NBT-compatible 10 ticks/s; beat follows BPM "
            "(default: precise)"
        ),
    )
    parser.add_argument(
        "--max-chord-notes",
        type=_bounded_int(1, 24),
        default=12,
        help=(
            "maximum simultaneous accompaniment voices, including protected "
            "background instruments (default: 12)"
        ),
    )
    parser.add_argument(
        "--sensitivity",
        type=_bounded_float(0.0, 1.0),
        default=0.5,
        help="detection sensitivity from 0.0 to 1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--retrigger-beats",
        type=_bounded_float(0.0, 16.0),
        default=2.0,
        help=(
            "quiet sustained-note continuation interval in beats; "
            "0 preserves source attacks only (default: 2)"
        ),
    )
    parser.add_argument(
        "--instrument",
        choices=("auto", *INSTRUMENTS.keys()),
        default="auto",
        help="instrument for pitched notes (default: auto)",
    )
    parser.add_argument(
        "--separation",
        choices=("demucs", "basic"),
        default="demucs",
        help="source separation: demucs for quality, basic for speed (default: demucs)",
    )
    parser.add_argument(
        "--transcription",
        choices=("ai", "dsp"),
        default="ai",
        help="transcription: ai uses Basic Pitch, dsp uses CQT/pYIN (default: ai)",
    )
    parser.add_argument(
        "--vocals",
        choices=("auto", "on", "off"),
        default="auto",
        help="vocal layer: auto detects vocals; on/off overrides it (default: auto)",
    )
    parser.add_argument(
        "--no-drums", action="store_true", help="do not generate percussion notes"
    )
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument(
        "--minecraft-range",
        dest="minecraft_range",
        action="store_true",
        help=(
            "octave-fold every pitch into Minecraft's playable key range "
            "33..57 (default)"
        ),
    )
    range_group.add_argument(
        "--full-range",
        dest="minecraft_range",
        action="store_false",
        help=(
            "preserve all 88 NBS keys for native NBS playback; Minecraft "
            "importers may discard out-of-range notes"
        ),
    )
    parser.set_defaults(minecraft_range=True)
    parser.add_argument(
        "--time-signature",
        type=int,
        choices=range(2, 9),
        default=4,
        help="time-signature numerator (default: 4)",
    )
    parser.add_argument("--title", help="song title stored in NBS (default: file name)")
    parser.add_argument("--author", default="", help="author stored in NBS")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output file"
    )
    return parser


def _path_from_user_input(value: str) -> Path:
    """Turn a pasted or drag-and-dropped path into an absolute Path."""

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return Path(os.path.expandvars(cleaned)).expanduser().resolve()


def _prompt_for_input_path() -> Path | None:
    print("=" * 58)
    print(" Audio to Open Note Block Studio (NBS) Converter")
    print("=" * 58)
    print("Enter the path of the audio file to convert.")
    print("You can also drag and drop the file into this window.")
    print("The first run may take several minutes while AI models are prepared.")

    while True:
        try:
            value = input("\nAudio file: ").strip()
        except EOFError:
            return None
        if not value:
            print("No input was provided; exiting.")
            return None
        input_path = _path_from_user_input(value)
        if input_path.is_file():
            return input_path
        print(f"File not found: {input_path}")
        print("Check the path and try again.")


def _pause_before_exit() -> None:
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    interactive = args.input is None
    if interactive:
        input_path = _prompt_for_input_path()
        if input_path is None:
            return 0
    else:
        input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_suffix(".nbs")
    )

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")
    if input_path == output_path:
        parser.error("The input and output paths must be different.")
    if output_path.exists() and not args.force:
        if not interactive:
            parser.error(
                f"Output file already exists: {output_path} (use --force to overwrite)"
            )
        try:
            overwrite = input(
                f"\nThe output file already exists. Overwrite it?\n"
                f"{output_path}\n[y/N]: "
            )
        except EOFError:
            overwrite = ""
        if overwrite.strip().lower() not in {"y", "yes"}:
            print("Conversion cancelled.")
            _pause_before_exit()
            return 0

    title = args.title or input_path.stem
    metadata = (title, args.author, input_path.name)
    if any(
        value.encode("cp1252", errors="replace").decode("cp1252") != value
        for value in metadata
    ):
        print(
            "Warning: NBS v5 string limitations require unsupported title, "
            "author, or source-file characters to be replaced with '?'.",
            file=sys.stderr,
        )

    instrument = None if args.instrument == "auto" else INSTRUMENTS[args.instrument]
    config = ConversionConfig(
        bpm=args.bpm,
        ticks_per_beat=args.ticks_per_beat,
        max_chord_notes=args.max_chord_notes,
        sensitivity=args.sensitivity,
        retrigger_beats=args.retrigger_beats,
        instrument=instrument,
        include_drums=not args.no_drums,
        minecraft_range=args.minecraft_range,
        time_signature=args.time_signature,
        separation=args.separation,
        transcription=args.transcription,
        timing=args.timing,
        vocals=args.vocals,
    )

    progress_display = _ConsoleProgress()
    progress_display.update(0.0, "Preparing conversion...", force=True)
    try:
        result = convert_audio_to_nbs(
            input_path,
            output_path,
            config,
            title=title,
            author=args.author,
            progress=progress_display.message,
            progress_update=progress_display.update,
        )
    except ConversionError as exc:
        progress_display.close_line()
        print(f"Error: {exc}", file=sys.stderr)
        if interactive:
            _pause_before_exit()
        return 1
    except KeyboardInterrupt:
        progress_display.close_line()
        print("\nConversion cancelled.", file=sys.stderr)
        if interactive:
            _pause_before_exit()
        return 130
    except Exception:
        progress_display.close_line()
        raise

    progress_display.complete()
    print(f"Conversion complete: {result.output_path}")
    timing_label = (
        "high-resolution NBS"
        if result.timing == "precise"
        else (
            "Minecraft/NBT synchronized"
            if result.timing == "minecraft"
            else "BPM beat grid"
        )
    )
    print(
        f"Song BPM: {result.effective_bpm:.2f} "
        f"(detected: {result.detected_bpm:.2f}) / "
        f"NBS timeline: {result.ticks_per_second:.2f} ticks/s "
        f"({timing_label})"
    )
    print(
        "Maximum verified source-time rounding error: "
        f"{result.maximum_timing_error_seconds * 1000.0:.2f} ms"
    )
    if result.minecraft_range:
        print(
            "Minecraft pitch range: verified keys "
            f"{MINECRAFT_KEY_MIN}..{MINECRAFT_KEY_MAX}; "
            f"{result.octave_folded_notes} notes octave-folded; "
            "0 out-of-range notes"
        )
    else:
        print(
            "Pitch range: full 88-key NBS; Minecraft importers may ignore "
            f"keys outside {MINECRAFT_KEY_MIN}..{MINECRAFT_KEY_MAX}"
        )
    vocal_label = {
        "detected": "vocals detected (dedicated layer)",
        "instrumental": "no vocals (residual merged into accompaniment)",
        "forced_on": "vocals enabled by override",
        "forced_off": "vocals disabled by override",
        "unavailable": "basic separation (no dedicated detection)",
    }.get(result.vocal_handling, result.vocal_handling)
    print(f"Vocal handling: {vocal_label}")
    print(
        f"Notes: {result.total_notes} "
        f"(vocals {result.vocal_notes}, bass {result.bass_notes}, "
        f"accompaniment {result.accompaniment_notes}, drums {result.drum_notes}) / "
        f"Layers: {result.layer_count} / "
        f"Duration: {result.duration_seconds:.1f}s"
    )
    if interactive:
        _pause_before_exit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
