import io
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mp3_to_nbs import (
    MINECRAFT_TIMELINE_TPS,
    PRECISE_TIMELINE_TPS,
    ConversionConfig,
    ConversionError,
    ConversionResult,
    NbsNote,
    _ConsoleProgress,
    _DemucsProgressParser,
    _QuantizedPitchEvent,
    _TimedPitchEvent,
    _VoicedPitchEvent,
    _analyze_timed_pitch_events,
    _apply_audio_loudness,
    _apply_pitch_loudness,
    _ai_events_to_nbs,
    _arrange_polyphonic_events,
    _balance_song_dynamics,
    _classify_vocal_scores,
    _coalesce_polyphonic_onsets,
    _demucs_layer_layout,
    _estimate_stem_delay_seconds,
    _estimate_tempo_and_beats,
    _extract_pitch_events,
    _fuse_accompaniment_events,
    _merge_accompaniment_passes,
    _merge_adaptive_recovery_events,
    _merge_accompaniment_notes,
    _merge_essential_line_notes,
    _merge_instrument_background_events,
    _iter_phase_locked_retrigger_ticks,
    _path_from_user_input,
    _locally_normalize_for_transcription,
    _predict_timed_pitch_events,
    _quantize_timed_pitch_events,
    _refine_tempo_from_beat_times,
    _resolve_tempo_octave,
    _remove_cross_stem_duplicates,
    _remove_overlapping_timed_duplicates,
    _resolve_timing_grid,
    _round_tick,
    _round_tick_array,
    _select_local_accompaniment_recovery,
    _select_accompaniment_focus_path,
    _select_background_stem_roles,
    _select_drum_components,
    _select_monophonic_events,
    _shift_audio_to_timeline,
    _source_cache_key,
    _stabilize_layer_instruments,
    _stable_voice_instruments,
    _validate_minecraft_key_range,
    _validate_source_timing,
    _write_instrumental_accompaniment_stem,
    build_parser,
    choose_instrument,
    fold_key_to_minecraft_range,
    write_nbs,
)


class _NbsReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def _read(self, fmt: str):
        value = struct.unpack_from(fmt, self.data, self.offset)[0]
        self.offset += struct.calcsize(fmt)
        return value

    def u8(self):
        return self._read("<B")

    def u16(self):
        return self._read("<H")

    def i16(self):
        return self._read("<h")

    def u32(self):
        return self._read("<I")

    def string(self):
        size = self.u32()
        value = self.data[self.offset : self.offset + size].decode("cp1252")
        self.offset += size
        return value


class PitchMappingTests(unittest.TestCase):
    def test_drag_and_drop_quotes_are_removed(self):
        expected = Path("music.mp3").resolve()
        self.assertEqual(_path_from_user_input('  "music.mp3"  '), expected)
        self.assertEqual(_path_from_user_input("'music.mp3'"), expected)

    def test_keys_are_folded_by_octaves(self):
        self.assertEqual(fold_key_to_minecraft_range(0), 36)
        self.assertEqual(fold_key_to_minecraft_range(33), 33)
        self.assertEqual(fold_key_to_minecraft_range(57), 57)
        self.assertEqual(fold_key_to_minecraft_range(87), 51)

    def test_auto_instrument_uses_source_octave(self):
        self.assertEqual(choose_instrument(40, None), 1)
        self.assertEqual(choose_instrument(55, None), 5)
        self.assertEqual(choose_instrument(72, None), 0)
        self.assertEqual(choose_instrument(90, None), 6)
        self.assertEqual(choose_instrument(40, 14), 14)

class SeparationTests(unittest.TestCase):
    def test_only_exact_same_instrument_note_is_removed_after_folding(self):
        foreground = [NbsNote(8, 0, 6, 45)]
        background = [
            NbsNote(7, 2, 6, 45),
            NbsNote(8, 2, 6, 57),
            NbsNote(8, 3, 0, 45),
            NbsNote(8, 4, 6, 45),
        ]
        self.assertEqual(
            _remove_cross_stem_duplicates(background, foreground),
            background[:3],
        )

    def test_cache_key_depends_on_audio_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.mp3"
            second = Path(temporary_directory) / "second.mp3"
            first.write_bytes(b"same audio")
            second.write_bytes(b"same audio")
            self.assertEqual(_source_cache_key(first), _source_cache_key(second))
            second.write_bytes(b"different audio")
            self.assertNotEqual(_source_cache_key(first), _source_cache_key(second))

    def test_result_breaks_melodic_notes_down_by_stem(self):
        result = ConversionResult(
            output_path=Path("result.nbs"),
            detected_bpm=120,
            effective_bpm=120,
            ticks_per_second=8,
            duration_seconds=10,
            vocal_notes=3,
            bass_notes=4,
            accompaniment_notes=5,
            drum_notes=6,
            layer_count=8,
        )
        self.assertEqual(result.melodic_notes, 12)
        self.assertEqual(result.total_notes, 18)

    def test_adjacent_same_pitch_attacks_are_not_joined(self):
        events = [
            _TimedPitchEvent(0.00, 0.60, 60, 0.7, onset_strength=0.8),
            _TimedPitchEvent(0.50, 0.90, 60, 0.8, onset_strength=0.9),
        ]
        quantized = _quantize_timed_pitch_events(
            events, 0.0, 0.25, 8, join_gap_ticks=1
        )
        self.assertEqual(
            [(event.start_tick, event.end_tick, event.midi) for event in quantized],
            [(0, 3, 60), (2, 4, 60)],
        )

    def test_only_sub_tick_chord_jitter_is_coalesced(self):
        chord = _coalesce_polyphonic_onsets(
            [
                _TimedPitchEvent(1.046, 1.50, 60, 0.90),
                _TimedPitchEvent(1.054, 1.55, 64, 0.75),
            ]
        )
        audible_strum = _coalesce_polyphonic_onsets(
            [
                _TimedPitchEvent(2.00, 2.20, 60, 0.90),
                _TimedPitchEvent(2.05, 2.25, 64, 0.80),
            ]
        )

        self.assertEqual(chord[0].start_seconds, chord[1].start_seconds)
        self.assertNotEqual(
            audible_strum[0].start_seconds,
            audible_strum[1].start_seconds,
        )

    def test_chord_coalescing_cannot_chain_beyond_its_tolerance(self):
        events = [
            _TimedPitchEvent(0.00, 0.30, 60, 0.60),
            _TimedPitchEvent(0.01, 0.30, 64, 0.70),
            _TimedPitchEvent(0.02, 0.30, 67, 0.90),
        ]

        coalesced = _coalesce_polyphonic_onsets(
            events, tolerance_seconds=0.015
        )
        by_midi = {event.midi: event.start_seconds for event in coalesced}

        self.assertEqual(by_midi[60], by_midi[64])
        self.assertEqual(by_midi[67], 0.02)
        self.assertLessEqual(abs(by_midi[60] - events[0].start_seconds), 0.0125)

    def test_pitch_specific_onset_refinement_finds_one_physical_attack(self):
        import librosa

        sample_rate = 22_050
        times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = np.zeros_like(times)
        active = (times >= 1.0) & (times < 1.5)
        audio[active] = 0.8 * np.sin(
            2.0 * np.pi * 261.6256 * times[active]
        )
        config = ConversionConfig(bpm=120)

        refined = [
            _analyze_timed_pitch_events(
                [_TimedPitchEvent(guess, 1.45, 60, 0.8)],
                audio,
                sample_rate,
                config,
                librosa,
                np,
                role="other",
                reject_unsupported=False,
            )[0]
            for guess in (0.95, 0.97, 1.02, 1.05)
        ]

        self.assertTrue(
            all(abs(event.start_seconds - 1.0) <= 0.01 for event in refined)
        )
        self.assertLessEqual(
            max(event.start_seconds for event in refined)
            - min(event.start_seconds for event in refined),
            0.003,
        )

    def test_onset_refinement_does_not_collapse_rapid_repeated_pitch(self):
        import librosa

        sample_rate = 22_050
        times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = np.zeros_like(times)
        for start in (1.0, 1.075):
            active = (times >= start) & (times < start + 0.055)
            local_time = times[active] - start
            audio[active] += (
                np.exp(-18.0 * local_time)
                * np.sin(2.0 * np.pi * 261.6256 * times[active])
            )

        refined = _analyze_timed_pitch_events(
            [
                _TimedPitchEvent(0.985, 1.055, 60, 0.8),
                _TimedPitchEvent(1.060, 1.140, 60, 0.8),
            ],
            audio,
            sample_rate,
            ConversionConfig(bpm=120),
            librosa,
            np,
            role="piano",
            reject_unsupported=False,
        )

        self.assertEqual(len(refined), 2)
        self.assertGreater(
            refined[1].start_seconds - refined[0].start_seconds, 0.04
        )

    def test_only_near_identical_ai_stem_leakage_is_removed(self):
        foreground = [_TimedPitchEvent(1.0, 2.0, 60, 0.8)]
        background = [
            _TimedPitchEvent(0.5, 1.5, 72, 0.7),
            _TimedPitchEvent(1.2, 1.8, 64, 0.7),
            _TimedPitchEvent(2.0, 2.5, 48, 0.7),
            _TimedPitchEvent(1.02, 1.95, 60, 0.7),
            _TimedPitchEvent(1.0, 1.12, 60, 0.7),
        ]
        self.assertEqual(
            _remove_overlapping_timed_duplicates(background, foreground),
            background[:3] + background[4:],
        )

    def test_independently_supported_instrument_unison_is_not_leakage(self):
        foreground = [_TimedPitchEvent(1.0, 2.0, 60, 0.8)]
        guitar_unison = _TimedPitchEvent(
            1.01, 1.95, 60, 0.7, source_role="guitar"
        )

        self.assertEqual(
            _remove_overlapping_timed_duplicates(
                [guitar_unison], foreground
            ),
            [guitar_unison],
        )

    def test_transient_pass_keeps_short_funk_attack_without_doubling_primary(self):
        primary = [_TimedPitchEvent(1.0, 1.4, 52, 0.75)]
        recovery = [
            _TimedPitchEvent(1.01, 1.39, 52, 0.50),
            _TimedPitchEvent(1.5, 1.59, 55, 0.48),
        ]
        merged = _merge_accompaniment_passes(primary, recovery, 10.0)
        self.assertEqual(merged, [primary[0], recovery[1]])

    def test_audio_evidence_rejects_an_unsupported_neural_pitch(self):
        import librosa

        sample_rate = 22_050
        times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = (0.8 * np.sin(2.0 * np.pi * 261.6256 * times)).astype(np.float32)
        candidates = [
            _TimedPitchEvent(0.25, 1.75, 60, 0.70),
            _TimedPitchEvent(0.25, 1.75, 73, 0.55),
        ]
        fused = _fuse_accompaniment_events(
            candidates,
            [],
            audio,
            sample_rate,
            ConversionConfig(bpm=120),
            librosa,
            np,
        )
        self.assertEqual([event.midi for event in fused], [60])

    def test_transient_ai_pass_accepts_low_guitar_and_55ms_notes(self):
        captured = {}

        def fake_predict(_path, _model, **settings):
            captured.update(settings)
            return None, None, [(0.0, 0.06, 40, 0.5, None)]

        events = _predict_timed_pitch_events(
            Path("unused.wav"),
            object(),
            fake_predict,
            role="other_transient",
            sensitivity=0.5,
        )
        self.assertEqual([event.midi for event in events], [40])
        self.assertLessEqual(captured["minimum_note_length"], 55.0)
        self.assertLessEqual(captured["minimum_frequency"], 83.0)
        self.assertFalse(captured["melodia_trick"])

    def test_neural_pass_keeps_the_complete_nbs_pitch_range(self):
        captured = {}

        def fake_predict(_path, _model, **settings):
            captured.update(settings)
            return None, None, [
                (0.0, 0.04, 21, 0.8, None),
                (0.1, 0.14, 108, 0.8, None),
            ]

        events = _predict_timed_pitch_events(
            Path("unused.wav"),
            object(),
            fake_predict,
            role="other_transient",
            sensitivity=0.5,
        )

        self.assertEqual([event.midi for event in events], [21, 108])
        self.assertLessEqual(captured["minimum_frequency"], 27.5)
        self.assertGreaterEqual(captured["maximum_frequency"], 4186.0)
        self.assertLessEqual(captured["minimum_note_length"], 30.0)

    def test_adaptive_model_pass_uses_more_permissive_neural_gates(self):
        settings_by_pass = []

        def fake_predict(_path, _model, **settings):
            settings_by_pass.append(settings)
            return None, None, []

        for adaptive in (False, True):
            _predict_timed_pitch_events(
                Path("unused.wav"),
                object(),
                fake_predict,
                role="piano",
                sensitivity=0.5,
                adaptive_recovery=adaptive,
            )

        ordinary, adaptive = settings_by_pass
        self.assertLess(adaptive["onset_threshold"], ordinary["onset_threshold"])
        self.assertLess(adaptive["frame_threshold"], ordinary["frame_threshold"])
        self.assertLess(
            adaptive["minimum_note_length"], ordinary["minimum_note_length"]
        )

    def test_local_model_normalization_exposes_quiet_phrase_without_moving_attack(self):
        import librosa

        sample_rate = 8_000
        times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = np.zeros_like(times)
        loud = (times >= 0.10) & (times < 0.75)
        quiet = (times >= 1.20) & (times < 1.85)
        audio[loud] = np.sin(2.0 * np.pi * 220.0 * times[loud])
        audio[quiet] = 0.03 * np.sin(2.0 * np.pi * 220.0 * times[quiet])

        normalized = _locally_normalize_for_transcription(
            audio, sample_rate, librosa, np
        )

        raw_ratio = np.sqrt(np.mean(audio[quiet] ** 2)) / np.sqrt(
            np.mean(audio[loud] ** 2)
        )
        normalized_ratio = np.sqrt(np.mean(normalized[quiet] ** 2)) / np.sqrt(
            np.mean(normalized[loud] ** 2)
        )
        self.assertGreater(normalized_ratio, raw_ratio * 3.0)
        self.assertEqual(
            np.flatnonzero(normalized[quiet] != 0.0)[0],
            np.flatnonzero(audio[quiet] != 0.0)[0],
        )
        self.assertTrue(np.all(normalized[~(loud | quiet)] == 0.0))

    def test_adaptive_recovery_keeps_repeated_quiet_pitch_but_rejects_noise(self):
        primary = [_TimedPitchEvent(0.0, 0.2, 60, 0.65)]
        adaptive = [
            _TimedPitchEvent(
                1.0,
                1.12,
                64,
                0.42,
                model_confidence=0.40,
                pitch_loudness=0.035,
                onset_strength=0.55,
                pitch_snr_db=13.0,
            ),
            _TimedPitchEvent(
                2.0,
                2.12,
                64,
                0.41,
                model_confidence=0.39,
                pitch_loudness=0.030,
                onset_strength=0.50,
                pitch_snr_db=12.0,
            ),
            _TimedPitchEvent(
                3.0,
                3.10,
                81,
                0.40,
                model_confidence=0.70,
                pitch_loudness=0.010,
                onset_strength=0.10,
                pitch_snr_db=0.0,
            ),
        ]

        merged = _merge_adaptive_recovery_events(
            primary,
            adaptive,
            4.0,
            0.5,
            role="other",
        )

        self.assertEqual(
            [(event.start_seconds, event.midi) for event in merged],
            [(0.0, 60), (1.0, 64), (2.0, 64)],
        )

    def test_local_recovery_uses_one_tick_for_an_uncovered_strong_onset(self):
        class FakeLibrosa:
            class onset:
                @staticmethod
                def onset_strength(**_kwargs):
                    return np.array([0.0, 0.1, 1.0, 0.1, 0.0])

                @staticmethod
                def onset_detect(**_kwargs):
                    return np.array([2])

            class feature:
                @staticmethod
                def rms(**_kwargs):
                    return np.ones((1, 10), dtype=np.float32)

            @staticmethod
            def frames_to_time(frames, **_kwargs):
                return np.asarray(frames, dtype=np.float32) * 0.1

        primary = [NbsNote(10, 2, 0, 45)]
        recovery = [
            NbsNote(1, 2, 0, 40),
            NbsNote(2, 2, 0, 42),
            NbsNote(3, 2, 0, 44),
            NbsNote(4, 2, 0, 46),
        ]
        selected = _select_local_accompaniment_recovery(
            primary,
            recovery,
            np.ones(1_000, dtype=np.float32),
            1_000,
            0.0,
            0.1,
            20,
            ConversionConfig(bpm=120),
            FakeLibrosa,
            np,
        )
        self.assertEqual(selected, [replace(recovery[1], layer=3)])

    def test_local_recovery_can_fill_a_missing_tone_in_an_existing_chord(self):
        class FakeLibrosa:
            class onset:
                @staticmethod
                def onset_strength(**_kwargs):
                    return np.array([0.0, 0.1, 1.0, 0.1, 0.0])

                @staticmethod
                def onset_detect(**_kwargs):
                    return np.array([2])

            class feature:
                @staticmethod
                def rms(**_kwargs):
                    return np.ones((1, 10), dtype=np.float32)

            @staticmethod
            def frames_to_time(frames, **_kwargs):
                return np.asarray(frames, dtype=np.float32) * 0.1

        primary = [NbsNote(2, 2, 0, 39, source_midi=60)]
        missing_tone = NbsNote(2, 2, 0, 43, source_midi=64)

        selected = _select_local_accompaniment_recovery(
            primary,
            [missing_tone],
            np.ones(1_000, dtype=np.float32),
            1_000,
            0.0,
            0.1,
            20,
            ConversionConfig(bpm=120),
            FakeLibrosa,
            np,
        )

        self.assertEqual(selected, [replace(missing_tone, layer=3)])

    def test_local_recovery_keeps_independent_physical_onsets(self):
        class DenseFakeLibrosa:
            class onset:
                @staticmethod
                def onset_strength(**_kwargs):
                    return np.ones(60, dtype=np.float32)

                @staticmethod
                def onset_detect(**_kwargs):
                    return np.arange(50)

            class feature:
                @staticmethod
                def rms(**_kwargs):
                    return np.ones((1, 60), dtype=np.float32)

            @staticmethod
            def frames_to_time(frames, **_kwargs):
                return np.asarray(frames, dtype=np.float32) * 0.1

        primary = [NbsNote(100 + tick * 5, 2, 0, 45) for tick in range(10)]
        recovery = [NbsNote(tick, 2, 0, 40 + tick % 12) for tick in range(50)]
        selected = _select_local_accompaniment_recovery(
            primary,
            recovery,
            np.ones(2_000, dtype=np.float32),
            1_000,
            0.0,
            0.1,
            200,
            ConversionConfig(bpm=120),
            DenseFakeLibrosa,
            np,
        )
        self.assertGreater(len(selected), 4)
        self.assertTrue(
            all(
                abs(first.tick - second.tick) > 1
                for first, second in zip(selected, selected[1:])
            )
        )

    def test_default_configuration_is_minecraft_pitch_safe(self):
        config = ConversionConfig()
        self.assertEqual(config.max_chord_notes, 12)
        self.assertTrue(config.minecraft_range)
        self.assertEqual(config.transcription, "ai")
        self.assertEqual(config.retrigger_beats, 2.0)
        self.assertEqual(config.vocals, "auto")
        self.assertEqual(config.timing, "precise")
        self.assertEqual(config.hop_length, 256)

    def test_instrumental_layout_has_no_vocal_layer(self):
        bass, accompaniment, drums, names = _demucs_layer_layout(False, 4)
        self.assertEqual((bass, accompaniment, drums), (0, 1, 5))
        self.assertEqual(
            names,
            [
                "Bass",
                "Accompaniment lead",
                "Accompaniment low anchor",
                "Accompaniment voice 3",
                "Accompaniment voice 4",
            ],
        )

    def test_global_monophonic_path_rejects_an_octave_outlier(self):
        events = [
            _QuantizedPitchEvent(0, 4, 60, 0.72),
            _QuantizedPitchEvent(4, 8, 62, 0.70),
            _QuantizedPitchEvent(4, 8, 74, 0.78),
            _QuantizedPitchEvent(8, 12, 64, 0.71),
        ]
        selected = _select_monophonic_events(events, prefer_low=False)
        self.assertEqual([event.midi for event in selected], [60, 62, 64])

    def test_polyphonic_planner_preserves_lead_and_low_anchor(self):
        events = [
            _QuantizedPitchEvent(0, 4, 47, 0.68),
            _QuantizedPitchEvent(0, 4, 55, 0.88),
            _QuantizedPitchEvent(0, 4, 60, 0.82),
            _QuantizedPitchEvent(4, 8, 49, 0.70),
            _QuantizedPitchEvent(4, 8, 57, 0.86),
            _QuantizedPitchEvent(4, 8, 62, 0.80),
        ]
        arranged = _arrange_polyphonic_events(events, 3)
        by_tick = {}
        for event in arranged:
            by_tick.setdefault(event.start_tick, {})[event.voice] = event.midi
        self.assertEqual(by_tick[0][0], 60)
        self.assertEqual(by_tick[0][1], 47)
        self.assertEqual(by_tick[4][0], 62)
        self.assertEqual(by_tick[4][1], 49)

    def test_polyphonic_planner_keeps_a_dense_supported_chord(self):
        pitches = [36, 48, 55, 60, 64, 67, 72, 76, 79, 84]
        events = [
            _QuantizedPitchEvent(0, 12, midi, 0.40 + index * 0.02)
            for index, midi in enumerate(pitches)
        ]

        arranged = _arrange_polyphonic_events(events, max_voices=12)

        self.assertEqual({event.midi for event in arranged}, set(pitches))
        self.assertEqual(len({event.voice for event in arranged}), len(pitches))

    def test_full_range_output_keeps_distinct_octave_doublings(self):
        notes = _ai_events_to_nbs(
            [
                _TimedPitchEvent(0.0, 0.5, midi, 0.8)
                for midi in (48, 60, 72)
            ],
            0.0,
            0.025,
            40,
            np.zeros(40, dtype=np.int16),
            ConversionConfig(
                bpm=120, max_chord_notes=3, minecraft_range=False
            ),
            layer_offset=0,
            max_notes=3,
            default_instrument=0,
            velocity_scale=1.0,
        )

        self.assertEqual({note.source_midi for note in notes}, {48, 60, 72})
        self.assertEqual(len({note.key for note in notes}), 3)

    def test_default_output_folds_every_source_octave_into_minecraft_range(self):
        notes = _ai_events_to_nbs(
            [
                _TimedPitchEvent(0.0, 0.5, midi, 0.8)
                for midi in (21, 60, 108)
            ],
            0.0,
            0.025,
            40,
            np.zeros(40, dtype=np.int16),
            ConversionConfig(bpm=120, max_chord_notes=3),
            layer_offset=0,
            max_notes=3,
            default_instrument=0,
            velocity_scale=1.0,
        )

        self.assertEqual({note.source_midi for note in notes}, {21, 60, 108})
        self.assertTrue(all(33 <= note.key <= 57 for note in notes))
        self.assertEqual(_validate_minecraft_key_range(notes), 2)

    def test_minecraft_range_invariant_rejects_an_unfolded_note(self):
        with self.assertRaisesRegex(ConversionError, "outside 33\\.\\.57"):
            _validate_minecraft_key_range([NbsNote(0, 0, 0, 58)])

    def test_command_line_uses_minecraft_range_unless_full_range_is_explicit(self):
        parser = build_parser()

        self.assertTrue(parser.parse_args(["song.mp3"]).minecraft_range)
        self.assertTrue(
            parser.parse_args(["song.mp3", "--minecraft-range"]).minecraft_range
        )
        self.assertFalse(
            parser.parse_args(["song.mp3", "--full-range"]).minecraft_range
        )

    def test_polyphonic_planner_does_not_drop_a_quiet_continuing_focus(self):
        events = [
            _QuantizedPitchEvent(0, 1, 60, 0.82),
            _QuantizedPitchEvent(1, 2, 62, 0.34),
            _QuantizedPitchEvent(1, 2, 74, 0.92),
            _QuantizedPitchEvent(2, 3, 64, 0.35),
            _QuantizedPitchEvent(2, 3, 76, 0.88),
            _QuantizedPitchEvent(3, 4, 65, 0.38),
            _QuantizedPitchEvent(3, 4, 77, 0.84),
            _QuantizedPitchEvent(4, 5, 67, 0.80),
        ]

        arranged = _arrange_polyphonic_events(events, 3)
        focus = {
            event.start_tick: event.midi for event in arranged if event.voice == 0
        }

        self.assertEqual(focus, {0: 60, 1: 62, 2: 64, 3: 65, 4: 67})

    def test_polyphonic_planner_reserves_a_background_instrument_lane(self):
        events = [
            _QuantizedPitchEvent(0, 4, 47, 0.68),
            _QuantizedPitchEvent(0, 4, 60, 0.82),
            _QuantizedPitchEvent(0, 4, 64, 0.70),
            _QuantizedPitchEvent(0, 3, 57, 0.66, source_role="guitar"),
            _QuantizedPitchEvent(4, 8, 49, 0.70),
            _QuantizedPitchEvent(4, 8, 62, 0.80),
            _QuantizedPitchEvent(4, 8, 65, 0.69),
            _QuantizedPitchEvent(4, 7, 59, 0.65, source_role="guitar"),
        ]
        arranged = _arrange_polyphonic_events(events, 4)
        by_tick = {}
        for event in arranged:
            by_tick.setdefault(event.start_tick, {})[event.voice] = event
        self.assertEqual(by_tick[0][0].midi, 60)
        self.assertEqual(by_tick[0][1].midi, 47)
        self.assertEqual(by_tick[0][3].midi, 57)
        self.assertEqual(by_tick[4][3].midi, 59)
        self.assertEqual(
            _stable_voice_instruments(arranged, None, None)[3], 5
        )

    def test_polyphonic_planner_preserves_an_isolated_background_chord(self):
        events = [
            _QuantizedPitchEvent(0, 8, 60, 0.82),
            _QuantizedPitchEvent(0, 8, 64, 0.76),
            _QuantizedPitchEvent(0, 8, 67, 0.72),
            _QuantizedPitchEvent(0, 8, 48, 0.68, source_role="piano"),
            _QuantizedPitchEvent(0, 8, 52, 0.66, source_role="piano"),
            _QuantizedPitchEvent(0, 8, 55, 0.64, source_role="piano"),
        ]

        arranged = _arrange_polyphonic_events(events, 9)
        piano_events = [
            event for event in arranged if event.source_role == "piano"
        ]

        self.assertEqual({event.midi for event in piano_events}, {48, 52, 55})
        self.assertEqual(len({event.voice for event in piano_events}), 3)
        instruments = _stable_voice_instruments(arranged, None, None)
        self.assertTrue(
            all(instruments[event.voice] == 0 for event in piano_events)
        )

    def test_background_merge_requires_recurrence_and_avoids_doubles(self):
        primary = [_TimedPitchEvent(1.0, 1.4, 60, 0.72)]
        guitar = [
            _TimedPitchEvent(1.03, 1.35, 60, 0.76, "guitar"),
            _TimedPitchEvent(2.0, 2.3, 64, 0.68, "guitar"),
            _TimedPitchEvent(8.0, 8.3, 64, 0.67, "guitar"),
            _TimedPitchEvent(14.0, 14.3, 64, 0.69, "guitar"),
            _TimedPitchEvent(16.0, 16.2, 67, 0.60, "guitar"),
        ]
        merged = _merge_instrument_background_events(
            primary, {"guitar": guitar}, 20.0
        )
        self.assertEqual(
            [(event.start_seconds, event.midi) for event in merged],
            [(1.0, 60), (2.0, 64), (8.0, 64), (14.0, 64)],
        )
        self.assertTrue(all(event.source_role == "guitar" for event in merged[1:]))

    def test_clear_one_off_quiet_instrument_note_survives_background_merge(self):
        quiet_piano = _TimedPitchEvent(
            3.0,
            3.08,
            72,
            0.38,
            "piano",
            model_confidence=0.42,
            pitch_loudness=0.04,
            onset_strength=0.55,
            pitch_snr_db=15.0,
        )

        merged = _merge_instrument_background_events(
            [], {"piano": [quiet_piano]}, 10.0
        )

        self.assertEqual(merged, [quiet_piano])

    def test_background_stem_selection_rejects_weak_separator_bleed(self):
        reference = np.ones(20_000, dtype=np.float32)
        stems = {
            "guitar": np.full(20_000, 0.20, dtype=np.float32),
            "piano": np.full(20_000, 0.01, dtype=np.float32),
        }
        self.assertEqual(
            _select_background_stem_roles(
                stems, reference, np, maximum_roles=2
            ),
            ["guitar"],
        )

    def test_spectral_recovery_cannot_invade_a_background_instrument_lane(self):
        primary = [
            NbsNote(0, 4, 5, 45, 60, source_role="guitar")
        ]
        recovery = [NbsNote(8, 4, 0, 48, 70)]
        merged = _merge_accompaniment_notes(
            primary,
            recovery,
            layer_offset=1,
            max_notes=4,
        )
        self.assertEqual([(note.tick, note.layer) for note in merged], [(0, 4), (8, 1)])

    def test_song_focus_ignores_one_off_register_distraction(self):
        events = [
            (0, [_QuantizedPitchEvent(0, 2, 60, 0.70)]),
            (1, [_QuantizedPitchEvent(1, 2, 72, 0.95)]),
            (2, [_QuantizedPitchEvent(2, 4, 60, 0.70)]),
        ]
        focus = _select_accompaniment_focus_path(events)
        self.assertEqual({tick: event.midi for tick, event in focus.items()}, {0: 60, 2: 60})

    def test_song_focus_accepts_a_sustained_section_handoff(self):
        events = [
            (0, [_QuantizedPitchEvent(0, 1, 60, 0.70)]),
            (1, [_QuantizedPitchEvent(1, 2, 60, 0.70)]),
            (2, [_QuantizedPitchEvent(2, 3, 72, 0.80)]),
            (3, [_QuantizedPitchEvent(3, 4, 72, 0.80)]),
            (4, [_QuantizedPitchEvent(4, 5, 72, 0.80)]),
        ]
        focus = _select_accompaniment_focus_path(events)
        self.assertEqual([focus[tick].midi for tick in sorted(focus)], [60, 60, 72, 72, 72])

    def test_each_arranged_voice_uses_one_whole_song_instrument(self):
        events = [
            _VoicedPitchEvent(0, 4, 55, 0.8, 0),
            _VoicedPitchEvent(4, 8, 72, 0.6, 0),
            _VoicedPitchEvent(8, 12, 55, 0.8, 0),
        ]
        instruments = _stable_voice_instruments(events, None, None)
        self.assertEqual(instruments, {0: 5})

    def test_recovery_cannot_change_a_layers_dominant_timbre(self):
        notes = [
            NbsNote(0, 2, 5, 45, 80),
            NbsNote(4, 2, 5, 47, 80),
            NbsNote(8, 2, 0, 49, 30),
            NbsNote(0, 3, 6, 40, 80),
            NbsNote(0, 4, 6, 52, 80),
        ]
        stable = _stabilize_layer_instruments(
            notes,
            layer_offset=2,
            layer_count=3,
            requested_instrument=None,
        )
        by_layer = {
            layer: {note.instrument for note in stable if note.layer == layer}
            for layer in (2, 3, 4)
        }
        self.assertEqual(by_layer, {2: {5}, 3: {5}, 4: {0}})

    def test_vocal_phrase_remains_louder_than_supporting_roles(self):
        config = ConversionConfig(bpm=120)
        vocals, bass, accompaniment, drums = _balance_song_dynamics(
            [NbsNote(5, 0, 6, 45, 40)],
            [NbsNote(5, 1, 1, 40, 100)],
            [
                NbsNote(5, 2, 0, 45, 100),
                NbsNote(5, 3, 5, 40, 100),
            ],
            [NbsNote(5, 5, 2, 45, 100)],
            use_vocals=True,
            accompaniment_layer_offset=2,
            drum_layer_offset=5,
            max_chord_notes=3,
            tick_count=20,
            config=config,
        )
        self.assertGreater(vocals[0].velocity, bass[0].velocity)
        self.assertGreater(vocals[0].velocity, accompaniment[0].velocity)
        self.assertGreater(vocals[0].velocity, drums[0].velocity)

    def test_sustain_continuation_stays_audible_but_below_its_attack(self):
        _, bass, _, _ = _balance_song_dynamics(
            [],
            [
                NbsNote(0, 0, 1, 40, 80),
                NbsNote(4, 0, 1, 40, 80, continuation=True),
            ],
            [],
            [],
            use_vocals=False,
            accompaniment_layer_offset=1,
            drum_layer_offset=4,
            max_chord_notes=3,
            tick_count=10,
            config=ConversionConfig(bpm=135.0),
        )

        self.assertGreater(bass[1].velocity, 1)
        self.assertLessEqual(bass[1].velocity, round(bass[0].velocity * 0.58))
        self.assertTrue(bass[1].continuation)

    def test_measured_quiet_note_is_not_raised_to_a_role_volume_floor(self):
        _, bass, accompaniment, _ = _balance_song_dynamics(
            [],
            [NbsNote(0, 0, 1, 40, 3, audio_dynamic=True)],
            [NbsNote(0, 1, 0, 45, 4, audio_dynamic=True)],
            [],
            use_vocals=False,
            accompaniment_layer_offset=1,
            drum_layer_offset=4,
            max_chord_notes=3,
            tick_count=10,
            config=ConversionConfig(bpm=120),
        )

        self.assertLessEqual(bass[0].velocity, 5)
        self.assertLessEqual(accompaniment[0].velocity, 6)

    def test_note_velocity_follows_measured_source_loudness(self):
        audio = np.concatenate(
            (
                np.full(500, 0.05, dtype=np.float32),
                np.full(500, 0.80, dtype=np.float32),
            )
        )
        notes = _apply_audio_loudness(
            [NbsNote(2, 0, 0, 45, 80), NbsNote(7, 0, 0, 47, 80)],
            audio,
            1_000,
            0.0,
            0.1,
            10,
            np,
        )

        self.assertLess(notes[0].velocity, notes[1].velocity)
        self.assertTrue(all(note.audio_dynamic for note in notes))

    def test_pitch_loudness_does_not_promote_an_unsupported_simultaneous_note(self):
        import librosa

        sample_rate = 22_050
        times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = np.zeros_like(times)
        active = (times >= 0.8) & (times < 1.2)
        audio[active] = 0.8 * np.sin(
            2.0 * np.pi * 261.6256 * times[active]
        )
        notes = [
            NbsNote(
                10,
                0,
                0,
                39,
                100,
                source_midi=60,
                source_time_seconds=1.0,
            ),
            NbsNote(
                10,
                1,
                0,
                45,
                10,
                source_midi=66,
                source_time_seconds=1.0,
            ),
        ]

        measured = _apply_pitch_loudness(
            notes,
            audio,
            sample_rate,
            0.0,
            0.1,
            20,
            ConversionConfig(bpm=120),
            librosa,
            np,
        )

        self.assertEqual([note.source_midi for note in measured], [60])

    def test_pitch_loudness_retains_real_quiet_to_loud_dynamics(self):
        import librosa

        sample_rate = 22_050
        times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = np.zeros_like(times)
        quiet = (times >= 0.45) & (times < 0.70)
        loud = (times >= 1.45) & (times < 1.70)
        audio[quiet] = 0.05 * np.sin(2.0 * np.pi * 261.6256 * times[quiet])
        audio[loud] = 0.80 * np.sin(2.0 * np.pi * 261.6256 * times[loud])
        notes = [
            NbsNote(
                5, 0, 0, 39, 80, source_midi=60, source_time_seconds=0.5
            ),
            NbsNote(
                15, 0, 0, 39, 80, source_midi=60, source_time_seconds=1.5
            ),
        ]

        measured = _apply_pitch_loudness(
            notes,
            audio,
            sample_rate,
            0.0,
            0.1,
            20,
            ConversionConfig(bpm=120),
            librosa,
            np,
        )

        self.assertEqual(len(measured), 2)
        self.assertLess(measured[0].velocity, measured[1].velocity)
        self.assertLess(measured[0].source_loudness, measured[1].source_loudness)

    def test_continuation_stops_after_the_source_has_faded(self):
        audio = np.concatenate(
            (
                np.full(300, 0.50, dtype=np.float32),
                np.zeros(700, dtype=np.float32),
            )
        )
        notes = _apply_audio_loudness(
            [
                NbsNote(1, 0, 0, 45, 80),
                NbsNote(7, 0, 0, 45, 40, continuation=True),
            ],
            audio,
            1_000,
            0.0,
            0.1,
            10,
            np,
        )

        self.assertEqual([note.tick for note in notes], [1])

    def test_line_fallback_only_fills_real_neural_gaps(self):
        primary = [
            NbsNote(0, 0, 6, 45, 80),
            NbsNote(20, 0, 6, 47, 80),
        ]
        fallback = [
            NbsNote(1, 0, 6, 45, 90),
            NbsNote(8, 0, 6, 46, 55),
            NbsNote(12, 0, 6, 47, 55),
            NbsNote(30, 0, 6, 48, 50),
        ]
        merged = _merge_essential_line_notes(
            primary,
            fallback,
            coverage_radius_ticks=2,
            neighbor_gap_ticks=5,
        )
        self.assertEqual([note.tick for note in merged], [0, 8, 12, 20])

    def test_drum_component_selection_keeps_simultaneous_supported_hits(self):
        self.assertEqual(_select_drum_components([0.9, 0.2, 0.7], 0.5), [0, 2])
        self.assertEqual(
            _select_drum_components([0.90, 0.82, 0.78], 0.5),
            [0, 1, 2],
        )
        self.assertEqual(_select_drum_components([0.1, 0.12, 0.08], 0.5), [])

    def test_sustained_voice_is_kept_but_instrument_leakage_is_rejected(self):
        vocal_scores = np.zeros((100, 521), dtype=np.float32)
        vocal_scores[20:50, 24] = 0.8
        vocal_scores[:, 135] = 0.1
        detected = _classify_vocal_scores(
            vocal_scores,
            relative_loudness_db=-12.0,
            energy_active_ratio=0.5,
            np=np,
        )
        self.assertTrue(detected.present)
        self.assertGreaterEqual(detected.longest_vocal_seconds, 10.0)

        instrumental_scores = np.zeros((100, 521), dtype=np.float32)
        instrumental_scores[:, 135] = 0.9
        instrumental_scores[:, 24] = 0.2
        rejected = _classify_vocal_scores(
            instrumental_scores,
            relative_loudness_db=-8.0,
            energy_active_ratio=0.8,
            np=np,
        )
        self.assertFalse(rejected.present)

    def test_short_vocal_like_leak_does_not_create_a_vocal_layer(self):
        scores = np.zeros((100, 521), dtype=np.float32)
        scores[40:45, 0] = 0.95
        detection = _classify_vocal_scores(
            scores,
            relative_loudness_db=-10.0,
            energy_active_ratio=0.5,
            np=np,
        )
        self.assertFalse(detection.present)

    def test_instrumental_accompaniment_cache_preserves_aligned_audio(self):
        import soundfile as sf

        audio = np.linspace(-0.5, 0.5, 2_000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = _write_instrumental_accompaniment_stem(
                Path(temporary_directory), audio, 1_000, 0.5, np
            )
            restored, sample_rate = sf.read(path, dtype="float32")
        self.assertEqual(sample_rate, 1_000)
        np.testing.assert_allclose(restored, audio * 0.5, atol=1e-7)


class ProgressTests(unittest.TestCase):
    def test_demucs_progress_combines_passes_and_ignores_duplicate_100(self):
        parser = _DemucsProgressParser(expected_passes=40)
        self.assertAlmostEqual(
            parser.feed(" 50%|bar| 10/20 [00:01, 5.0seconds/s]"),
            0.5 / 40,
        )
        self.assertAlmostEqual(
            parser.feed("100%|bar| 20/20 [00:02, 5.0seconds/s]"),
            1.0 / 40,
        )
        self.assertEqual(parser.displayed_pass, 1)
        self.assertAlmostEqual(
            parser.feed("100%|bar| 20/20 [00:02, 5.0seconds/s]"),
            1.0 / 40,
        )
        self.assertAlmostEqual(
            parser.feed(" 25%|bar| 5/20 [00:01, 5.0seconds/s]"),
            1.25 / 40,
        )
        self.assertEqual(parser.completed_passes, 1)
        self.assertEqual(parser.active_pass, 2)
        self.assertEqual(parser.displayed_pass, 2)

    def test_console_progress_shows_exact_percent_and_eta_within_80_columns(self):
        stream = io.StringIO()
        display = _ConsoleProgress(stream)
        display.started_at -= 10.0
        display.update(0.25, "A very long stage name that must be truncated", force=True)
        line = stream.getvalue().strip()
        self.assertIn("25.00%", line)
        self.assertRegex(line, r"ETA about 3[01]s")
        self.assertLessEqual(display._display_width(line), 80)


class TimingTests(unittest.TestCase):
    def test_default_grid_preserves_short_native_nbs_attacks(self):
        config = ConversionConfig()
        ticks_per_second, displayed_bpm = _resolve_timing_grid(136.0, config)
        self.assertEqual(config.timing, "precise")
        self.assertEqual(ticks_per_second, PRECISE_TIMELINE_TPS)
        self.assertEqual(displayed_bpm, 136.0)

    def test_explicit_minecraft_grid_remains_nbt_compatible(self):
        config = ConversionConfig(timing="minecraft")
        ticks_per_second, displayed_bpm = _resolve_timing_grid(136.0, config)
        self.assertEqual(ticks_per_second, MINECRAFT_TIMELINE_TPS)
        self.assertEqual(displayed_bpm, 136.0)

    def test_precise_grid_adapts_resolution_instead_of_rejecting_long_audio(self):
        duration_seconds = 60.0 * 60.0
        ticks_per_second, displayed_bpm = _resolve_timing_grid(
            120.0,
            ConversionConfig(),
            duration_seconds,
        )

        self.assertLess(ticks_per_second, PRECISE_TIMELINE_TPS)
        self.assertLessEqual(
            int(np.ceil(duration_seconds * ticks_per_second)), 65_535
        )
        self.assertEqual(displayed_bpm, 120.0)

    def test_whole_song_beats_remove_frame_quantization_tempo_error(self):
        frame_seconds = 512 / 22_050
        ideal_times = 2.0 + np.arange(300) * (60.0 / 135.0)
        tracked_times = np.floor(ideal_times / frame_seconds + 0.5) * frame_seconds
        tracked_times = np.delete(tracked_times, [50, 121, 249])

        refined = _refine_tempo_from_beat_times(136.0, tracked_times, np)

        self.assertAlmostEqual(refined, 135.0, delta=0.05)

    def test_percussion_mix_consensus_recovers_click_track_tempo(self):
        import librosa

        sample_rate = 8_000
        expected_bpm = 135.0
        duration_seconds = 20.0
        click_times = np.arange(
            0.5, duration_seconds, 60.0 / expected_bpm
        )
        clicks = librosa.clicks(
            times=click_times,
            sr=sample_rate,
            length=round(sample_rate * duration_seconds),
            click_duration=0.015,
        ).astype(np.float32)

        _raw_bpm, refined_bpm, beat_times = _estimate_tempo_and_beats(
            clicks,
            clicks,
            sample_rate,
            128,
            librosa,
            np,
        )

        self.assertAlmostEqual(refined_bpm, expected_bpm, delta=0.05)
        self.assertGreater(len(beat_times), 30)

    def test_absolute_time_does_not_drift_on_minecraft_grid(self):
        source_time = 100.04
        events = [_TimedPitchEvent(source_time, source_time + 0.2, 60, 0.8)]
        quantized = _quantize_timed_pitch_events(events, 0.0, 0.1, 1_100)
        playback_time = quantized[0].start_tick / MINECRAFT_TIMELINE_TPS
        self.assertLessEqual(abs(playback_time - source_time), 0.05)

    def test_precise_grid_keeps_rapid_source_attacks_distinct(self):
        events = [
            _TimedPitchEvent(0.049, 0.070, 60, 0.8),
            _TimedPitchEvent(0.099, 0.120, 60, 0.8),
        ]
        quantized = _quantize_timed_pitch_events(
            events,
            0.0,
            1.0 / PRECISE_TIMELINE_TPS,
            20,
        )
        self.assertEqual([event.start_tick for event in quantized], [2, 4])
        self.assertTrue(
            all(
                abs(event.start_tick / PRECISE_TIMELINE_TPS - source.start_seconds)
                <= 0.5 / PRECISE_TIMELINE_TPS
                for event, source in zip(quantized, events)
            )
        )

    def test_half_tempo_extreme_is_doubled_only_with_pulse_support(self):
        envelope = np.zeros(600, dtype=np.float32)
        envelope[::50] = 1.0
        self.assertEqual(
            _resolve_tempo_octave(60.0, envelope, 100, 1, np),
            120.0,
        )
        self.assertEqual(
            _resolve_tempo_octave(90.0, envelope, 100, 1, np),
            90.0,
        )

    def test_source_timing_invariant_rejects_more_than_half_tick_error(self):
        self.assertLess(
            _validate_source_timing(
                [NbsNote(4, 0, 0, 45, source_time_seconds=0.101)],
                0.0,
                PRECISE_TIMELINE_TPS,
            ),
            0.002,
        )
        with self.assertRaises(RuntimeError):
            _validate_source_timing(
                [NbsNote(4, 0, 0, 45, source_time_seconds=0.15)],
                0.0,
                PRECISE_TIMELINE_TPS,
            )

    def test_quantization_retains_the_unrounded_canonical_source_time(self):
        event = _TimedPitchEvent(0.149, 0.40, 60, 0.8)

        quantized = _quantize_timed_pitch_events([event], 0.0, 0.1, 10)

        self.assertEqual(quantized[0].start_tick, 1)
        self.assertEqual(quantized[0].source_time_seconds, 0.149)

    def test_half_tick_rounding_is_identical_for_scalar_and_frame_paths(self):
        values = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])
        expected = np.array([-2, -1, 1, 2, 3])
        np.testing.assert_array_equal(_round_tick_array(values, np), expected)
        self.assertEqual([_round_tick(float(value)) for value in values], expected.tolist())

    def test_fractional_beat_retriggers_remain_phase_locked_for_a_long_song(self):
        config = ConversionConfig(
            bpm=136.0, retrigger_beats=1.0, timing="minecraft"
        )
        interval = MINECRAFT_TIMELINE_TPS * 60.0 / 136.0
        retriggers = list(_iter_phase_locked_retrigger_ticks(0, 500, config))

        for repeat_index, tick in enumerate(retriggers, start=1):
            exact_position = repeat_index * interval
            self.assertEqual(tick, _round_tick(exact_position))
            self.assertLessEqual(abs(tick - exact_position), 0.5)

        self.assertGreater(len(retriggers), 100)
        self.assertEqual(retriggers[99], _round_tick(100 * interval))
        self.assertNotEqual(retriggers[99], 100 * round(interval))

    def test_spectral_retriggers_use_the_same_fractional_clock(self):
        config = ConversionConfig(
            bpm=136.0, retrigger_beats=1.0, timing="minecraft"
        )
        tick_db = np.full((88, 500), -80.0, dtype=np.float32)
        tick_db[39, :] = 0.0
        events = _extract_pitch_events(
            tick_db,
            config,
            np.zeros(500, dtype=np.int16),
            np,
        )

        expected = [0, *_iter_phase_locked_retrigger_ticks(0, 500, config)]
        self.assertEqual([event.tick for event in events], expected)

    def test_ai_retriggers_preserve_the_sub_tick_source_phase(self):
        config = ConversionConfig(
            bpm=136.0, retrigger_beats=1.0, timing="minecraft"
        )
        event = _TimedPitchEvent(0.049, 2.0, 60, 0.8)
        notes = _ai_events_to_nbs(
            [event],
            0.0,
            0.1,
            30,
            np.zeros(30, dtype=np.int16),
            config,
            layer_offset=0,
            max_notes=1,
            default_instrument=0,
            velocity_scale=1.0,
            monophonic=True,
        )

        interval = MINECRAFT_TIMELINE_TPS * 60.0 / 136.0
        expected = [
            _round_tick(0.49 + repeat_index * interval)
            for repeat_index in range(5)
        ]
        self.assertEqual([note.tick for note in notes], expected)
        self.assertFalse(notes[0].continuation)
        self.assertTrue(all(note.continuation for note in notes[1:]))
        self.assertTrue(
            all(note.velocity < notes[0].velocity for note in notes[1:])
        )
        expected_source_times = [
            0.049 + repeat_index * 60.0 / 136.0
            for repeat_index in range(5)
        ]
        np.testing.assert_allclose(
            [note.source_time_seconds for note in notes],
            expected_source_times,
            atol=1e-9,
        )

    def test_retrigger_zero_preserves_source_attacks_only(self):
        notes = _ai_events_to_nbs(
            [_TimedPitchEvent(0.0, 5.0, 60, 0.8)],
            0.0,
            0.1,
            60,
            np.zeros(60, dtype=np.int16),
            ConversionConfig(bpm=135.0, retrigger_beats=0.0),
            layer_offset=0,
            max_notes=1,
            default_instrument=0,
            velocity_scale=1.0,
            monophonic=True,
        )

        self.assertEqual([note.tick for note in notes], [0])

    def test_ai_retrigger_never_crosses_the_exact_source_release(self):
        config = ConversionConfig(bpm=136.0, retrigger_beats=1.0)
        notes = _ai_events_to_nbs(
            [_TimedPitchEvent(0.0, 0.43, 60, 0.8)],
            0.0,
            0.1,
            10,
            np.zeros(10, dtype=np.int16),
            config,
            layer_offset=0,
            max_notes=1,
            default_instrument=0,
            velocity_scale=1.0,
            monophonic=True,
        )

        self.assertEqual([note.tick for note in notes], [0])

    def test_common_stem_decoder_delay_is_measured_and_removed(self):
        sample_rate = 1_000
        generator = np.random.default_rng(42)
        source = generator.standard_normal(5_000).astype(np.float32)
        delay_samples = 23
        delayed = np.zeros_like(source)
        delayed[delay_samples:] = source[:-delay_samples]

        measured = _estimate_stem_delay_seconds(
            source, [delayed], sample_rate, np
        )
        aligned = _shift_audio_to_timeline(delayed, measured, sample_rate, np)

        self.assertAlmostEqual(measured, delay_samples / sample_rate, places=6)
        np.testing.assert_allclose(
            aligned[:-delay_samples], source[:-delay_samples], atol=1e-6
        )


class NbsWriterTests(unittest.TestCase):
    def test_writes_parseable_v5_notes_and_metadata(self):
        notes = [
            NbsNote(0, 0, 0, 45, 91, -25),
            NbsNote(4, 1, 3, 45, 72, 30),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.nbs"
            write_nbs(
                output,
                notes,
                ["Lead", "Snare"],
                8.0,
                title="Test",
                author="Author",
                source_name="input.mp3",
            )
            reader = _NbsReader(output.read_bytes())

        self.assertEqual(reader.u16(), 0)
        self.assertEqual(reader.u8(), 5)
        self.assertEqual(reader.u8(), 16)
        self.assertEqual(reader.u16(), 4)
        self.assertEqual(reader.u16(), 2)
        self.assertEqual(reader.string(), "Test")
        self.assertEqual(reader.string(), "Author")
        self.assertEqual(reader.string(), "")
        self.assertIn("approximate", reader.string())
        self.assertEqual(reader.u16(), 800)
        self.assertEqual([reader.u8(), reader.u8(), reader.u8()], [0, 10, 4])
        self.assertEqual([reader.u32() for _ in range(5)], [0, 0, 0, 2, 0])
        self.assertEqual(reader.string(), "input.mp3")
        self.assertEqual([reader.u8(), reader.u8(), reader.u16()], [0, 0, 0])

        self.assertEqual(reader.u16(), 1)
        self.assertEqual(reader.u16(), 1)
        self.assertEqual(
            [reader.u8(), reader.u8(), reader.u8(), reader.u8(), reader.i16()],
            [0, 45, 91, 75, 0],
        )
        self.assertEqual(reader.u16(), 0)
        self.assertEqual(reader.u16(), 4)
        self.assertEqual(reader.u16(), 2)
        self.assertEqual(
            [reader.u8(), reader.u8(), reader.u8(), reader.u8(), reader.i16()],
            [3, 45, 72, 130, 0],
        )
        self.assertEqual(reader.u16(), 0)
        self.assertEqual(reader.u16(), 0)

        self.assertEqual(reader.string(), "Lead")
        self.assertEqual([reader.u8(), reader.u8(), reader.u8()], [0, 100, 100])
        self.assertEqual(reader.string(), "Snare")
        self.assertEqual([reader.u8(), reader.u8(), reader.u8()], [0, 100, 100])
        self.assertEqual(reader.u8(), 0)
        self.assertEqual(reader.offset, len(reader.data))


if __name__ == "__main__":
    unittest.main()
