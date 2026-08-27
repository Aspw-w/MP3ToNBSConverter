# MP3 to NBS

This project converts mixed audio into an arranged Open Note Block Studio
`.nbs` v5 song. The default output prioritizes timing and transcription quality:
important melody and bass phrases are protected, locally quiet passages receive
an independent recovery pass, weak false notes are rejected, and accompaniment
density is controlled before the file is written. An explicit 10 ticks/s mode
remains available for Minecraft schematic and structure export.

## Conversion architecture

The default high-quality pipeline uses several independent analyses instead of
trusting a single threshold or model:

1. [Demucs](https://pypi.org/project/demucs/) `htdemucs_ft` separates vocals,
   bass, drums, and accompaniment. The four-model ensemble runs with ten random
   shifts and averages the results. A lightweight `htdemucs_6s` cue pass also
   isolates piano and guitar, so a quieter instrument is not masked by the
   combined accompaniment stem.
2. Waveform correlation measures and removes any common decoder or separator
   delay, keeping every stem on the source timeline.
3. A pinned, checksum-verified
   [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
   model decides whether sustained vocals are present. Instrumental leakage in
   the vocal stem is merged back into the accompaniment.
4. [Basic Pitch](https://github.com/spotify/basic-pitch) transcribes vocals,
   bass, accompaniment, and a second pass specialized for short attacks. Every
   active melodic stem also receives a locally normalized weak-phrase pass.
   Smooth gain compression is used only as model input; it cannot alter output
   velocity. Any audible piano or guitar cue selected for a protected lane is
   transcribed independently in both ordinary and weak-phrase passes.
5. The two accompaniment passes are not merged blindly. Neural confidence,
   pitch-band loudness, pitch-to-neighborhood S/N, physical attack strength,
   duration, and agreement between passes remain independent measurements. A
   loud neighboring tone can no longer validate an unsupported candidate pitch.
6. Vocal and bass candidates are solved as complete phrase paths. A global
   continuity optimizer rejects isolated octave errors, while a conservative
   pYIN track fills only genuine gaps in the neural result.
7. A whole-song hidden-state tracker follows one persistent accompaniment
   focus. Isolated register distractions become support-only events, while a
   sustained new phrase can make a deliberate section handoff.
8. Guitar or piano attacks that were masked in the combined pass are admitted
   after validation against their untouched isolated stem. Recurrence supports
   ambiguous notes, while a one-off quiet note can survive when its pitch-band
   S/N and physical attack are independently clear. Core-note duplicates,
   separator leakage, and excess events above a strict song-length budget are
   rejected. The first three accompaniment lanes remain lead, low anchor, and
   harmony; additional lanes are reserved for independently supported
   instruments.
9. Each neural onset is refined exactly once from a short-window, pitch-harmonic
   attack track in its own aligned stem. Drum attacks from the full mix and a
   second beat-lattice snap cannot move melodic notes. Overlapping chord tones
   may share an onset only when the complete group fits within the tolerance;
   transitive chaining cannot collapse a short sequence. A role-aware voice
   planner then protects the lead and low chord anchor, limits chord size, and
   assigns one stable instrument to each NBS layer for the entire song.
10. Per-note velocity is measured from that note's own CQT pitch band at its
    unquantized source time. Neural confidence only trims the result and is never
    treated as volume. Quiet source notes retain quiet velocities without a
    role-specific minimum-volume boost. A very quiet note may still survive when
    local pitch S/N and attack evidence are strong, while pitch-band silence
    removes a false candidate before writing. Protected guitar and piano lanes
    use their own isolated waveforms rather than the louder combined
    accompaniment.
11. Independent CQT analysis checks active regions that still have no credible
   AI note and restores only locally supported missing notes.
12. Kick, snare, and hi-hat transients are analyzed in separate frequency bands.
   Composite hits may retain two strongly supported components instead of being
   collapsed into one drum or expanded into a noisy three-note stack.
13. Tempo is tracked from a high-resolution consensus of the percussion and
    complete mix, resolves only well-supported half/double-tempo extremes, and
    is fitted across the complete beat sequence instead of trusting one
    hop-quantized candidate. After its single physical-onset refinement, every
    AI, spectral, pitch, and percussion event uses the same deterministic
    absolute-time quantizer exactly once. Sustained-note repeats remain
    phase-locked to the original sub-tick onset and are rounded independently,
    so neither fractional beat lengths nor a one-frame BPM estimate can
    accumulate timing drift.
14. The default native NBS timeline is `40.00 ticks/s`, limiting serialization
    error to 12.5 ms and keeping rapid notes distinct. A pre-write invariant
    checks every timed neural and percussion note against its source position.
    `--timing minecraft` deliberately uses the coarser NBT-compatible
    `10.00 ticks/s` grid when in-game schematic/structure timing is required.

## Setup

On Windows, double-click [setup.bat](setup.bat). It creates `.venv` and installs
PyTorch, Demucs, Basic Pitch with its ONNX backend, the audio dependencies, and
the vocal detector.

When an NVIDIA GPU is available, the setup script installs the CUDA build of
PyTorch. The first setup downloads roughly 2 GB. Demucs model weights and the
approximately 16 MB vocal detector are downloaded when first needed.

If MP3 loading fails because FFmpeg is unavailable, install
[FFmpeg](https://ffmpeg.org/download.html) and add it to `PATH`.

## Convert audio

Double-click [convert.bat](convert.bat), then paste or drag the input path into
the window. By default, the converter writes a file with the same base name and
an `.nbs` extension beside the source audio.

Run it directly from PowerShell when you need options:

```powershell
.\.venv\Scripts\python.exe mp3_to_nbs.py "C:\Music\song.mp3"
```

Choose an output path and author:

```powershell
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 -o converted.nbs --author "Your Name"
```

Use `--force` to overwrite an existing output file.

## Common options

```powershell
# Override detected BPM while retaining the precise 40 ticks/s timeline.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --bpm 128

# Deliberately use the coarser Minecraft schematic/structure-compatible timeline.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --timing minecraft

# Use an eight-subdivision BPM grid for NBS-only playback.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --timing beat --bpm 128 --ticks-per-beat 8

# Disable or force the dedicated vocal layer when automatic detection is unsuitable.
.\.venv\Scripts\python.exe mp3_to_nbs.py instrumental.mp3 --vocals off
.\.venv\Scripts\python.exe mp3_to_nbs.py vocal.mp3 --vocals on

# Reduce or increase the maximum accompaniment voice count. The default four
# lanes are lead, low anchor, harmony, and one protected background instrument.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --max-chord-notes 2

# Increase sensitivity only if the automatic weak-phrase recovery still misses notes.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --sensitivity 0.6

# Omit percussion.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --no-drums

# Compare the neural transcription with the CQT/pYIN-only path.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --transcription dsp

# Skip Demucs for a faster, lower-quality conversion.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --separation basic

# Keep the full 88-key NBS range instead of folding pitches into Minecraft's range.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --full-range
```

Show every option with:

```powershell
.\.venv\Scripts\python.exe mp3_to_nbs.py --help
```

## Progress, performance, and cache

The console combines model preparation, all Demucs passes, transcription, audio
validation, arrangement, and writing into one progress display:

```text
[##########--------]  54.75% | ETA about 12s | AI separation 31/40
```

The default mode prioritizes separation and arrangement quality over speed. It
can use substantial GPU, CPU, VRAM, RAM, power, and disk bandwidth on long songs.

Separated stems are stored in `.stem_cache` using content- and model-derived
keys. A second conversion of the same source skips both Demucs passes and begins
with transcription. Downloaded detection models are stored in `.model_cache`.
Both directories can be removed while no conversion is running if disk space
must be reclaimed.

## Accuracy limits

A finished stereo mix does not contain the original MIDI, score, isolated
instruments, or note-length decisions. No converter can reconstruct those data
perfectly for every recording. Distorted guitars, dense choirs, heavy reverb,
closely voiced chords, and unusual percussion can still require manual review.

NBS notes also have no conventional MIDI duration. A true source onset is kept
at its measured, role-balanced velocity, while a held note receives a quiet
continuation every two beats by default. Each continuation follows the source
loudness envelope and disappears after the stem has faded, avoiding the
mechanical one-note repetition heard at a one-beat interval. Continuations also
cannot be normalized back into false full-strength accents. Use
`--retrigger-beats 0` when a sound pack has long samples and needs no sustain
refresh. Each continuation is calculated from its original source onset rather
than from the preceding rounded tick. Minecraft's playable note-block range is
only two octaves, so the default mode folds source octaves and keeps one note per
pitch class to avoid noisy unison stacks.

Use the default sensitivity first. The converter already runs a locally
normalized weak-phrase pass and validates its candidates against the untouched
source, so a global sensitivity increase is normally unnecessary. Raise it in
small steps only when an important part remains absent; excessive sensitivity
necessarily increases ambiguous candidates. If a short shout is intentionally part of the melody, use
`--vocals on`. If an instrument is mistaken for a singer, use `--vocals off`.
The default fourth accompaniment lane protects the most prominent independently
separated guitar or piano part. Increase `--max-chord-notes` to 5 only when a
song clearly contains both and the extra density is appropriate.

The default `--timing precise` and optional `--timing beat` can produce a tick
rate other than 10, 5, or 2.5 ticks/s. Such a song plays at the intended speed
in Open Note Block Studio but can change speed after Minecraft NBT or structure
export. Select `--timing minecraft` for that in-game export path; its 100 ms grid
necessarily merges source attacks that occur inside the same tick.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
