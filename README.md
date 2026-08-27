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
   All passes retain Basic Pitch's complete A0-C8 output instead of applying
   role-specific hard cutoffs, and the transient pass accepts attacks down to
   30 ms before source validation.
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
   and separator leakage are rejected. Independently verified guitar or piano
   unisons are retained rather than mistaken for leakage. The voice planner
   allocates background lanes from measured simultaneous demand (including
   multiple lanes for an isolated piano/guitar chord), leaving all remaining
   lanes available to supported core tones.
9. Each neural onset is refined exactly once from a short-window, pitch-harmonic
   attack track in its own aligned stem. Drum attacks from the full mix and a
   second beat-lattice snap cannot move melodic notes. Overlapping chord tones
   may share an onset only when the complete group fits inside a 20 ms window
   and no tone moves by more than 12.5 ms; audible strums stay staggered.
   Overlapping repeated attacks at the same pitch are kept as separate notes.
   A role-aware voice planner then protects the lead and low chord anchor,
   retains up to 12 independently supported tones by default, and assigns one
   stable instrument to each NBS layer for the entire song.
10. Per-note velocity is measured from that note's own CQT pitch band at its
    unquantized source time. Neural confidence only trims the result and is never
    treated as volume. Quiet source notes retain quiet velocities without a
    role-specific minimum-volume boost. A very quiet note may still survive when
    local pitch S/N and attack evidence are strong, while pitch-band silence
    removes a false candidate before writing. Protected guitar and piano lanes
    use their own isolated waveforms rather than the louder combined
    accompaniment.
11. Independent CQT analysis checks every physical attack for pitch-specific
   holes, including a missing inner tone at an onset where the AI already found
   part of the chord. Recovery is no longer limited to a small percentage of
   the primary transcription.
12. Kick, snare, and hi-hat transients are analyzed in separate frequency bands.
   Composite hits retain every independently supported component—including a
   real three-part kick/snare/hat hit—without expanding weak bands into a noisy
   stack. Neighboring band detections are grouped into one bounded physical
   attack and its time is refined between analysis frames.
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
    checks every timed neural, spectral, and percussion note against its source
    position. For recordings too long for 40 ticks/s in NBS v5's 16-bit
    timeline, precise mode automatically chooses the finest representable rate
    rather than failing or accumulating drift.
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

# Deliberately use Minecraft-compatible timing and fold pitches into its range.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --timing minecraft --minecraft-range

# Use an eight-subdivision BPM grid for NBS-only playback.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --timing beat --bpm 128 --ticks-per-beat 8

# Disable or force the dedicated vocal layer when automatic detection is unsuitable.
.\.venv\Scripts\python.exe mp3_to_nbs.py instrumental.mp3 --vocals off
.\.venv\Scripts\python.exe mp3_to_nbs.py vocal.mp3 --vocals on

# Reduce or increase the maximum accompaniment voice count (default: 12).
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --max-chord-notes 2

# Increase sensitivity only if the automatic weak-phrase recovery still misses notes.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --sensitivity 0.6

# Omit percussion.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --no-drums

# Compare the neural transcription with the CQT/pYIN-only path.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --transcription dsp

# Skip Demucs for a faster, lower-quality conversion.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --separation basic

# Lossily fold the default full 88-key output for an in-game note-block build.
.\.venv\Scripts\python.exe mp3_to_nbs.py song.mp3 --minecraft-range
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
than from the preceding rounded tick. Native NBS output keeps the complete
88-key range and distinct octave doublings by default. `--minecraft-range`
folds out-of-range pitches by octaves and deduplicates only notes that collapse
to the exact same instrument and output key.

Use the default sensitivity first. The converter already runs a locally
normalized weak-phrase pass and validates its candidates against the untouched
source, so a global sensitivity increase is normally unnecessary. Raise it in
small steps only when an important part remains absent; excessive sensitivity
necessarily increases ambiguous candidates. If a short shout is intentionally
part of the melody, use `--vocals on`. If an instrument is mistaken for a
singer, use `--vocals off`. The default 12 accompaniment lanes cover dense
two-handed chords while reserving lanes only for piano or guitar stems that are
actually supported. Lower `--max-chord-notes` when a deliberately sparse
Minecraft arrangement is preferred.

The default `--timing precise` and optional `--timing beat` can produce a tick
rate other than 10, 5, or 2.5 ticks/s. Such a song plays at the intended speed
in Open Note Block Studio but can change speed after Minecraft NBT or structure
export. Select `--timing minecraft --minecraft-range` for that in-game export
path; its 100 ms grid necessarily merges source attacks that occur inside the
same tick, and its two-octave range cannot retain every source register.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
