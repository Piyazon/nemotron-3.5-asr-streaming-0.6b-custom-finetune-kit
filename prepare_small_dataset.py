#!/usr/bin/env python3
"""Prepare the locally downloaded Uyghur samples without changing their sources.

Requires ffmpeg and umsc==0.5.0. Use --backup-existing to revise existing output.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
import xml.etree.ElementTree as ET


SAMPLE_RATE = 16000
MIN_SECONDS = 1.0
MAX_SECONDS = 40.0
ULY_LETTERS = set("abcdefghijklmnopqrstuvwxyzéöü")
MERGE_TARGET_MS = 5000
MERGE_MAX_MS = 12000
MERGE_GAP_MS = 500


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def timestamp_ms(value):
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})\.(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


@dataclass
class Unit:
    ref: str
    speaker: str
    text: str
    morphology: str
    start_ms: int
    end_ms: int
    eaf_id: str
    xml_start_ms: int
    xml_end_ms: int


def read_units(xml_path, eaf_path):
    """Match IU tiers by speaker/order, checking text and the XML rounding error."""
    xml = ET.parse(xml_path).getroot()
    eaf = ET.parse(eaf_path).getroot()
    if eaf.find("HEADER").get("TIME_UNITS") != "milliseconds":
        raise ValueError(f"Unsupported EAF time units: {eaf_path}")
    slots = {s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE"))
             for s in eaf.findall("./TIME_ORDER/TIME_SLOT")}
    tiers = {}
    for tier in eaf.findall("TIER"):
        if tier.get("LINGUISTIC_TYPE_REF") != "Intonation Units":
            continue
        speaker = "Speaker " + tier.get("PARTICIPANT")
        if speaker in tiers:
            raise ValueError(f"Duplicate speaker tier: {speaker}")
        tiers[speaker] = tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION")
    positions = Counter()
    units = []
    refs = set()
    # Header metadata also contains an <annotation>; it is not speech.
    for annotation in xml.findall(".//transcript_body//annotation"):
        speaker, ref = annotation.get("who"), annotation.get("ref")
        if ref in refs:
            raise ValueError(f"Duplicate XML annotation: {ref}")
        refs.add(ref)
        text = normalize(annotation.findtext("iu", ""))
        start, end = map(timestamp_ms, annotation.findtext("timestamp").split("-"))
        aligned = tiers[speaker][positions[speaker]]
        positions[speaker] += 1
        eaf_start = slots[aligned.get("TIME_SLOT_REF1")]
        eaf_end = slots[aligned.get("TIME_SLOT_REF2")]
        eaf_text = normalize(aligned.findtext("ANNOTATION_VALUE", ""))
        if text != eaf_text or max(abs(start - eaf_start), abs(end - eaf_end)) > 1:
            raise ValueError(f"XML/EAF mismatch: {xml_path.name}, annotation {ref}")
        if not 0 <= eaf_start < eaf_end:
            raise ValueError(f"Invalid interval: {xml_path.name}, annotation {ref}")
        units.append(Unit(ref, speaker, text, annotation.findtext("seg", ""),
                          eaf_start, eaf_end, aligned.get("ANNOTATION_ID"), start, end))
    if dict(positions) != {speaker: len(items) for speaker, items in tiers.items()}:
        raise ValueError(f"XML/EAF annotation count mismatch: {xml_path}")
    return units, normalize(xml.findtext(".//rights", "Unspecified"))


def overlap_refs(units):
    """Any positive overlap with another speaker requires transcript review."""
    active = []
    overlapping = set()
    for unit in sorted(units, key=lambda u: (u.start_ms, u.end_ms, u.ref)):
        active = [other for other in active if other.end_ms > unit.start_ms]
        for other in active:
            if other.speaker != unit.speaker:
                overlapping.update((other.ref, unit.ref))
        active.append(unit)
    return overlapping


def text_issues(text, morphology=""):
    issues = []
    if not text:
        issues.append("empty_transcript")
    if re.search(r"[\[\]{}<>]|\bxxx\b", text, re.I):
        issues.append("redacted_or_unclear_words")
    if re.search(r"--|[–—]|\w-(?=\s|$|[,.!?])", text):
        issues.append("incomplete_speech")
    if re.search(r"\b(?:EN|CH):", morphology):
        issues.append("foreign_words_need_arabic_transcription")
    if any(c.isalpha() and c.lower() not in ULY_LETTERS for c in text):
        issues.append("letters_outside_uyghur_latin_alphabet")
    if re.search(r"\b(?:ha){2,}\b", text, re.I):
        issues.append("laughter_annotation")
    # Include hyphenated/repeated humming such as "Hm-mm. Hm-mm.".
    # Mixed speech containing these annotations also needs review; do not
    # silently delete a vocalization from the supplied target.
    if any("m" in token.lower() and set(token.lower()) <= {"h", "m"}
           for token in re.findall(r"[A-Za-zéöüÉÖÜ]+", text)):
        issues.append("nonlexical_vocalization")
    # These are ambiguous source conventions, not a rule that every hyphen
    # is wrong. Preserve other internal hyphens (paired words and ranges).
    if re.search(
        r"\b(?:we|essalamu)-(?:a|e)leykum(?:-essalam)?\b"
        r"|\b\w+-(?:da|de|gha|ge|qa|ke|din|tin|ning|ni)\b"
        r"|\brohiy-keypiyati\b|(?<!\S)-(?!\S)", text, re.I
    ):
        issues.append("hyphen_spelling_needs_review")
    if re.search(r"\b(?:online|zoom)-\w+", text, re.I):
        issues.append("foreign_words_need_arabic_transcription")
    return issues


def group_units(units, issues, target_ms=MERGE_TARGET_MS,
                max_ms=MERGE_MAX_MS, max_gap_ms=MERGE_GAP_MS,
                minimum_ms=1000):
    """Group consecutive, clean IUs; never bridge a flagged annotation or turn."""
    def split_run(run):
        chunks, pending = [], []
        for unit in run:
            if pending and (pending[-1].end_ms - pending[0].start_ms >= target_ms
                            or unit.end_ms - pending[0].start_ms > max_ms):
                chunks.append(pending)
                pending = []
            pending.append(unit)
        if pending:
            # Avoid an isolated short tail when it still fits the previous clip.
            if (chunks and pending[-1].end_ms - pending[0].start_ms < minimum_ms
                    and pending[-1].end_ms - chunks[-1][0].start_ms <= max_ms):
                chunks[-1].extend(pending)
            else:
                chunks.append(pending)
        return chunks

    groups, run = [], []
    for unit in sorted(units, key=lambda u: (u.start_ms, u.end_ms, u.ref)):
        if issues[unit.ref]:
            groups.extend(split_run(run))
            run = []
            groups.append([unit])
            continue
        if run and (unit.speaker != run[-1].speaker
                    or not 0 <= unit.start_ms - run[-1].end_ms <= max_gap_ms):
            groups.extend(split_run(run))
            run = []
        run.append(unit)
    groups.extend(split_run(run))
    return groups


def arabic_text(text, converter):
    # Keep lexical content, compounds and sentence punctuation. Ellipses in
    # these transcripts denote pauses, and apostrophe variants encode hamza.
    text = normalize(text).replace("’", "'").replace("‘", "'")
    text = re.sub(r"\.{3,}|…+", " ", text)
    result = normalize(converter(normalize(text)))
    if not valid_arabic(result):
        raise ValueError("Conversion did not produce an Arabic-only transcript")
    return result


def valid_arabic(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all("ARABIC" in unicodedata.name(c, "") for c in letters)


def decode_audio(source, target, ffmpeg):
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(source),
                    "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le", "-y", str(target)], check=True)
    with wave.open(str(target), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"Unexpected decoded format: {source}")
        return reader.readframes(reader.getnframes())


def cut_pcm(pcm, start_ms, end_ms):
    start = start_ms * SAMPLE_RATE // 1000
    end = end_ms * SAMPLE_RATE // 1000
    if not 0 <= start < end <= len(pcm) // 2:
        raise ValueError(f"Annotation outside decoded audio: {start_ms}-{end_ms} ms")
    return pcm[start * 2:end * 2]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class DatasetWriter:
    def __init__(self, staging, output):
        self.staging = staging
        self.output = output
        self.accepted = []
        self.review = []

    def add(self, clip_id, pcm, text, provenance, reasons=()):
        reasons = list(reasons)
        duration = len(pcm) / (2 * SAMPLE_RATE)
        if duration < MIN_SECONDS:
            reasons.append("below_1_second_after_merging")
        if duration > MAX_SECONDS:
            reasons.append("above_40_seconds")
        if not any(pcm):
            reasons.append("silent_audio")
        if text is None and not reasons:
            raise ValueError(f"Missing transcript: {clip_id}")
        if text is not None and not valid_arabic(text):
            reasons.append("non_arabic_transcript")
            text = None
        folder = "review/audio" if reasons else "audio"
        relative = Path(folder) / (clip_id + ".wav")
        path = self.staging / relative
        if path.exists():
            raise ValueError(f"Duplicate clip ID: {clip_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(pcm)
        row = {"audio_filepath": str(self.output / relative),
               "file_name": relative.as_posix(), "duration": duration, "text": text,
               "lang": "ug", "sample_rate": SAMPLE_RATE, "num_samples": len(pcm) // 2,
               "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
               "transcript_verified_against_audio": False, **provenance}
        if reasons:
            row["review_reasons"] = sorted(set(reasons))
            self.review.append(row)
        else:
            self.accepted.append(row)


def verify_outputs(writer):
    """Read every generated file back; verify frames, content and training text."""
    seen = set()
    for rows in (writer.accepted, writer.review):
        for row in rows:
            path = writer.staging / row["file_name"]
            if path in seen:
                raise ValueError(f"Duplicate output: {path}")
            seen.add(path)
            with wave.open(str(path), "rb") as reader:
                if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()) != (1, 2, SAMPLE_RATE, row["num_samples"]):
                    raise ValueError(f"WAV format/frame mismatch: {path}")
                pcm = reader.readframes(reader.getnframes())
            if hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
                raise ValueError(f"PCM mismatch: {path}")
            if row["text"] is not None and not valid_arabic(row["text"]):
                raise ValueError(f"Invalid output text: {path}")
            if "start_ms" in row and row["num_samples"] != (row["end_ms"] - row["start_ms"]) * 16:
                raise ValueError(f"Timestamp/frame mismatch: {path}")
            if "source_segments" in row:
                segments = row["source_segments"]
                if (segments[0]["start_ms"], segments[-1]["end_ms"]) != (row["start_ms"], row["end_ms"]):
                    raise ValueError(f"Merged source interval mismatch: {path}")
                for left, right in zip(segments, segments[1:]):
                    if not 0 <= right["start_ms"] - left["end_ms"] <= MERGE_GAP_MS:
                        raise ValueError(f"Invalid gap in merged clip: {path}")
    if len(list(writer.staging.rglob("*.wav"))) != len(seen):
        raise ValueError("Unindexed WAV files in output")


def prepare(source, output, ffmpeg, backup_existing=False):
    from umsc import UgMultiScriptConverter

    backup = None
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output is not a directory: {output}")
        if any(output.iterdir()):
            if not backup_existing:
                raise ValueError(f"Use --backup-existing or an empty output directory: {output}")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = output.with_name(f"{output.name}_backup_{stamp}")
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = UgMultiScriptConverter("ULS", "UAS")
    with tempfile.TemporaryDirectory(prefix=".small-dataset-", dir=output.parent) as work:
        staging = Path(work) / "dataset"
        staging.mkdir()
        decoded = Path(work) / "decoded.wav"
        writer = DatasetWriter(staging, output)
        timing_report = {}

        for number in range(1, 5):
            xml_path = source / "transcripts" / f"conversation_{number}.xml"
            eaf_path = source / "transcripts" / f"Conversation{number}_transcript_final.eaf"
            audio_path = source / "transcripts" / f"Conversation{number}.mp3"
            units, rights = read_units(xml_path, eaf_path)
            overlaps = overlap_refs(units)
            pcm = decode_audio(audio_path, decoded, ffmpeg)
            timing_report[f"conversation_{number}"] = {
                "annotations": len(units), "decoded_seconds": len(pcm) / 32000,
                "overlapping_annotations": len(overlaps),
                "xml_times_differing_by_at_most_1ms": sum(
                    (u.start_ms, u.end_ms) != (u.xml_start_ms, u.xml_end_ms) for u in units),
                "all_xml_and_eaf_texts_match": True,
            }
            issues_by_ref, texts = {}, {}
            for unit in units:
                issues = text_issues(unit.text, unit.morphology)
                # Ambiguous text stays null rather than inventing Arabic words
                # or deleting spoken content from its training target.
                # Retain an Arabic candidate for spelling-only review. It is
                # still excluded from the training manifest until corrected.
                text = None if set(issues) - {"hyphen_spelling_needs_review"} else arabic_text(unit.text, converter)
                if unit.ref in overlaps:
                    issues.append("overlapping_speakers")
                issues_by_ref[unit.ref], texts[unit.ref] = issues, text
            groups = group_units(units, issues_by_ref)
            covered = [unit.ref for group in groups for unit in group]
            if Counter(covered) != Counter(unit.ref for unit in units):
                raise ValueError("Grouping lost or duplicated source annotations")
            for group in groups:
                first, last = group[0], group[-1]
                refs = [unit.ref for unit in group]
                if len(group) > 1 and any(
                    unit.ref not in refs and unit.start_ms < last.end_ms
                    and unit.end_ms > first.start_ms for unit in units
                ):
                    raise ValueError("Merged audio includes an unlisted speech annotation")
                issues = sorted({issue for ref in refs for issue in issues_by_ref[ref]})
                text = normalize(" ".join(texts[ref] for ref in refs)) if all(texts[ref] is not None for ref in refs) else None
                clip_id = f"{int(first.ref):04d}"
                if len(group) > 1:
                    clip_id += f"-{int(last.ref):04d}"
                writer.add(f"conversation_{number}/{clip_id}",
                           cut_pcm(pcm, first.start_ms, last.end_ms), text, {
                               "source_dataset": f"conversation_{number}",
                               "source_audio": audio_path.relative_to(source).as_posix(),
                               "source_transcript": xml_path.relative_to(source).as_posix(),
                               "source_annotation_refs": refs,
                               "eaf_annotation_ids": [unit.eaf_id for unit in group],
                               "source_segments": [{"ref": unit.ref, "start_ms": unit.start_ms,
                                                    "end_ms": unit.end_ms} for unit in group],
                               "speaker_id": f"conversation_{number}_{first.speaker[-1]}",
                               "start_ms": first.start_ms, "end_ms": last.end_ms,
                               "merged_annotation_count": len(group),
                               "included_gap_ms": sum(right.start_ms - left.end_ms
                                                      for left, right in zip(group, group[1:])),
                               "timing_source": eaf_path.relative_to(source).as_posix(),
                               "transcript_source": "iu_converted_from_ULY_to_Arabic",
                               "source_rights": rights,
                           }, issues)
            print(f"Conversation {number}: grouped {len(units)} annotations into {len(groups)} clips", flush=True)

        aishell = source / "AISHELL-ASR0081-Samples"
        for line in (aishell / "DOC/content.txt").read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            name, text = line.split("\t", 1)
            audio_path = aishell / "WAV" / name
            writer.add(f"aishell/{Path(name).stem}", decode_audio(audio_path, decoded, ffmpeg),
                       normalize(text), {
                           "source_dataset": "AISHELL-ASR0081-Samples",
                           "source_audio": audio_path.relative_to(source).as_posix(),
                           "transcript_source": "AISHELL-ASR0081-Samples/DOC/content.txt",
                       })
        print("Prepared AISHELL sentence samples", flush=True)

        ocean = source / "speechocean_samples"
        for line in (ocean / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            audio_path = ocean / record["audio_file"]
            text = normalize(record["text"]) if record.get("text") else None
            writer.add(f"speechocean/{record['dataset']}/{audio_path.stem}",
                       decode_audio(audio_path, decoded, ffmpeg), text, {
                           "source_dataset": record["dataset"],
                           "source_audio": audio_path.relative_to(source).as_posix(),
                           "transcript_source": record["transcript_source"],
                           "source_page": record["source_page"],
                           "source_audio_url": record["source_audio_url"],
                           "original_sample_rate": record["sample_rate"],
                       }, [] if text else ["missing_transcript_and_timestamps"])
        print("Prepared SpeechOcean samples; checking every output WAV", flush=True)
        verify_outputs(writer)

        per_source = defaultdict(lambda: {"accepted_clips": 0, "accepted_seconds": 0,
                                          "review_clips": 0, "review_seconds": 0})
        for status, rows in (("accepted", writer.accepted), ("review", writer.review)):
            for row in rows:
                per_source[row["source_dataset"]][f"{status}_clips"] += 1
                per_source[row["source_dataset"]][f"{status}_seconds"] += row["duration"]
        durations = sorted(row["duration"] for row in writer.accepted)
        summary = {
            "accepted_clips": len(writer.accepted),
            "accepted_seconds": sum(durations),
            "review_clips": len(writer.review),
            "review_seconds": sum(row["duration"] for row in writer.review),
            "merged_accepted_clips": sum(row.get("merged_annotation_count", 1) > 1 for row in writer.accepted),
            "accepted_conversation_annotations": sum(row.get("merged_annotation_count", 0) for row in writer.accepted),
            "grouping": {"target_seconds": MERGE_TARGET_MS / 1000,
                         "max_merged_seconds": MERGE_MAX_MS / 1000,
                         "max_gap_seconds": MERGE_GAP_MS / 1000,
                         "minimum_accepted_seconds": MIN_SECONDS,
                         "same_speaker_only": True},
            "previous_output_backup": str(backup) if backup else None,
            "review_reasons": dict(Counter(reason for row in writer.review for reason in row["review_reasons"])),
            "accepted_duration_seconds": {
                "min": durations[0], "median": durations[len(durations) // 2], "max": durations[-1],
                "less_than_one_second": sum(d < 1 for d in durations),
            },
            "duplicate_accepted_pcm_hashes": {h: n for h, n in Counter(row["pcm_sha256"] for row in writer.accepted).items() if n > 1},
            "sample_rate": SAMPLE_RATE, "channels": 1, "encoding": "PCM_16",
            "transcript_script": "Arabic", "converter": "umsc==0.5.0 (ULS -> UAS)",
            "source_root_at_preparation": str(source),
            "per_source": dict(per_source), "conversation_alignment_checks": timing_report,
            "all_output_wavs_verified": True,
            "merged_intervals_checked_for_unlisted_annotations": True,
        }
        write_jsonl(staging / "manifest.jsonl", writer.accepted)
        write_jsonl(staging / "metadata.jsonl", [
            {k: v for k, v in row.items() if k != "audio_filepath"} for row in writer.accepted])
        write_jsonl(staging / "review/review.jsonl", writer.review)
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(
            f"# Prepared Uyghur samples\n\n"
            f"{len(writer.accepted):,} clips with Arabic transcripts, "
            f"{sum(durations) / 60:.2f} minutes. All audio is mono, 16 kHz, PCM16 WAV.\n\n"
            "- `audio/`: clips that passed the automated filters.\n"
            "- `manifest.jsonl`: NeMo entries with absolute local audio paths.\n"
            "- `metadata.jsonl`: the same entries with portable `file_name` paths relative to this folder.\n"
            "- `review/audio/` and `review/review.jsonl`: retained clips needing review, excluded from the main manifest.\n"
            "- `summary.json`: counts, durations, exclusions and alignment checks.\n\n"
            "## Preparation\n\n"
            "The four conversations are cut at their supplied intonation-unit boundaries, "
            "using ELAN's integer millisecond timestamps. Each ELAN text was checked against "
            "the XML `iu` text; their time differences are at most 1 ms. Each recording was "
            "decoded once before cutting, with no added padding, silence trimming, denoising "
            "or word-level realignment. Adjacent clean phrases from the same speaker are "
            "merged toward 5 seconds, up to 12 seconds per merged clip, with gaps no longer "
            "than 0.5 seconds. Merging stops at every flagged annotation or speaker change. "
            "The original audio between boundaries is retained, including pauses. A short "
            "tail is attached to its preceding group when it fits within 12 seconds. "
            "Remaining clips under 1 second are held for review. "
            "Original source files are retained outside this folder.\n\n"
            "Conversation text is converted mechanically from Uyghur Latin to Arabic with "
            "umsc 0.5.0. Whitespace and Unicode are normalized; apostrophe variants are unified "
            "before conversion and pause ellipses are removed. Punctuation, compounds, dialect "
            "spellings, fillers and spoken repetitions are otherwise retained. No spelling "
            "correction or Arabic-to-Latin conversion is performed. The AISHELL and eight "
            "captioned SpeechOcean samples retain their supplied Arabic text.\n\n"
            "Any positive overlap between annotated speakers is held for review. Units with "
            "redacted/unclear text, incomplete speech, tagged foreign words, unsupported Latin "
            "letters or nonlexical vocalizations (including hyphenated humming) are also "
            "held back. Internal compound/range hyphens are preserved. Hyphens in greetings, "
            "isolated separators, bare suffix attachments and the source phrase "
            "`روھىي-كەيپىياتى` are held for spelling review rather than automatically "
            "removed or replaced. An Arabic candidate is retained for spelling-only review; "
            "other ambiguous transcripts are null. Source annotation IDs identify the original text. "
            "The four untranscribed SpeechOcean previews remain whole in review, because "
            "they have no supplied timestamps or transcripts. Accepted durations are 1–40 s. "
            "Review duration sums may count simultaneous speech more than once.\n\n"
            "All output WAVs were read back and checked for format, frame count, PCM hash "
            "and timestamp agreement. These checks do not constitute listening verification "
            "of transcript accuracy or human review of the Arabic transliteration. "
            "SpeechOcean captions have not been verified against their recordings. "
            "Resampling 8 kHz telephone audio to 16 kHz does not restore missing bandwidth.\n\n"
            "## Use\n\n"
            "Use only `manifest.jsonl` for supervised training. If moving this directory, "
            "rebuild absolute audio paths from `metadata.jsonl`'s relative `file_name`. "
            "No train/validation split is assigned; keep whole conversations together "
            "when splitting to avoid adjacent clips crossing splits.\n\n"
            "The source conversation XML states: ‘Restricted access: Creators allow academic "
            "use.’ Source restrictions are preserved; preparation does not grant redistribution "
            "rights. Check source terms before public upload. SpeechOcean source URLs are "
            "included in each corresponding metadata entry.\n\n"
            "To reproduce into a new or empty folder from the repository root:\n\n"
            "```sh\nsource /Users/pi/Desktop/PyEnv/Nemo/bin/activate\n"
            "python -m pip install umsc==0.5.0\n"
            "python prepare_small_dataset.py --output small_dataset/final_new\n```\n\n"
            "Use `--backup-existing` to rebuild an existing output. The previous folder "
            "is preserved verbatim in a dated sibling directory. Its `metadata.jsonl` "
            "uses portable relative paths; its absolute manifest paths still refer to "
            "the original folder location until the backup is restored there.\n",
            encoding="utf-8")
        # Publish after verification and retain the previous output verbatim.
        # Its portable metadata remains usable under the backup directory;
        # its absolute paths refer to the original location until restored.
        if backup:
            output.rename(backup)
        elif output.exists():
            output.rmdir()  # Fails safely if anything appeared in the directory.
        try:
            staging.rename(output)
        except OSError:
            if backup:
                backup.rename(output)
            raise
        if backup:
            print(f"Previous dataset preserved at {backup}", flush=True)
        print(json.dumps({k: summary[k] for k in ("accepted_clips", "accepted_seconds", "review_clips", "review_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("small_dataset"))
    parser.add_argument("--output", type=Path, default=Path("small_dataset/final"))
    parser.add_argument("--backup-existing", action="store_true",
                        help="Preserve existing output in a dated sibling folder before publishing")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg")
    args = parser.parse_args()
    prepare(args.input.resolve(), args.output.resolve(), args.ffmpeg, args.backup_existing)


if __name__ == "__main__":
    main()
