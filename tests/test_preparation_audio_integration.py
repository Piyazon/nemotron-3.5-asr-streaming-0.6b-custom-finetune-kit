"""Real Hugging Face mapping and audio I/O, using synthetic local recordings.

These checks do not require TorchCodec, a GPU, or access to the remote corpora.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import prepare_hf_uyghur_fast as prepare


HAS_AUDIO_LIBRARIES = all(
    importlib.util.find_spec(name) is not None for name in ("datasets", "numpy", "soundfile")
)


@unittest.skipUnless(HAS_AUDIO_LIBRARIES, "Install datasets, numpy and soundfile for real audio checks")
class PreparationAudioIntegrationTests(unittest.TestCase):
    def test_multiprocess_export_preserves_audio_and_transcripts(self):
        from datasets import Dataset
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = {}
            parts = []
            for source_index, source in enumerate(prepare.DATASET_IDS):
                clips = [
                    (0.25 * np.sin(2 * np.pi * (220 + source_index * 110) * np.arange(8000 + i * 800) / 16000)).astype(np.float32)
                    for i in range(4)
                ]
                texts = [f"مەن باردىم {i}." for i in range(4)]
                ds = Dataset.from_dict({
                    "audio": [{"array": clip.tolist(), "sampling_rate": 16000} for clip in clips],
                    "sentence": texts,
                    "sentence_latn": ["men bardim"] * len(clips),
                })
                for fmt in ("wav", "flac"):
                    manifest = root / f"{source_index}-{fmt}.jsonl"
                    stats = prepare.export_split(ds, "train", manifest, root / "audio", 2, 2, fmt, source)
                    self.assertEqual(stats[:2], (4, 0))
                    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
                    self.assertEqual([row["text"] for row in rows], texts)
                    for i, row in enumerate(rows):
                        path = Path(row["audio_filepath"])
                        self.assertNotIn(path, originals)
                        originals[path] = path.stat().st_mtime_ns
                        audio, rate = sf.read(path, dtype="float32")
                        self.assertEqual(rate, 16000)
                        self.assertEqual(audio.shape, clips[i].shape)
                        self.assertLessEqual(float(np.max(np.abs(audio - clips[i]))), 1 / 32768)
                        self.assertEqual(row["language"], "ug-CN")
                        self.assertAlmostEqual(row["duration"], len(audio) / rate, places=4)
                    # Resume must reuse completed files and preserve the same rows.
                    prepare.export_split(ds, "train", manifest, root / "audio", 2, 2, fmt, source)
                    for row in rows:
                        path = Path(row["audio_filepath"])
                        self.assertEqual(path.stat().st_mtime_ns, originals[path])
                    if fmt == "flac":
                        parts.append(manifest)
            combined = root / "combined.jsonl"
            prepare.merge_manifests(parts, combined)
            rows = [json.loads(line) for line in combined.read_text().splitlines()]
            self.assertEqual(len(rows), 4 * len(prepare.DATASET_IDS))
            self.assertEqual(len({row["audio_filepath"] for row in rows}), len(rows))
            self.assertEqual({row["source_dataset"] for row in rows}, set(prepare.DATASET_IDS))


if __name__ == "__main__":
    unittest.main()
