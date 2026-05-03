import unittest
from pathlib import Path

from src.metadata_utils import derive_binary_label, infer_folder_label, normalize_datasplit, normalize_key, snake_case


class MetadataUtilsTest(unittest.TestCase):
    def test_label_synonyms_are_normalized(self) -> None:
        self.assertEqual(derive_binary_label("Alzheimer's"), (1, "dementia"))
        self.assertEqual(derive_binary_label("control"), (0, "non_dementia"))
        self.assertEqual(derive_binary_label(""), (None, "unknown"))

    def test_folder_label_accepts_project_nodementia_name(self) -> None:
        raw_root = Path("data/raw_audio")

        self.assertEqual(infer_folder_label(raw_root / "dementia" / "speaker" / "clip.wav", raw_root), "dementia")
        self.assertEqual(
            infer_folder_label(raw_root / "nodementia" / "speaker" / "clip.wav", raw_root),
            "non_dementia",
        )
        self.assertEqual(
            infer_folder_label(raw_root / "non_dementia" / "speaker" / "clip.wav", raw_root),
            "non_dementia",
        )

    def test_text_normalizers(self) -> None:
        self.assertEqual(snake_case("First Symptoms"), "first_symptoms")
        self.assertEqual(normalize_key("B. B. King"), "bbking")
        self.assertEqual(normalize_datasplit("validation"), "valid")


if __name__ == "__main__":
    unittest.main()
