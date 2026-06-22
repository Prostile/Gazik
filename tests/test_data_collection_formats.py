from pathlib import Path

from synthetic_well_logs.datasets import load_well_metadata, validate_dataset_structure

ROOT = Path(__file__).parents[1]


def test_example_metadata_tops_and_lithology_are_valid() -> None:
    root = ROOT / "data/raw_las"
    metadata = next(root.rglob("well_metadata.yaml"))
    assert load_well_metadata(metadata).well_id == "WELL_001"
    report = validate_dataset_structure(root)
    assert report["valid"], report["errors"]
    assert report["well_count"] == 1


def test_invalid_metadata_has_clear_error(tmp_path: Path) -> None:
    metadata = tmp_path / "basin/WELL_BAD/metadata/well_metadata.yaml"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("well_id: WELL_BAD\nsource: synthetic\n", encoding="utf-8")
    report = validate_dataset_structure(tmp_path)
    assert not report["valid"]
    assert "invalid metadata" in report["errors"][0]

