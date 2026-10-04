"""Explicit retention for immutable study generations."""

from __future__ import annotations

import json

from pcd.cli import main
from pcd.results import prune_generations


def _study_with_generations(tmp_path):
    root = tmp_path / "study"
    generations = root / "generations"
    paths = [generations / f"g_{index}" for index in range(4)]
    for path in paths:
        path.mkdir(parents=True)
    (root / "study_result.json").write_text(
        json.dumps({"artifacts": {"generation": "generations/g_1"}}),
        encoding="utf-8",
    )
    return root, paths


def test_generation_prune_is_a_dry_run_and_preserves_the_active_generation(tmp_path):
    root, paths = _study_with_generations(tmp_path)

    plan = prune_generations(root, keep=1)

    assert not plan["applied"]
    assert "generations/g_1" in plan["kept"]
    assert all(path.exists() for path in paths)


def test_generation_prune_cli_requires_apply_before_removal(tmp_path, capsys):
    root, paths = _study_with_generations(tmp_path)

    main(["result-prune", str(root), "--keep", "1", "--apply", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["applied"]
    assert paths[1].exists()
    assert sum(path.exists() for path in paths) == 2
