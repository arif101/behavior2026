"""CPU tests for scripts/b1k/add_target_points.py (G3 point-dataset converter).

Builds a synthetic sweep fixture that exactly matches the documented formats
(labels_*.jsonl with {"frame","M","objs"} at TWO records per video frame + a LeRobot dataset
with 61-dim R1Pro observation.state and the per-frame robot2cam pose column), runs the
converter end-to-end via its CLI, and checks the geometry against hand-computed values.

The fixture uses a base pose with a nontrivial yaw (90 deg) so that a missing/incorrect
base-frame rotation of the displacement vector fails loudly.

Run:
    PYTHONPATH=src:packages/openpi-client/src .venv/bin/python -m pytest tests/test_add_target_points.py -v
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONVERTER = REPO_ROOT / "scripts" / "b1k" / "add_target_points.py"

_spec = importlib.util.spec_from_file_location("add_target_points", CONVERTER)
atp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atp)

# ----------------------------------------------------------------- fixture geometry

N_FRAMES = 60
RECORDS_PER_FRAME = 2

# Robot base world pose: yaw 90 deg + translation (rotation must matter for the test).
R_WB = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T_WB_POS = np.array([1.0, 2.0, 0.0])

# Camera pose in the BASE frame (pos + quat xyzw: 90 deg about x), i.e. T_base_cam.
CAM_POS_BASE = np.array([0.1, 0.0, 1.4])
CAM_QUAT_XYZW = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])

# Constant BASE-frame EE positions (R1Pro state slices 17:20 / 42:45).
EE_BASE = {"left": np.array([0.3, 0.2, 0.9]), "right": np.array([0.3, -0.2, 0.9])}

CAN1_A = np.array([1.5, 2.8, 0.8])
CAN1_B = np.array([0.2, 3.5, 0.8])
CAN2_A = np.array([2.5, 1.5, 0.8])
CAN2_B = np.array([0.3, 3.6, 0.8])
TRASH = np.array([0.2, 3.5, 0.4])


def _t_world_base() -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = R_WB
    t[:3, 3] = T_WB_POS
    return t


def _m_cam_world() -> np.ndarray:
    """Camera->world transform M = T_world_base @ T_base_cam (what the labels store)."""
    t_bc = np.eye(4)
    t_bc[:3, :3] = atp.quat_to_rot(CAM_QUAT_XYZW)
    t_bc[:3, 3] = CAM_POS_BASE
    return _t_world_base() @ t_bc


def _moving_traj(a: np.ndarray, b: np.ndarray, move_start: int, move_end: int) -> np.ndarray:
    """(N,3): at `a` until move_start, linear to `b` by move_end, then static."""
    traj = np.zeros((N_FRAMES, 3))
    for k in range(N_FRAMES):
        if k <= move_start:
            traj[k] = a
        elif k >= move_end:
            traj[k] = b
        else:
            traj[k] = a + (b - a) * (k - move_start) / (move_end - move_start)
    return traj


def _episode_objects(episode: int) -> dict[str, np.ndarray]:
    if episode == 0:
        # can_1 relocated mid-episode; can_2 never moves -> stays the live target to the end.
        can1 = _moving_traj(CAN1_A, CAN1_B, 20, 24)
        can2 = np.tile(CAN2_A, (N_FRAMES, 1))
    else:
        # Both cans relocated -> the tail of the episode has NO live target (mask must go False).
        can1 = _moving_traj(CAN1_A, CAN1_B, 10, 14)
        can2 = _moving_traj(CAN2_A, CAN2_B, 25, 29)
    return {
        "can_of_soda_1": can1,
        "can_of_soda_2": can2,
        "trash_can_1": np.tile(TRASH, (N_FRAMES, 1)),  # reference category: never a target
    }


def build_fixture(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Synthetic sweep output: LeRobot dataset + labels dir + task_targets.json."""
    dataset = root / "sweep_out" / "picking_up_trash" / "b1k" / "picking_up_trash"
    labels_dir = root / "sweep_labels" / "picking_up_trash"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "videos").mkdir()
    (dataset / "videos" / "placeholder.txt").write_text("not a real video")
    labels_dir.mkdir(parents=True)

    m_flat = _m_cam_world().reshape(-1).tolist()
    r2c = np.concatenate([CAM_POS_BASE, CAM_QUAT_XYZW]).astype(np.float32)

    features = {
        "observation.state": {"dtype": "float32", "shape": [61], "names": None},
        "observation.robot2cam_pose.zed_link_camera_0": {"dtype": "float32", "shape": [7], "names": None},
        "action": {"dtype": "float32", "shape": [23], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "R1Pro",
        "total_episodes": 2,
        "total_frames": 2 * N_FRAMES,
        "chunks_size": 1000,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
    }
    (dataset / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    with open(dataset / "meta" / "episodes.jsonl", "w") as f:
        for ep in range(2):
            f.write(json.dumps({"episode_index": ep, "tasks": ["picking_up_trash"], "length": N_FRAMES}) + "\n")
    (dataset / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "picking_up_trash"}) + "\n")

    rng = np.random.default_rng(0)
    for ep in range(2):
        objs = _episode_objects(ep)
        # --- parquet ---
        state = rng.normal(size=(N_FRAMES, 61)).astype(np.float32) * 0.01
        state[:, 17:20] = EE_BASE["left"]
        state[:, 42:45] = EE_BASE["right"]
        table = pa.table(
            {
                "observation.state": pa.array([r.tolist() for r in state], type=pa.list_(pa.float32())),
                "observation.robot2cam_pose.zed_link_camera_0": pa.array(
                    [r2c.tolist()] * N_FRAMES, type=pa.list_(pa.float32())
                ),
                "action": pa.array(
                    [r.tolist() for r in rng.normal(size=(N_FRAMES, 23)).astype(np.float32)],
                    type=pa.list_(pa.float32()),
                ),
                "timestamp": pa.array(np.arange(N_FRAMES, dtype=np.float32) / 30.0),
                "frame_index": pa.array(np.arange(N_FRAMES, dtype=np.int64)),
                "episode_index": pa.array(np.full(N_FRAMES, ep, dtype=np.int64)),
                "index": pa.array(np.arange(N_FRAMES, dtype=np.int64) + ep * N_FRAMES),
                "task_index": pa.array(np.zeros(N_FRAMES, dtype=np.int64)),
            }
        )
        pq.write_table(table, dataset / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        # --- labels: TWO records per video frame (both carry that frame's state) ---
        with open(labels_dir / f"labels_{ep:06d}.jsonl", "w") as f:
            for rec in range(RECORDS_PER_FRAME * N_FRAMES):
                k = rec // RECORDS_PER_FRAME
                f.write(
                    json.dumps(
                        {
                            "frame": rec,
                            "M": m_flat,
                            "objs": {name: traj[k].tolist() for name, traj in objs.items()},
                        }
                    )
                    + "\n"
                )

    task_targets = {
        "picking_up_trash": {"targets": ["can_of_soda"], "references": ["trash_can"], "n_goal_literals": 1}
    }
    tt_path = root / "task_targets.json"
    tt_path.write_text(json.dumps(task_targets))
    return {"dataset": dataset, "labels_dir": labels_dir, "task_targets": tt_path}


def expected_delta(obj_world: np.ndarray, arm: str) -> np.ndarray:
    """Hand-computable reference: R_world_base^T @ (p_obj_w - t_wb) - ee_base."""
    return R_WB.T @ (obj_world - T_WB_POS) - EE_BASE[arm]


def run_converter(fixture: dict, out: pathlib.Path, *extra: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--dataset", str(fixture["dataset"]),
            "--labels-dir", str(fixture["labels_dir"]),
            "--task", "picking_up_trash",
            "--task-targets", str(fixture["task_targets"]),
            "--out", str(out),
            "--validate",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"converter failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result.stdout


def _read_points(out_dir: pathlib.Path, ep: int) -> tuple[np.ndarray, np.ndarray, pa.Table]:
    table = pq.read_table(out_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    pts = np.array(table.column("target_points").to_pylist(), dtype=np.float32).reshape(-1, 2, 3)
    mask = np.array(table.column("target_points_mask").to_pylist(), dtype=bool)
    return pts, mask, table


# ---------------------------------------------------------------------- unit tests


def test_label_record_index_matches_contract():
    # frame k of an N-frame video maps to record idx = int((k+1)*LR/N) - 1
    n, lr = N_FRAMES, RECORDS_PER_FRAME * N_FRAMES
    for k in (0, 1, 29, 59):
        assert atp.label_record_index(k, lr, n) == int((k + 1) * lr / n) - 1 == 2 * k + 1
    # Degenerate: fewer records than frames still clamps into range.
    assert atp.label_record_index(0, 1, 10) == 0


def test_quat_to_rot_basics():
    np.testing.assert_allclose(atp.quat_to_rot(np.array([0, 0, 0, 1.0])), np.eye(3), atol=1e-12)
    yaw90 = atp.quat_to_rot(np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]))
    np.testing.assert_allclose(yaw90, R_WB, atol=1e-12)


# ------------------------------------------------------------------ end-to-end tests


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("g3_fixture")
    fixture = build_fixture(root)
    out = root / "picking_up_trash_pts"
    stdout = run_converter(fixture, out)
    return fixture, out, stdout


def test_geometry_against_hand_computed(converted):
    _, out, _ = converted
    pts, mask, _ = _read_points(out, 0)
    # Frame 0: both arms target can_1 (nearest); base-frame displacement, hand-computed.
    np.testing.assert_allclose(pts[0, 0], expected_delta(CAN1_A, "left"), atol=1e-5)   # [0.5, -0.7, -0.1]
    np.testing.assert_allclose(pts[0, 1], expected_delta(CAN1_A, "right"), atol=1e-5)  # [0.5, -0.3, -0.1]
    # Rotation preserves norm: |delta| == world-frame EE->object distance.
    ee_w_left = R_WB @ EE_BASE["left"] + T_WB_POS
    np.testing.assert_allclose(np.linalg.norm(pts[0, 0]), np.linalg.norm(CAN1_A - ee_w_left), atol=1e-5)
    # Frame 59: can_1 was relocated (retired ~frame 29) -> both arms switch to can_2.
    np.testing.assert_allclose(pts[59, 0], expected_delta(CAN2_A, "left"), atol=1e-5)
    np.testing.assert_allclose(pts[59, 1], expected_delta(CAN2_A, "right"), atol=1e-5)
    assert mask.all(), "episode 0 always has a live target (can_2 never relocates)"


def test_mask_goes_false_when_task_done(converted):
    _, out, _ = converted
    pts, mask, _ = _read_points(out, 1)
    assert mask[0].all()
    # can_2 (last live target) retires at frame 34 (stationary-from + hold logic).
    assert mask[33].all(), "target still live before retirement"
    assert not mask[34:].any(), "mask must be all-False once every target is relocated"
    assert np.all(pts[34:] == 0), "sentinel frames must carry zero points"
    # Between can_1's retirement and can_2's: both arms point at can_2.
    np.testing.assert_allclose(pts[20, 0], expected_delta(CAN2_A, "left"), atol=1e-5)


def test_source_untouched_and_copy_integrity(converted):
    fixture, out, _ = converted
    src_table = pq.read_table(fixture["dataset"] / "data" / "chunk-000" / "episode_000000.parquet")
    assert "target_points" not in src_table.column_names, "source dataset was modified!"
    src_info = json.loads((fixture["dataset"] / "meta" / "info.json").read_text())
    assert "target_points" not in src_info["features"]

    _, _, out_table = _read_points(out, 0)
    # Original columns byte-identical; only the two new columns added.
    for col in src_table.column_names:
        assert out_table.column(col).equals(src_table.column(col)), f"column {col} changed"
    assert set(out_table.column_names) - set(src_table.column_names) == {"target_points", "target_points_mask"}
    assert len(out_table) == len(src_table)

    out_info = json.loads((out / "meta" / "info.json").read_text())
    assert out_info["features"]["target_points"] == {
        "dtype": "float32",
        "shape": [6],
        "names": ["lx", "ly", "lz", "rx", "ry", "rz"],
    }
    assert out_info["features"]["target_points_mask"]["dtype"] == "bool"
    # Non-feature metadata + episodes metadata unchanged.
    assert {k: v for k, v in out_info.items() if k != "features"} == {
        k: v for k, v in src_info.items() if k != "features"
    }
    assert (out / "meta" / "episodes.jsonl").read_text() == (fixture["dataset"] / "meta" / "episodes.jsonl").read_text()
    assert (out / "videos" / "placeholder.txt").exists()


def test_validate_output_mentions_stats(converted):
    _, _, stdout = converted
    assert "mask coverage" in stdout
    assert "spot f=" in stdout
    assert "plausibility" in stdout
    assert "relocated: can_of_soda_1" in stdout


def test_rerun_with_overwrite_is_idempotent(converted, tmp_path):
    fixture, out, _ = converted
    before, mask_before, _ = _read_points(out, 0)
    run_converter(fixture, out, "--overwrite")
    after, mask_after, _ = _read_points(out, 0)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(mask_before, mask_after)


def test_b1k_inputs_accepts_converter_rows(converted):
    """The written columns must round-trip through B1KInputs' packing contract
    (np.asarray(...).reshape(2, 3) on the flat list[6] / reshape(2) on the mask)."""
    _, out, _ = converted
    pts, mask, table = _read_points(out, 0)
    row = 0
    points = np.asarray(table.column("target_points").to_pylist()[row], dtype=np.float32).reshape(2, 3)
    m = np.asarray(table.column("target_points_mask").to_pylist()[row]).astype(bool).reshape(2)
    np.testing.assert_allclose(points, pts[row])
    np.testing.assert_array_equal(m, mask[row])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
