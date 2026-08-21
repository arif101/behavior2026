"""Point the openpi b1k robot config at the robot name the eval harness actually uses.

THE MISMATCH
------------
openpi_fork/src/openpi/configs/robots/b1k.py declares the R1Pro robot as `name="robot"`, and
builds every observation key from that:

    name="robot"
    obs_key="robot::robot:zed_link:Camera:0::rgb"

The BEHAVIOR eval harness names the robot `robot_r1` and flattens observations as
`obs["robot_r1"]["camera"]["rgb"] -> obs["robot_r1::camera:::rgb"]` (evaluator.py:211), with the
head camera at `robot_r1:zed_link:Camera:0` (eval_utils.py:145). So the policy server receives
`robot_r1::proprio` and friends, while B1KPolicyWrapper looks up `f"{self.robot.name}::proprio"`
= `robot::proprio`, and dies with:

    KeyError: 'robot::proprio'          (eval_b1k_wrapper.py:132)

surfacing on the eval side as a websocket 1011 internal error on the FIRST inference call.

Our own og_smoke.py already carried the note "eval kit requires exactly this name" against
robot_r1, and the smoke test confirmed the live obs tree is rooted at robot_r1.

WHY A PATCH SCRIPT rather than editing in place: /root/openpi_fork on a box is not a git checkout,
so in-place edits die with the box. Same reasoning as patch_gate_config.py.

SAFE FOR THE TRAINED CHECKPOINT: these keys are used only on the SERVING path. Training reads
`dataset_key` (observation.rgb.*) from the parquet, which this does not touch. The observation
`resolution=[240, 240]` is also left alone deliberately -- the model was trained at 240, and
RGBDFullResWrapper renders at 720/480 and downsamples into it, which is exactly the
render-high/downsample path Comet measured 0.30 -> 0.60 on.
"""

import argparse
import pathlib
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/openpi_fork/src/openpi/configs/robots/b1k.py")
    ap.add_argument("--robot-name", default="robot_r1")
    a = ap.parse_args()

    p = pathlib.Path(a.config)
    src = p.read_text()
    orig = src

    # 1) the robot's own name -> drives f"{self.robot.name}::proprio"
    src = src.replace('name="robot",', f'name="{a.robot_name}",', 1)

    # 2) every observation key: "robot::robot:<link>:Camera:0::rgb"
    src = re.sub(r'obs_key="robot::robot:', f'obs_key="{a.robot_name}::{a.robot_name}:', src)

    if src == orig:
        raise SystemExit(f"no substitutions made in {p} — already patched, or the layout changed")

    p.write_text(src)

    n_name = len(re.findall(rf'name="{a.robot_name}",', src))
    n_obs = len(re.findall(rf'obs_key="{a.robot_name}::{a.robot_name}:', src))
    print(f"patched {p}")
    print(f"  robot name entries : {n_name}")
    print(f"  obs_key entries    : {n_obs}")
    for line in src.splitlines():
        if "obs_key=" in line or re.search(rf'name="{a.robot_name}"', line):
            print("   ", line.strip())


if __name__ == "__main__":
    main()
