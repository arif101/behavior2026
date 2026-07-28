"""Add the `episodes` filter to pi05_radio_gate's dataset_kwargs.

Without it LeRobot reads meta/info.json — which describes all 20,000 episodes of the full
challenge set, because meta/ is copied wholesale from the source mirror — cannot reconcile that
against the 2 radio shards actually present locally, and falls back to a Hub lookup. That surfaces
as a misleading `RepositoryNotFoundError: 401 ... repo b1k_radio`, i.e. a local-load failure
wearing an auth error's clothes. The G3 configs used exactly this filter for the same reason
(dataset_kwargs={"tolerance_s": 5e-4, "episodes": G3_EPISODES}).

turning_on_radio is task 0, so its episode_index values are 0..199 (verified against the joined
parquets).
"""

import ast
import pathlib

P = pathlib.Path("/root/openpi/src/openpi/training/config.py")
OLD = '''                dataset_root="/root/b1k_radio",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},'''
NEW = '''                dataset_root="/root/b1k_radio",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4, "episodes": list(range(200))},'''

s = P.read_text()
if '"episodes": list(range(200))' in s:
    print("episodes filter already present")
else:
    assert OLD in s, "anchor not found — pi05_radio_gate layout changed"
    s = s.replace(OLD, NEW, 1)
    ast.parse(s)
    P.write_text(s)
    print("patched: episodes filter added to pi05_radio_gate")
