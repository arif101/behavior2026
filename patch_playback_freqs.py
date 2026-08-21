"""Patch OmniGibson DataPlaybackWrapper.create_from_hdf5: honor OG_PLAYBACK_REAL_FREQS=1.

Stock behavior hard-codes action/rendering/physics frequencies to 1000 Hz during playback
(data_wrapper.py ~379) to minimize physics leakage between per-frame state loads. That is
correct for STATE playback but silently breaks POLICY-CONTROL-AFTER-RESTORE: each env.step
advances 1 ms of physics instead of the eval env's 33 ms (30 Hz actions x 120 Hz physics),
time-dilating all control ~33x and turning dynamic grasps into quasi-static pushes (measured:
demo-action replay shoved the radio off the table; ladder runs contaminated).

With OG_PLAYBACK_REAL_FREQS=1 the RECORDED frequencies are kept so post-restore dynamics match
the collection/eval env. Default behavior unchanged. Idempotent, compile-before-write.
"""

import os
import py_compile

TARGET = os.environ.get(
    "OG_DATA_WRAPPER", "/root/bw/BEHAVIOR-1K/OmniGibson/omnigibson/envs/data_wrapper.py"
)
MARKER = "OG_PLAYBACK_REAL_FREQS"

ANCHOR = """        if include_contacts:
            # Minimize physics leakage during playback (we need to take an env step when loading state)
            config["env"]["action_frequency"] = 1000.0
            config["env"]["rendering_frequency"] = 1000.0
            config["env"]["physics_frequency"] = 1000.0"""

REPLACEMENT = """        if include_contacts:
            import os as _os
            if _os.environ.get("OG_PLAYBACK_REAL_FREQS") == "1":
                # Policy-control-after-restore mode: keep RECORDED frequencies so post-restore
                # dynamics match the collection/eval env (1000 Hz stepping time-dilates control
                # ~33x and breaks contact maneuvers like grasps).
                pass
            else:
                # Minimize physics leakage during playback (we need to take an env step when loading state)
                config["env"]["action_frequency"] = 1000.0
                config["env"]["rendering_frequency"] = 1000.0
                config["env"]["physics_frequency"] = 1000.0"""


def main():
    s = open(TARGET).read()
    if MARKER in s:
        print("SKIP: already patched")
        return
    assert ANCHOR in s, "anchor not found in data_wrapper.py"
    assert s.count(ANCHOR) == 1, "anchor not unique"
    s2 = s.replace(ANCHOR, REPLACEMENT, 1)
    compile(s2, TARGET, "exec")
    open(TARGET, "w").write(s2)
    py_compile.compile(TARGET, doraise=True)
    print("PATCHED:", TARGET)


if __name__ == "__main__":
    main()
