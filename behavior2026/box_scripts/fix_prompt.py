"""Align the EVAL prompt with the TRAINING prompt.

Phase A trained with prompt_from_task=True, so the prompt was the dataset's task string --
the bare identifier "turning_on_radio". serve_b1k.py instead pulls a descriptive sentence from
TASK_REGISTRY ("Turn on the radio receiver that's on the table..."), which the policy never saw.
That put the language channel out of distribution for every rollout measured so far.
"""
import pathlib
import re

p = pathlib.Path("/root/openpi_adaln/src/openpi/configs/tasks/b1k.py")
s = p.read_text()
orig = s
s = re.sub(r'("turning_on_radio"\s*:\s*)"[^"]*"', r'\1"turning_on_radio"', s)
p.write_text(s)
print("patched:", s != orig)
