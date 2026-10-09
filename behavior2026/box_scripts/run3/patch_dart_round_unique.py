"""Patch /root/dart_round.sh so every conversion pushes to a UNIQUE HF path b1k_radio_dart_r<round>_<UTC stamp>.
Why (2026-09-26): dart_convert.sh does `rm -rf /root/b1k_radio_dart_r$ROUND` and upload_folder to the same path_in_repo,
so re-running a round (after a pause/restart) rebuilt the root from the renders present and would overwrite the round's
earlier push on HF (the r6 push of 14:33 was overwritten by the 23:21 restart; r8's 9 clips were saved by stopping the
convert before its push). Old versions remain in HF revision history. Run with no dart_round.sh executing. Idempotent."""
import pathlib, shutil, time
p = pathlib.Path("/root/dart_round.sh"); s = p.read_text()
if "RTAG=" in s:
    print("already patched"); raise SystemExit(0)
shutil.copy(p, f"/root/dart_round.sh.bak_{time.strftime('%m%d%H%M')}")
old = "PY=/root/miniconda3/envs/behavior/bin/python; ROUND=${ROUND:-1}\n"
assert s.count(old) == 1
s = s.replace(old, old + 'RTAG=${ROUND}_$(date -u +%m%d%H%M)   # unique per conversion: never overwrite an earlier push of the same round\n')
for a, b in [("BAR=strict ROUND=$ROUND bash /root/dart_convert.sh > /root/dart_convert_r$ROUND.out", "BAR=strict ROUND=$RTAG bash /root/dart_convert.sh > /root/dart_convert_r$RTAG.out"),
             ("/root/dart_convert_r$ROUND.out | tail -5", "/root/dart_convert_r$RTAG.out | tail -5"),
             ("if grep -q HF_PUSH_OK /root/dart_convert_r$ROUND.out; then", "if grep -q HF_PUSH_OK /root/dart_convert_r$RTAG.out; then"),
             ('say "round $ROUND converted+pushed, renders deleted"', 'say "round $ROUND converted+pushed as b1k_radio_dart_r$RTAG, renders deleted"')]:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
p.write_text(s); print("dart_round.sh patched: unique per-conversion HF path (RTAG)")
