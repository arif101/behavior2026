"""Patch /root/dart_round.sh (sim box) so the DART loop stops burning 40-min timeouts on demos whose scripted approach stalls
(2026-09-26: demos 50 and 60 stalled 0.45-0.76 m from the target on every tag, 4 x 40 min lost):
  - skip demos listed in /root/dart_skip_demos.txt
  - per-attempt timeout 2400 s -> 1800 s (successful attempts took 7-26 min)
  - a demo whose attempt produced neither OBS_SAVED nor RESULT twice is appended to the skip list automatically
Run on the box with NO dart_round.sh executing (bash reads scripts incrementally). Idempotent."""
import pathlib, shutil, time
p = pathlib.Path("/root/dart_round.sh"); s = p.read_text()
if "dart_skip_demos" in s:
    print("already patched"); raise SystemExit(0)
shutil.copy(p, f"/root/dart_round.sh.bak_{time.strftime('%m%d%H%M')}")
old = "for d in $DEMOS; do for pert in $PERTS; do tag=${pert##*,}\n"
assert s.count(old) == 1
s = s.replace(old, old + '  grep -qx "$d" /root/dart_skip_demos.txt 2>/dev/null && continue   # demos whose scripted approach stalls are skipped\n')
assert s.count("timeout 2400 $PY") == 1
s = s.replace("timeout 2400 $PY", "timeout 1800 $PY")
marker = '  rm -f /root/fap_tmp_$d.hdf5; n=$((n+1))\n'
assert s.count(marker) == 1
i = s.index(marker) + len(marker)
j = s.index("\n", i) + 1                      # end of the echo "d$d $tag ..." line that follows
s = s[:j] + ('  grep -qE "OBS_SAVED|RESULT d" /root/dart_logs/d${d}_${tag}.log || { echo $d >> /root/dart_fail_demos.txt; '
             '[ $(grep -cx "$d" /root/dart_fail_demos.txt) -ge 2 ] && ! grep -qx "$d" /root/dart_skip_demos.txt 2>/dev/null && '
             '{ echo $d >> /root/dart_skip_demos.txt; say "demo $d skipped after 2 failed attempts"; }; }\n') + s[j:]
p.write_text(s); print("dart_round.sh patched: skip list, 1800 s timeout, auto-skip after 2 failures")
