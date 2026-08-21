#!/bin/bash
# Extract an evenly-spaced strip of frames from an eval rollout video, for failure analysis.
# Usage: extract_frames.sh <video> <out_dir> [n_frames]
V="$1"; OUT="${2:-/root/frames}"; N="${3:-12}"
mkdir -p "$OUT"; rm -f "$OUT"/*.png
DUR=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$V" 2>/dev/null)
NF=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$V" 2>/dev/null)
echo "video=$V duration=${DUR}s frames=${NF}"
# sample N frames evenly across the whole rollout
ffmpeg -v error -i "$V" -vf "select='not(mod(n\,max(1,floor(${NF:-100}/$N))))',scale=480:-1" -vsync 0 "$OUT/f_%03d.png" 2>&1 | tail -2
ls "$OUT"/*.png 2>/dev/null | head -20
echo "extracted: $(ls "$OUT"/*.png 2>/dev/null | wc -l) frames"
