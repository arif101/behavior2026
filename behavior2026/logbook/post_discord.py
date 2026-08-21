#!/usr/bin/env python3
"""Post a B26 status update to the team Discord webhook.

Usage: python3 post_discord.py <file.md>   (or pipe text on stdin)
Splits on '\n---\n' into separate messages (Discord 2000-char limit per message).
Format convention (per Arif, 2026-08-18): (1) simple explanation with an NFL/NBA
analogy; (2) PhD-level analysis tied to goals + architecture; (3) next-step bullets;
(4) explicit "HUMAN INPUT NEEDED" section if any.
"""

import json
import sys
import urllib.request

WEBHOOK = open("/root/.discord_webhook").read().strip()


def post(text):
    for chunk in text.split("\n---\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        while chunk:
            part, chunk = chunk[:1990], chunk[1990:]
            req = urllib.request.Request(
                WEBHOOK, data=json.dumps({"content": part}).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "b26-logbook/1.0"})
            with urllib.request.urlopen(req) as r:
                if r.status not in (200, 204):
                    print(f"WARN status={r.status}", file=sys.stderr)


if __name__ == "__main__":
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    post(src)
    print("posted")
