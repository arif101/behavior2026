import base64, pathlib
A = pathlib.Path('/root/cmp_assets')
def b64(p):
    return base64.b64encode((A / p).read_bytes()).decode()
M = {k: b64(v) for k, v in {
    'v20': 'relay20.mp4', 'v100': 'relay100.mp4', 'v260': 'chain260.mp4',
    'approach': 'relay_approach.jpg', 'toggle': 'relay_toggle.jpg',
    'tilt': 'fail_tilt.jpg', 'down': 'fail_facedown.jpg', 'gone': 'fail_gone.jpg',
}.items()}

html = f'''<title>Press and Topple</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --bg: #F4F5F3; --surface: #FFFFFF; --ink: #0F1A1C; --muted: #566467;
  --line: #D8DCD9; --accent: #12656B; --good: #1E7A4A; --bad: #A63A22;
  --chip-good-bg: #E4F0E9; --chip-bad-bg: #F6E5E0; --shade: #EDEFEC;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0E1315; --surface: #161C1E; --ink: #E8EDEB; --muted: #94A3A4;
    --line: #2A3335; --accent: #4FB3B8; --good: #5BC189; --bad: #E08163;
    --chip-good-bg: #163024; --chip-bad-bg: #33201A; --shade: #1B2224;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0E1315; --surface: #161C1E; --ink: #E8EDEB; --muted: #94A3A4;
  --line: #2A3335; --accent: #4FB3B8; --good: #5BC189; --bad: #E08163;
  --chip-good-bg: #163024; --chip-bad-bg: #33201A; --shade: #1B2224;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--ink);
  font-family: "Source Serif 4", Georgia, serif; line-height: 1.55;
  margin: 0; padding: 40px 24px 72px;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; }}
h1, h2, h3, .lab {{ font-family: Archivo, "Helvetica Neue", Arial, sans-serif; }}
h1 {{ font-size: 2.1rem; font-weight: 700; letter-spacing: -0.015em; margin: 0 0 10px; text-wrap: balance; }}
.dek {{ font-size: 1.06rem; color: var(--muted); max-width: 64ch; margin: 0 0 22px; }}
.prov {{ font-family: "IBM Plex Mono", monospace; font-size: .74rem; color: var(--muted);
  letter-spacing: .02em; border-top: 1px solid var(--line); padding-top: 12px; margin-bottom: 34px; }}
.tracks {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 30px; }}
.track {{ display: flex; flex-direction: column; gap: 16px; }}
.head {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  border-bottom: 2px solid var(--edge); padding-bottom: 10px; }}
.track.win {{ --edge: var(--good); }}
.track.lose {{ --edge: var(--bad); }}
.head h2 {{ font-size: 1.22rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; }}
.chip {{ font-family: "IBM Plex Mono", monospace; font-size: .7rem; font-weight: 500;
  padding: 3px 9px; border-radius: 3px; letter-spacing: .03em; }}
.win .chip {{ background: var(--chip-good-bg); color: var(--good); }}
.lose .chip {{ background: var(--chip-bad-bg); color: var(--bad); }}
video {{ width: 100%; display: block; border: 1px solid var(--line); background: var(--shade); }}
.vcap {{ font-family: "IBM Plex Mono", monospace; font-size: .72rem; color: var(--muted); margin-top: -8px; }}
.strip {{ display: flex; flex-direction: column; gap: 14px; }}
figure {{ margin: 0; }}
figure img {{ width: 100%; display: block; border: 1px solid var(--line); }}
figcaption {{ font-family: "IBM Plex Mono", monospace; font-size: .73rem; line-height: 1.5;
  color: var(--muted); padding-top: 7px; }}
figcaption b {{ color: var(--ink); font-weight: 500; }}
.note {{ font-size: .95rem; border-left: 2px solid var(--edge); padding: 2px 0 2px 14px; color: var(--ink); }}
.lab {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); font-weight: 600; }}
.verdict {{ margin-top: 44px; border-top: 1px solid var(--line); padding-top: 22px; }}
.verdict p {{ max-width: 74ch; }}
.num {{ font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums;
  color: var(--accent); font-weight: 500; }}
a {{ color: var(--accent); }}
@media (max-width: 520px) {{ body {{ padding: 26px 16px 56px; }} h1 {{ font-size: 1.7rem; }} }}
</style>

<div class="wrap">
  <h1>Press and Topple</h1>
  <p class="dek">Two ways to turn on the radio, filmed. The trained policy presses the button
  while still holding it. The scripted chain sets it down first &mdash; and that is where it dies.</p>
  <p class="prov">turning_on_radio &middot; R1 Pro &middot; policy relay: radio_run2 &pi;0.5 from post-transport handoff
  &middot; scripted chain: rt_replay_test51 &middot; runs 2026-09-01</p>

  <div class="tracks">

    <section class="track win">
      <div class="head"><h2>The policy presses in hand</h2><span class="chip">TOGGLED &middot; weld intact</span></div>
      <video controls preload="metadata" src="data:video/mp4;base64,{M['v20']}"></video>
      <p class="vcap">d20 &middot; policy takes over at the handoff, toggles at step 214</p>
      <div class="strip">
        <figure>
          <img alt="Robot holds the radio in its right gripper while the left arm descends toward the button dome" src="data:image/jpeg;base64,{M['approach']}">
          <figcaption><b>step 208 &mdash; the free hand comes in.</b> Right gripper keeps the radio;
          the left fingertips close on the dome. Right hand stays parked at the grip, <b>0.167 m</b> from the button.</figcaption>
        </figure>
        <figure>
          <img alt="Left arm pressed onto the radio top at the moment the toggle fires" src="data:image/jpeg;base64,{M['toggle']}">
          <figcaption><b>d100, step 226 &mdash; contact.</b> Left fingers <b>0.068 m</b> from the button
          metalink, right hand <b>0.178 m</b> away. Only one hand is near it, and the radio never left the gripper.</figcaption>
        </figure>
      </div>
      <p class="note">No set-down, no release, nothing scripted after the handoff. The policy inherited
      this strategy from the human demonstrations, which press the same way &mdash; one hand holds, the other
      reaches over.</p>
    </section>

    <section class="track lose">
      <div class="head"><h2>The scripted chain topples it</h2><span class="chip">NOT TOGGLED &middot; d260</span></div>
      <video controls preload="metadata" src="data:video/mp4;base64,{M['v260']}"></video>
      <p class="vcap">d260 &middot; set-down completes, press servo never reaches the button</p>
      <div class="strip">
        <figure>
          <img alt="Radio held at a steep tilt as the gripper lowers it toward the glass table" src="data:image/jpeg;base64,{M['tilt']}">
          <figcaption><b>Release &mdash; the cause.</b> The weld froze whatever attitude the grasp produced,
          so the radio comes down tilted about <b>45&deg;</b> and touches on an edge, not its base.</figcaption>
        </figure>
        <figure>
          <img alt="Radio lying face-down on the table with its back panel and antenna facing up" src="data:image/jpeg;base64,{M['down']}">
          <figcaption><b>It lands face-down.</b> Back panel and antenna up, button pressed against the glass.
          Set-down gate: <b>upright &minus;0.05</b>, <b>z 0.469 m</b> &mdash; failed, and correctly logged as failed.</figcaption>
        </figure>
        <figure>
          <img alt="Empty table and floor strip; the gripper reaches at a target that is no longer there" src="data:image/jpeg;base64,{M['gone']}">
          <figcaption><b>Then it slides out of reach.</b> The camera aims between fingertip and button;
          the radio has dropped into the table&ndash;couch gap. Closest approach all run: <b>0.134 m</b>.</figcaption>
        </figure>
      </div>
      <p class="note">The press machinery was never the problem &mdash; it went 3-for-4 whenever the button was
      reachable. Every failure here happens before the press begins.</p>
    </section>

  </div>

  <div class="verdict">
    <p class="lab">What this decided</p>
    <p>Set-down exists only because early in-hand presses failed &mdash; and those attempts ran with a frozen
    torso and a button position retargeted from the demo instead of measured live, both since fixed. Then the
    policy pressed in hand on its own, three times out of three filmed runs, and the argument was over. The
    overnight episode factory now runs policy-only: roughly <span class="num">48%</span> of draws toggle
    (<span class="num">30</span> of <span class="num">62</span> at last count), and each success is saved as a
    complete training episode &mdash; a manufactured honest grasp followed by a learned press.</p>
  </div>
</div>
'''
out = pathlib.Path('/root/behavior2026/press_and_topple.html')
out.write_text(html)
print(f"{out} : {len(html)/1e6:.2f} MB")
