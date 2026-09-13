"""Derive factory_approach_cap_v11.py from v10: BASE DRIVE-UP approach phase.
v10 covered the rig's gap with an arm+trunk servo (and, unintentionally, torso joint 4, which
the demos never move). v11 keeps joint 4 locked and covers the gap the way the humans did: the
holonomic base drives along the pull line to the radio's REST spot before the (unchanged)
staging/servo/closure/weld, and drives back to the demo's stance (radio in hand) before the
demo's transport replay. Anchored string edits on the v10 source; fails loudly if an anchor moved."""
import pathlib, sys

src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2])


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('OUT = "/root/factory_clips_approach"', 'OUT = "/root/factory_clips_approach"\nTARGET_GAP = 0.40      # hand->grasp-pose gap the arm+trunk (10-DOF) servo handles comfortably\nMAX_DRIVE = 1.60       # m; longest rig pull in the set is ~1.04 m (d10)\nBASE_CMD_MAX = 0.30    # |base command| cap (demo base commands span about +-0.33)\nSTANDOFF = 0.15        # m short of the full pull: the gated arm servo (STAGE/APPROACH) covers the rest')

# ---- reach gate -> base drive-up ------------------------------------------------------------
old_gate = '''    if gap0 > 0.55:
        # arm+trunk reach envelope (~0.5 m of approach): beyond it the demo needs base
        # motion first (d10: pull 1.04 m). Log and skip — a base-drive variant is the fix.
        json.dump(dict(demo=a.demo, t0=t_pre, closure=closure, K=a.K, gap0=round(gap0, 4),
                       pull=np.round(pull, 4).tolist(), ok=False, skip="REACH"),
                  open(f"{OUT}/d{a.demo:03d}_meta.json", "w"), indent=1)
        print(f"RESULT d{a.demo} SKIP_REACH gap0={gap0:.3f} pull={np.linalg.norm(pull):.3f}", flush=True)
        os._exit(0)

    q = q61()
    hold = np.asarray(acts[t_pre], np.float32).copy()
    hold[0:3] = 0.0
    hold[7:14] = q[P["left"]["arm_qpos"]]
    hold[15:22] = q[P["right"]["arm_qpos"]]
    hold[A_TORSO] = q[P["trunk_qpos"]]
    hold[6] = 0.0  # LOCK torso joint 4 at 0 (10-DOF demo convention)
    hold[22] = 1.0  # open
'''
new_gate = '''    q = q61()
    hold = np.asarray(acts[t_pre], np.float32).copy()
    hold[0:3] = 0.0
    hold[7:14] = q[P["left"]["arm_qpos"]]
    hold[15:22] = q[P["right"]["arm_qpos"]]
    hold[A_TORSO] = q[P["trunk_qpos"]]
    hold[6] = 0.0  # LOCK torso joint 4 at 0 (10-DOF demo convention)
    hold[22] = 1.0  # open

    # ---- phase A0 (v11): BASE DRIVE-UP along the pull line ---------------------------------
    # The rig moved the radio by `pull` into the hand; the honest counterpart is the base moving
    # by -pull (horizontal) so the demo's own closure posture lands the hand at the REST radio.
    # Holonomic base velocity commands (action dims 0:2), arm+trunk held, gripper open. The
    # command->displacement mapping is calibrated in place (BCAL), then a closed-loop drive.
    def base_xy_yaw():
        pB, qB = rob.get_position_orientation()
        pB = _np(pB).copy(); yaw = float(R.from_quat(_np(qB)).as_euler("xyz")[2])
        return pB, yaw

    def step_base(cmd):
        if CAP[0]:
            capture(cmd)
        wrapper.env.step(cmd)
        cmds_log.append(np.asarray(cmd, np.float32).copy())

    def drive_base(goal_xy, tag, v_max=BASE_CMD_MAX, tol=0.03, max_steps=900, watch_radio=None):
        """Closed-loop planar drive of the base to goal_xy (world). Returns (ok, dist_left, steps)."""
        blocked = 0
        for it in range(max_steps):
            pB, yaw = base_xy_yaw()
            err = goal_xy - pB[:2]; dist = float(np.linalg.norm(err))
            if dist < tol:
                print(f"{tag} reached it={it} dist={dist:.3f}", flush=True)
                return True, dist, it
            v_w = err / dist * min(1.0, dist / 0.15)                 # unit-ish, slows in the last 15 cm
            c, s = np.cos(yaw), np.sin(yaw)
            v_b = np.array([c * v_w[0] + s * v_w[1], -s * v_w[0] + c * v_w[1]])
            cmd = hold.copy(); cmd[0:2] = np.clip(BCAL_K @ v_b * v_max, -BASE_CMD_MAX, BASE_CMD_MAX); cmd[2] = 0.0
            step_base(cmd)
            if watch_radio is not None:
                _d = float(np.linalg.norm(radio_pose()[0] - watch_radio))
                HON["pre_disp"] = max(HON["pre_disp"], _d); HON["max_disp"] = max(HON["max_disp"], _d)
                if _d > 0.012:   # abort BELOW the 2 cm honesty certificate (d20 grazed to 2.13 cm and was rejected)
                    print(f"{tag} RADIO_TOUCHED it={it} disp={_d:.3f} dist={dist:.3f} -- stopping the drive", flush=True)
                    return False, dist, it
            moved = float(np.linalg.norm(base_xy_yaw()[0][:2] - pB[:2]))
            blocked = blocked + 1 if moved < 2e-4 else 0
            if blocked >= 40:
                print(f"{tag} BLOCKED it={it} dist={dist:.3f} (no base motion for 40 steps)", flush=True)
                return False, dist, it
            if it % 30 == 0:
                print(f"{tag} it={it} dist={dist:.3f} moved/step={moved:.4f} cmd={np.round(cmd[0:2], 3).tolist()}", flush=True)
        print(f"{tag} budget exhausted dist={dist:.3f}", flush=True)
        return False, dist, max_steps

    pB_demo, yaw_demo = base_xy_yaw()          # the demo's stance at t_pre (== closure stance)
    pull_h = np.array([pull[0], pull[1], 0.0])
    drive_dist = float(np.linalg.norm(pull_h))
    BASE = {"driven": False, "dist": 0.0, "steps": 0, "back_ok": None, "back_steps": 0, "gap_after": gap0}
    HON = {"pre_disp": 0.0, "max_disp": 0.0}   # honesty accumulator for the base phases (folded into phase A's HON)
    # BCAL: probe the two planar command channels for 8 steps each -> m/step per unit command (always)
    Kc = []
    for ch in (0, 1):
        p0, y0 = base_xy_yaw(); pr = hold.copy(); pr[ch] = 0.15
        for _ in range(8):
            wrapper.env.step(pr); cmds_log.append(pr.copy())
        p1, _ = base_xy_yaw(); dw = (p1[:2] - p0[:2]) / 8.0 / 0.15
        c, s = np.cos(y0), np.sin(y0)
        Kc.append(np.array([c * dw[0] + s * dw[1], -s * dw[0] + c * dw[1]]))  # base-frame m/step per unit
        for _ in range(8):
            wrapper.env.step(hold); cmds_log.append(hold.copy())
    Kmat = np.stack(Kc, 1)
    try:
        BCAL_K = np.linalg.inv(Kmat) * np.linalg.norm(Kmat, ord=2)
    except np.linalg.LinAlgError:
        BCAL_K = np.eye(2)
    print(f"BCAL base m/step per unit cmd: ch0 {np.round(Kc[0], 4).tolist()} ch1 {np.round(Kc[1], 4).tolist()}", flush=True)
    drive_base(pB_demo[:2], "BCAL_RESET", tol=0.01, max_steps=200)
    stage_p = tgt_p + 0.10 * stage_dir_w          # the approach corridor's staging point (as phase A)

    def hand_drive(target_hand_p, tag, max_steps=900):
        """Drive the base so the (held) hand lands at target_hand_p (planar); returns drive_base's result."""
        pB, _ = base_xy_yaw(); goal = pB[:2] + (np.asarray(target_hand_p)[:2] - poseR()[0][:2])
        return drive_base(goal, tag, watch_radio=pRest.copy(), max_steps=max_steps)

    grab("predrive")
    CAP[0] = True                                  # the base approach IS part of the manufactured clip
    # A0.1: the rig's pull, minus a standoff (d10: arriving on the grasp pose nudged the radio 4 cm)
    if gap0 > TARGET_GAP and 0.05 < drive_dist <= MAX_DRIVE:
        frac = max(0.0, 1.0 - STANDOFF / drive_dist) if drive_dist > STANDOFF + 0.05 else 0.0
        okd, left, nst = drive_base(pB_demo[:2] - pull_h[:2] * frac, "DRIVE", watch_radio=pRest.copy())
        BASE.update(driven=True, dist=drive_dist, steps=nst, drive_ok=bool(okd), dist_left=round(left, 3))
        print(f"DRIVE done ok={okd} dist={drive_dist:.3f} left={left:.3f} steps={nst} radio_disp={HON['pre_disp']:.4f} "
              f"hand->grasp-pose gap now {np.linalg.norm(tgt_p - poseR()[0]):.3f} m", flush=True)
    # A0.2: corridor RETREAT — back the whole robot 0.25 m along the approach corridor if the hand is
    # near the radio (d20: a straight staging sweep from 0.33 m knocked the radio over)
    if np.linalg.norm(poseR()[0] - tgt_p) < 0.45:
        okr, leftr, nr = hand_drive(poseR()[0] + 0.25 * stage_dir_w, "BASE_RETREAT", max_steps=400)
        BASE.update(retreat_ok=bool(okr), retreat_steps=nr)
    # A0.3: corridor ADVANCE — land the hand on the staging point; the arm then owns only the last 10 cm
    oka, lefta, na = hand_drive(stage_p, "BASE_STAGE", max_steps=600)
    BASE.update(driven=True, stage_ok=bool(oka), stage_steps=na, stage_left=round(lefta, 3))
    CAP[0] = False                                 # RCAL/TCAL probe jiggles are not part of the clip
    grab("drive_end")
    DRIVE_DISP = HON["pre_disp"]
    pE0, RE0 = poseR()
    gap0 = float(np.linalg.norm(tgt_p - pE0))
    BASE["gap_after"] = round(gap0, 4)
    print(f"BASE_STAGE done ok={oka} hand->stage_p {lefta:.3f} m hand->grasp-pose {gap0:.3f} m radio_disp={DRIVE_DISP:.4f}", flush=True)
    if gap0 > 0.55:
        json.dump(dict(demo=a.demo, t0=t_pre, closure=closure, K=a.K, gap0=round(gap0, 4),
                       pull=np.round(pull, 4).tolist(), base=BASE, ok=False, skip="REACH"),
                  open(f"{OUT}/d{a.demo:03d}_meta.json", "w"), indent=1)
        print(f"RESULT d{a.demo} SKIP_REACH gap0={gap0:.3f} pull={np.linalg.norm(pull):.3f} base={BASE}", flush=True)
        os._exit(0)
    q = q61()
    hold[7:14] = q[P["left"]["arm_qpos"]]
    hold[15:22] = q[P["right"]["arm_qpos"]]
    hold[A_TORSO] = q[P["trunk_qpos"]]
    hold[6] = 0.0
'''
rep(old_gate, new_gate)

# honesty accumulator: seed pre_disp with the drive's radio displacement
rep('''    HON = {"radio_rest": radio_pose()[0].copy(), "R_rest": radio_pose()[1], "max_disp": 0.0, "pre_disp": 0.0,
           "pre_rot": 0.0, "first_contact": None, "n": 0}''',
    '''    HON = {"radio_rest": radio_pose()[0].copy(), "R_rest": radio_pose()[1], "max_disp": DRIVE_DISP, "pre_disp": DRIVE_DISP,
           "pre_rot": 0.0, "first_contact": None, "n": 0}''')

# ---- phase C: carry -> drive back to the demo stance (radio in hand), then residual servo ----
old_carry = '''    pE, RE = poseR()
    carry_tgt = pE + pull  # hand displacement == the pull vector
    ok_c, best_c = servo(carry_tgt, RE, 200, "CARRY", stride=0.010)
'''
new_carry = '''    pE, RE = poseR()
    if BASE["driven"]:
        # v11: the base carries the held radio back to the demo's stance (the honest mirror of the
        # drive-up); the arm then closes the residual to the demo's post-pull radio pose.
        okb, leftb, nb = drive_base(pB_demo[:2], "DRIVE_BACK", tol=0.02, max_steps=900)
        BASE.update(back_ok=bool(okb), back_steps=nb, back_left=round(leftb, 3))
        print(f"DRIVE_BACK ok={okb} left={leftb:.3f} steps={nb} ag={native_ag()}", flush=True)
        if not native_ag():
            print(f"RESULT d{a.demo} AG_LOST_DRIVE_BACK", flush=True)
            os._exit(0)
        pE, RE = poseR()
        carry_tgt = pE + (p_rad_post - radio_pose()[0])
    else:
        carry_tgt = pE + pull  # hand displacement == the pull vector
    ok_c, best_c = servo(carry_tgt, RE, 200, "CARRY", stride=0.010)
'''
rep(old_carry, new_carry)

# ---- meta: record the base phase ----
rep('''                ag_intact=bool(native_ag()), ag_lost=ag_lost, ok=ok,
                n_cmds=len(cmds_log), kind="approach",''',
    '''                ag_intact=bool(native_ag()), ag_lost=ag_lost, ok=ok, base=BASE,
                n_cmds=len(cmds_log), kind="approach_v11_basedrive",''')
rep('"episode": "approach_single_pass"', '"episode": "approach_v11_basedrive"')

# ---- TRUE 10-DOF servo: with joint 4 locked, the Jacobian must not contain it (v10's lock zeroed the
# command AFTER an 11-DOF solve -> the solver kept assigning motion to a frozen joint; stalls/divergence)
rep('    for jch in range(3, 7):\n', '    for jch in range(3, 6):   # trunk joints 1-3 only (joint 4 locked)\n')
rep('''            cmd[15:22] = q_[P["right"]["arm_qpos"]] + dq[:7]
            cmd[A_TORSO] = q_[P["trunk_qpos"]] + dq[7:]
            cmd[6] = 0.0  # LOCK torso joint 4 at 0 (10-DOF demo convention; A5 joint-4 degenerate-std fix)
''', '''            cmd[15:22] = q_[P["right"]["arm_qpos"]] + dq[:7]
            cmd[3:6] = q_[P["trunk_qpos"]][:3] + dq[7:10]
            cmd[6] = 0.0  # LOCK torso joint 4 at 0 (10-DOF demo convention)
''')
rep('''        cmd[15:22] = q_[P["right"]["arm_qpos"]] + dq[:7]; cmd[A_TORSO] = q_[P["trunk_qpos"]] + dq[7:]
        cmd[6] = 0.0  # LOCK torso joint 4 (10-DOF)
''', '''        cmd[15:22] = q_[P["right"]["arm_qpos"]] + dq[:7]; cmd[3:6] = q_[P["trunk_qpos"]][:3] + dq[7:10]
        cmd[6] = 0.0  # LOCK torso joint 4 (10-DOF)
''')
rep('''        pred = Jp[:, 7:] @ dqt
''', '''        pred = Jp[:, 7:10] @ dqt[:3]
''')
rep('''        cmd[A_TORSO] = q_[P["trunk_qpos"]] + dqt
        cmd[6] = 0.0  # LOCK torso joint 4 at 0 (10-DOF demo convention; A5 joint-4 degenerate-std fix)
''', '''        cmd[3:6] = q_[P["trunk_qpos"]][:3] + dqt[:3]
        cmd[6] = 0.0  # LOCK torso joint 4 at 0 (10-DOF demo convention)
''')

out.write_text(src)
compile(src, str(out), "exec")
print("v11 written:", out, "lines", src.count("\\n"))
