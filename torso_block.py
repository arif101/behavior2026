    # phase 3B: 11-DOF TORSO+ARM aimed press on the resting radio
    if True:
        from omnigibson.object_states import ToggledOn as _TO2

        def rf2():
            return np.mean([_np(l.get_position_orientation()[0])
                            for l in rob.finger_links["right"]], axis=0)

        def btn2():
            return _np(radio.states[_TO2].link.get_position_orientation()[0])

        pre_toggled = toggled()
        print(f"PRESS start: pre_toggled={pre_toggled}", flush=True)
        qf2 = q61()
        holdB = np.asarray(acts[press0], np.float32).copy()
        holdB[0:3] = 0.0
        holdB[7:14] = qf2[P["left"]["arm_qpos"]]
        holdB[A_TORSO] = qf2[P["trunk_qpos"]]
        holdB[22] = -1.0

        # probe the 4 trunk action channels empirically (3x4 block)
        Tcols = []
        for jch in range(3, 7):
            qs = q61()
            e0 = rf2()
            pr = holdB.copy()
            pr[15:22] = qs[P["right"]["arm_qpos"]]
            pr[A_TORSO] = qs[P["trunk_qpos"]]
            pr[jch] = pr[jch] + 0.05
            step_cmd(pr)
            Tcols.append((rf2() - e0) / 0.05)
            un = pr.copy()
            un[jch] = un[jch] - 0.05
            step_cmd(un)
        Jt = np.stack(Tcols, axis=1)
        print(f"TCAL trunk column norms: {[round(float(np.linalg.norm(c)),3) for c in Tcols]}",
              flush=True)

        def fetch_J11():
            Ja = fetch_JR()
            return np.concatenate([Ja, Jt], axis=1)

        wp2 = 0
        best_btn = 9.9
        for it in range(300):
            bp = btn2()
            rfn = rf2()
            d_btn = float(np.linalg.norm(bp - rfn))
            best_btn = min(best_btn, d_btn)
            targets = [bp + np.array([0.0, 0.0, 0.06]),
                       bp + np.array([0.0, 0.0, -0.012])]
            target = targets[wp2]
            dtn = float(np.linalg.norm(target - rfn))
            if dtn < 0.02 and wp2 < 1:
                wp2 += 1
                print(f"TBTN waypoint -> {wp2} at it={it}", flush=True)
                continue
            if toggled() and not pre_toggled:
                tgl_frame = -4
                print(f"TOGGLED by aimed torso press at it={it}", flush=True)
                break
            e_b = rfn
            dq = np.clip(np.linalg.pinv(fetch_J11())
                         @ ((target - rfn) / (dtn + 1e-9)
                            * min(0.015, dtn)), -0.09, 0.09)
            q_ = q61()
            cmd = holdB.copy()
            cmd[15:22] = q_[P["right"]["arm_qpos"]] + dq[:7]
            cmd[A_TORSO] = q_[P["trunk_qpos"]] + np.clip(dq[7:11], -0.05, 0.05)
            step_cmd(cmd)
            moved2 = rf2() - e_b
            cos2 = float(np.dot(moved2, target - e_b) /
                         (np.linalg.norm(moved2)
                          * np.linalg.norm(target - e_b) + 1e-9))
            if it % 10 == 0:
                print(f"TBTN it={it} wp={wp2} dist={dtn:.4f} btn={d_btn:.4f} "
                      f"cos={cos2:.2f} trunk_dq={np.linalg.norm(dq[7:11]):.3f} "
                      f"tg={toggled()}", flush=True)
                img = grab(0.5 * (rf2() + btn2()))
                frames.append(img)
                imageio.imwrite(f"{OUT}/tb{it:03d}.png", img)
        print(f"TBTN done: best_btn={best_btn:.4f} toggled={toggled()} "
              f"pre={pre_toggled}", flush=True)
        if toggled() and tgl_frame is None:
            tgl_frame = -4

