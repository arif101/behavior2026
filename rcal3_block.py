        pairs = []
        qb0 = q61()[P["right"]["arm_qpos"]].copy()
        for jidx in (16, 18, 20):
            qs = q61()[P["right"]["arm_qpos"]].copy()
            es = poseR()[0].copy()
            probeC = holdS.copy()
            probeC[15:22] = qs
            probeC[jidx] += 0.05
            step_cmd(probeC)
            pairs.append((q61()[P["right"]["arm_qpos"]] - qs,
                          poseR()[0] - es))
            undoC = probeC.copy()
            undoC[15:22] = qs
            step_cmd(undoC)
        Jf0 = _np(rob.get_jacobian())
        bestC, bcC = None, -2.0
        for rowC in range(Jf0.shape[0]):
            for blkC in (0, 3):
                scores = []
                ok = True
                for dqC, deC in pairs:
                    predC = Jf0[rowC, blkC:blkC + 3, :][:, armR_idx] @ dqC
                    nC = np.linalg.norm(predC) * np.linalg.norm(deC)
                    if nC < 1e-12:
                        ok = False
                        break
                    cC = float(np.dot(predC, deC) / nC)
                    mC = min(np.linalg.norm(predC), np.linalg.norm(deC)) / (
                        max(np.linalg.norm(predC), np.linalg.norm(deC)) + 1e-12)
                    scores.append(cC * mC)
                if ok and min(scores) > bcC:
                    bcC, bestC = min(scores), (rowC, blkC)
        RSEL[0] = bestC
        rest = holdS.copy()
        rest[15:22] = qb0
        step_cmd(rest)
        print(f"RCAL3 sel={bestC} worst_score={bcC:.2f}", flush=True)
