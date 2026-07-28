"""AABB geometric predicate evaluation (v4 fix).

The OmniGibson object_states path (OnTop/Inside/NextTo via get_value) returned
all-zero for kinematic relations under playback -- those states lean on contact
data that the replay does not populate (include_contacts=False), while joint/
logical states like ToggledOn worked. This module evaluates kinematic relations
directly from object world AABBs (center + half-extent), the spec-sanctioned
fallback (STAGE_HEAD_OWNERSHIP gotcha #3: nearest-point, not centers).

All functions take AABBs as (center[3], half_extent[3]) numpy arrays in world
frame and return bool. Thresholds are module constants, tuned against a known-
achieved demo frame (see validate step). Distances in meters.
"""
import numpy as np

# --- tunables (validated against achieved-goal frames) ---
ONTOP_Z_GAP = 0.10      # max gap between A-bottom and B-top to count as "on"
ONTOP_XY_MARGIN = 0.05  # A's XY center may sit this far outside B's XY AABB
NEXTTO_XY_GAP = 0.20    # max horizontal nearest-point gap for "next to"
TOUCH_GAP = 0.03        # 3D nearest-point gap to count as "touching"
INSIDE_FRAC = 0.5       # min fraction of A's center-containment for "inside"


def _lo(c, e):
    return np.asarray(c) - np.asarray(e)


def _hi(c, e):
    return np.asarray(c) + np.asarray(e)


def _axis_gap(cA, eA, cB, eB):
    """Per-axis nearest-point gap between two AABBs (0 where they overlap)."""
    return np.maximum(np.abs(np.asarray(cA) - np.asarray(cB))
                      - (np.asarray(eA) + np.asarray(eB)), 0.0)


def _xy_overlap(cA, eA, cB, eB, margin=0.0):
    g = _axis_gap(cA, eA, cB, eB)
    return bool(g[0] <= margin and g[1] <= margin)


def ontop(cA, eA, cB, eB):
    """A resting on top of B: XY overlap, A above B, A-bottom near B-top."""
    if not _xy_overlap(cA, eA, cB, eB, ONTOP_XY_MARGIN):
        return False
    a_bottom = _lo(cA, eA)[2]
    b_top = _hi(cB, eB)[2]
    return bool(cA[2] > cB[2] and -ONTOP_Z_GAP <= (a_bottom - b_top) <= ONTOP_Z_GAP)


def inside(cA, eA, cB, eB):
    """A inside B: A's center within B's AABB and A smaller than B on each axis
    (a loose containment; particle/whole-containment is stricter than needed
    for a satisfaction flip)."""
    lo, hi = _lo(cB, eB), _hi(cB, eB)
    c = np.asarray(cA)
    center_in = bool(np.all(c >= lo) and np.all(c <= hi))
    smaller = bool(np.all(np.asarray(eA) <= np.asarray(eB) + 1e-6))
    return center_in and smaller


def nextto(cA, eA, cB, eB):
    """A beside B: small horizontal gap and vertical overlap."""
    g = _axis_gap(cA, eA, cB, eB)
    xy_gap = float(np.hypot(g[0], g[1]))
    z_overlap = g[2] <= 0.10
    return bool(xy_gap <= NEXTTO_XY_GAP and z_overlap)


def under(cA, eA, cB, eB):
    """A under B: XY overlap and A below B."""
    return bool(_xy_overlap(cA, eA, cB, eB, ONTOP_XY_MARGIN) and cA[2] < cB[2])


def touching(cA, eA, cB, eB):
    """3D nearest-point gap ~ 0."""
    return bool(np.linalg.norm(_axis_gap(cA, eA, cB, eB)) <= TOUCH_GAP)


# taxonomy predicate name -> evaluator. overlaid/draped approximated by ontop.
KINEMATIC = {
    "ontop": ontop, "on": ontop, "overlaid": ontop, "draped": ontop,
    "inside": inside, "under": under, "nextto": nextto, "touching": touching,
}


def is_kinematic(pred):
    return pred in KINEMATIC


def evaluate(pred, cA, eA, cB, eB):
    """Kinematic predicate by name; raises if not kinematic (caller checks)."""
    return KINEMATIC[pred](cA, eA, cB, eB)


if __name__ == "__main__":
    import numpy as np
    # shoe (0.15x0.08x0.05) resting on a rack shelf top at z=0.4
    rack_c, rack_e = np.array([1.0, 0.5, 0.30]), np.array([0.25, 0.20, 0.10])
    shelf_top = rack_c[2] + rack_e[2]  # 0.40
    shoe_e = np.array([0.15, 0.08, 0.05])
    shoe_on = np.array([1.0, 0.5, shelf_top + shoe_e[2]])   # sitting on shelf
    shoe_off = np.array([2.0, 0.5, 0.05])                   # on the floor away
    assert ontop(shoe_on, shoe_e, rack_c, rack_e) is True, "on-rack should be True"
    assert ontop(shoe_off, shoe_e, rack_c, rack_e) is False, "floor should be False"
    # trash inside a bin
    bin_c, bin_e = np.array([0.0, 0.0, 0.3]), np.array([0.2, 0.2, 0.3])
    trash_in = np.array([0.0, 0.0, 0.3]); trash_e = np.array([0.05, 0.05, 0.05])
    trash_out = np.array([1.0, 0.0, 0.05])
    assert inside(trash_in, trash_e, bin_c, bin_e) is True
    assert inside(trash_out, trash_e, bin_c, bin_e) is False
    # nextto
    a = np.array([0.0, 0.0, 0.2]); ae = np.array([0.1, 0.1, 0.2])
    b_near = np.array([0.35, 0.0, 0.2]); be = np.array([0.1, 0.1, 0.2])
    b_far = np.array([2.0, 0.0, 0.2])
    assert nextto(a, ae, b_near, be) is True
    assert nextto(a, ae, b_far, be) is False
    print("AABB_PREDICATES_SELFTEST_OK")
