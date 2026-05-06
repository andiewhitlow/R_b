import math
import random
import numpy as np
import awkward as ak


def addthrustvariables(events):
    """
    Fully vectorized thrust computation across all events.
    Assumes jets have already been selected (typically 2 jets per event).
    """
    # convert jagged jet arrays to numpy: shape (n_events, n_jets)
    # pad to max number of jets, fill missing with 0
    px = ak.to_numpy(ak.fill_none(ak.pad_none(events["Jets_px"], 2, clip=True), 0.0))
    py = ak.to_numpy(ak.fill_none(ak.pad_none(events["Jets_py"], 2, clip=True), 0.0))
    pz = ak.to_numpy(ak.fill_none(ak.pad_none(events["Jets_pz"], 2, clip=True), 0.0))

    # stack into shape (n_events, n_jets, 3)
    momenta = np.stack([px, py, pz], axis=2)

    # magnitudes: shape (n_events, n_jets)
    mags = np.linalg.norm(momenta, axis=2)

    # denominator: sum of |p| per event, shape (n_events,)
    denom = np.sum(mags, axis=1)
    denom = np.where(denom == 0, 1.0, denom)  # avoid division by zero

    # for 2-jet events, thrust axis is along jet 0 direction
    # try both +/- jet directions as seeds and pick best
    # seed axis: unit vector of jet 0, shape (n_events, 3)
    mag0 = mags[:, 0:1]  # shape (n_events, 1)
    safe_mag0 = np.where(mag0 == 0, 1.0, mag0)
    seed = momenta[:, 0, :] / safe_mag0  # shape (n_events, 3)

    best_thrust = np.zeros(len(events))
    best_axis = np.zeros((len(events), 3))

    for sign in [1.0]:
        axis = sign * seed  # shape (n_events, 3)

        # iterate hemisphere updates
        for _ in range(20):
            # dots: proper einsum for (n_events, n_jets, 3) x (n_events, 3)
            dots = np.sum(momenta * axis[:, None, :], axis=2)
            signs = np.where(dots >= 0, 1.0, -1.0)  # shape (n_events, n_jets)
            # signed momentum sum: shape (n_events, 3)
            vec = np.sum(signs[:, :, None] * momenta, axis=1)
            vmag = np.linalg.norm(vec, axis=1, keepdims=True)
            vmag_safe = np.where(vmag == 0, 1.0, vmag)
            new_axis = vec / vmag_safe
            # check convergence per event
            diff = np.linalg.norm(new_axis - axis, axis=1)
            axis = new_axis

        # compute thrust for this seed
        proj = np.sum(momenta * axis[:, None, :], axis=2)
        T = np.sum(np.abs(proj), axis=1) / denom

        # keep best
        improve = T > best_thrust
        best_thrust = np.where(improve, T, best_thrust)
        best_axis = np.where(improve[:, None], axis, best_axis)

    events = ak.with_field(events, ak.Array(best_thrust),       "Event_thrust")
    events = ak.with_field(events, ak.Array(best_axis[:, 0]),   "Event_thrust_x")
    events = ak.with_field(events, ak.Array(best_axis[:, 1]),   "Event_thrust_y")
    events = ak.with_field(events, ak.Array(best_axis[:, 2]),   "Event_thrust_z")
    events = ak.with_field(events, ak.Array(np.abs(best_axis[:, 2])), "Event_costhrust")
    return events