"""Pure selection: no file access, target labels, or generator identities."""
import numpy as np

def select(y, raw_a, raw_b, mean, std):
    a = np.asarray(raw_a)[:, :2].sum(1)
    b = np.asarray(raw_b)[:, :2].sum(1)
    z = np.asarray(raw_b)[:, 2:4]
    q = z * np.asarray(std) + np.asarray(mean)
    scores = z[:, 0] - z[:, 1]
    out = {'first10': dict(selected=list(range(10)), fallback=False)}
    for name, values in [('A-context-temporal', a), ('B-context-temporal', b), ('B-quality-CD', scores)]:
        out[name] = dict(selected=np.argsort(-values, kind='stable')[:10].tolist(), fallback=False)
    flat = np.asarray(y, dtype=np.float64).reshape(16, -1)
    # Mean squared distance: same native-channel scaling as S-MSE.
    norm = (flat * flat).mean(1)
    dist = np.maximum(norm[:, None] + norm[None, :] - 2 * flat @ flat.T / flat.shape[1], 0)
    for name, thresholds in [('train', np.asarray(mean)), ('pool-first10', q[:10].mean(0))]:
        gate = (q[:, 0] >= thresholds[0]) & (q[:, 1] <= thresholds[1])
        eligible = np.flatnonzero(gate).tolist()
        chosen = []
        while eligible and len(chosen) < 10:
            # Deterministic quality seed, then maximum marginal pairwise diversity.
            j = max(eligible, key=lambda i: (scores[i], -i)) if not chosen else max(eligible, key=lambda i: (dist[i, chosen].mean(), -i))
            chosen.append(j); eligible.remove(j)
        accepted = list(chosen)
        for i in np.argsort(-scores, kind='stable').tolist():
            if len(chosen) == 10: break
            if i not in chosen: chosen.append(i)
        out['vector-' + name] = dict(selected=chosen, fallback=len(accepted)<10,
            fallback_slots=10-len(accepted), gate_accepted=accepted, predicted_gate=gate.tolist(),
            C_threshold=float(thresholds[0]), logD_threshold=float(thresholds[1]))
    return out
