"""Simulate the N=64 reservoir in NEST on one input window and plot its raster.

Network: 21 input channels -> 64 recurrent LIF neurons (no readout).
  - neurons: iaf_psc_exp (leaky integrate-and-fire, exponential current synapses)
  - synapses: static, weights from two matrices in nA, W[source, destination]
  - delay: 1 ms on every connection; resolution --dt (default 1 ms, at most one spike
    per neuron per step)
  - initial state: all neurons at rest, zero synaptic current (one window, from scratch)

Outputs in --out_dir, named after the window and the resolution (e.g. W0000_dt1):
  raster_<window>_dt<dt>.png  input and recurrent raster (--bin ms bins) + population rate
  spikes_<window>_dt<dt>.csv  every recurrent spike: neuron, time_dt (spike time on the
                              simulation grid, ms), time_2ms (start of its --bin ms bin, ms),
                              count (always 1)

Usage:
    python simulate.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
INPUTS = HERE / 'inputs'


# iaf_psc_exp parameters of the recurrent neurons
NEURON = {
    'E_L': -65.0,        # mV, resting potential
    'V_reset': -65.0,    # mV, membrane reset after a spike (= E_L)
    'V_th': -50.0,       # mV, threshold (15 mV above rest)
    'V_m': -65.0,        # mV, initial membrane potential
    'C_m': 1000.0,       # pF
    'tau_m': 8.0,        # ms, membrane time constant
    'tau_syn_ex': 2.0,   # ms, decay of the synaptic current (positive weights)
    'tau_syn_in': 2.0,   # ms, decay of the synaptic current (negative weights)
    't_ref': 0.0,        # ms, no refractory period
    'I_e': 0.0,          # pA, no bias current
}
DELAY = 1.0              # ms, synaptic delay of every input and recurrent connection


def load_matrix(path):
    return np.loadtxt(path, delimiter=',', ndmin=2)


def load_window(path):
    """Input events (channel, time_ms, count) of one encoded window."""
    with open(path, newline='') as f:
        return [(int(r['neuron']), float(r['time_ms']), int(r['count']))
                for r in csv.DictReader(f)]


def simulate(J_in, J_rec, events, T, dt=1.0):
    """Run the network for one window; return recurrent spike (times_ms, neuron_index)."""
    import nest
    if hasattr(nest, 'VerbosityLevel'):                # NEST >= 3.10
        nest.verbosity = nest.VerbosityLevel.ERROR
    else:
        nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.resolution = dt                               # ms, integration step

    n_in, n_rec = J_in.shape

    # input channels: spike generators replaying the window. NEST cannot emit a spike at
    # t = 0, so events at 0 ms are moved to the first step (1 ms).
    spike_times = [[] for _ in range(n_in)]
    for ch, t, count in events:
        spike_times[ch].extend([max(t, dt)] * count)
    inputs = nest.Create('spike_generator', n_in)
    inputs.set([{'spike_times': sorted(ts)} for ts in spike_times])

    neurons = nest.Create('iaf_psc_exp', n_rec, params=NEURON)

    # connect each source to the destinations with a non-zero weight (nA -> pA);
    # no self-connections in the recurrent layer
    for sources, J, recurrent in ((inputs, J_in, False), (neurons, J_rec, True)):
        for i in range(J.shape[0]):
            targets = [j for j in np.flatnonzero(J[i]) if not (recurrent and j == i)]
            if targets:
                nest.Connect(sources[i], neurons[targets], 'all_to_all',
                             {'synapse_model': 'static_synapse', 'delay': DELAY,
                              'weight': (J[i, targets] * 1e3).reshape(-1, 1)})

    recorder = nest.Create('spike_recorder')
    nest.Connect(neurons, recorder)
    nest.Simulate(T + dt)          # one extra step so spikes at T - dt are recorded

    ev = recorder.get('events')
    times = np.asarray(ev['times'], dtype=float)
    index = np.asarray(ev['senders'], dtype=int) - neurons.tolist()[0]
    keep = times < T
    return times[keep], index[keep]


def bin_index(times, bin_ms):
    """Index of the bin each time falls in (the epsilon guards against float noise)."""
    return np.floor(np.asarray(times) / bin_ms + 1e-9).astype(int)


def bin_start(times, bin_ms):
    """Start time (ms) of the bin each time falls in."""
    return bin_index(times, bin_ms) * bin_ms


def bin_spikes(times, index, n, T, bin_ms):
    """Spike counts [neuron, bin]."""
    counts = np.zeros((n, int(np.ceil(T / bin_ms))))
    np.add.at(counts, (index, bin_index(times, bin_ms)), 1)
    return counts


def plot(B_in, B_rec, T, bin_ms, title, path):
    n_in, n_rec = B_in.shape[0], B_rec.shape[0]
    sec = T / 1e3
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 2.4, 0.8]))

    ax[0].imshow(B_in > 0, aspect='auto', interpolation='nearest',
                 cmap=ListedColormap(['white', 'tab:blue']),
                 extent=[0, T, n_in - 0.5, -0.5])
    ax[0].set_ylim(-2, n_in + 1)                       # margin around the edge channels
    ax[0].set_yticks(range(0, n_in, 5))
    ax[0].set_ylabel('input neuron')
    ax[0].set_title(f"input: {B_in.sum():.0f} spikes, {B_in.sum() / n_in / sec:.1f} Hz/neuron",
                    fontsize=10)

    silent = int((B_rec.sum(1) == 0).sum())
    ax[1].imshow(B_rec > 0, aspect='auto', interpolation='nearest', cmap='Greys',
                 extent=[0, T, n_rec - 0.5, -0.5])
    ax[1].invert_yaxis()                               # neuron 0 at the bottom
    ax[1].set_ylabel('recurrent neuron')
    ax[1].set_title(f"recurrent: {B_rec.sum():.0f} spikes, "
                    f"{B_rec.sum() / n_rec / sec:.1f} Hz/neuron, {silent} silent", fontsize=10)

    # population rate of the recurrent layer, one value per bin
    centers = (np.arange(B_rec.shape[1]) + 0.5) * bin_ms
    ax[2].plot(centers, B_rec.sum(0) / n_rec / (bin_ms / 1e3), lw=0.8, color='k')
    ax[2].set_ylabel('rate (Hz)')
    ax[2].set_xlabel('time (ms)')
    ax[2].set_xlim(0, T)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jin', default=str(INPUTS / 'Jin_hardware_exact_nA.csv'),
                   help="input -> recurrent weights, nA, 21 x 64")
    p.add_argument('--jres', default=str(INPUTS / 'Jres_hardware_exact_nA.csv'),
                   help="recurrent -> recurrent weights, nA, 64 x 64")
    p.add_argument('--window', default=str(INPUTS / 'W0000.csv'), help="encoded input window")
    p.add_argument('--T', type=float, default=2000.0, help="window length, ms")
    p.add_argument('--dt', type=float, default=1.0,
                   help="simulation resolution, ms (the delay stays %g ms)" % DELAY)
    p.add_argument('--bin', type=float, default=2.0, help="raster bin, ms")
    p.add_argument('--out_dir', default=str(HERE / 'results'))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    window = Path(args.window).stem
    events = load_window(args.window)

    J_in, J_rec = load_matrix(args.jin), load_matrix(args.jres)
    assert J_rec.shape == (J_in.shape[1],) * 2, "matrix shapes do not match"
    assert max(ch for ch, _, _ in events) < J_in.shape[0], "input channel beyond Jin rows"

    t, i = simulate(J_in, J_rec, events, args.T, args.dt)
    tag = f"{window}_dt{args.dt:g}"

    csv_path = out / f"spikes_{tag}.csv"
    order = np.lexsort((i, t))                         # by time, then neuron
    t_sorted = np.round(t[order], 6)                   # drop float noise of the step grid
    np.savetxt(csv_path, np.column_stack([i[order], t_sorted, bin_start(t_sorted, args.bin),
                                          np.ones(t.size)]),
               fmt=['%d', '%.10g', '%.10g', '%d'], delimiter=',', comments='',
               header=f"neuron,time_dt,time_{args.bin:g}ms,count")

    in_ch = np.repeat([ch for ch, _, _ in events], [c for _, _, c in events])
    in_t = np.repeat([t0 for _, t0, _ in events], [c for _, _, c in events])
    B_in = bin_spikes(in_t, in_ch, J_in.shape[0], args.T, args.bin)
    B_rec = bin_spikes(t, i, J_in.shape[1], args.T, args.bin)
    png_path = out / f"raster_{tag}.png"
    plot(B_in, B_rec, args.T, args.bin,
         f"N64 ({Path(args.jin).name} / {Path(args.jres).name}), window {window}, "
         f"dt {args.dt:g} ms, {args.bin:g} ms bins", png_path)
    print(f"max recurrent spikes per neuron per {args.bin:g} ms bin: {B_rec.max():.0f} "
          f"({int((B_rec > 1).sum())} bins with more than one)")
    print(f"{t.size} recurrent spikes ({t.size / J_in.shape[1] / (args.T / 1e3):.1f} Hz/neuron) "
          f"-> {png_path.name}, {csv_path.name}")

if __name__ == '__main__':
    main()
