# NEST-N64

Minimal NEST simulation of the N = 64 reservoir (21 inputs → 64 recurrent neurons) on one
2 s population-coded window, with the weight matrices as deployed on the hardware. It
produces the input/recurrent raster and the list of recurrent spikes.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

NEST is on PyPI from version 3.10 (`nest-simulator`); older versions are available from
conda-forge.

## Run

```sh
python simulate.py
```

Results go to `results/`:

| file | content |
|---|---|
| `raster_W0000_dt<dt>.png` | input (blue) and recurrent raster in 2 ms bins, recurrent population rate per bin |
| `spikes_W0000_dt<dt>.csv` | every recurrent spike: `neuron`, `time_dt` (spike time on the simulation grid, ms), `time_2ms` (start of its 2 ms bin, ms), `count` (always 1) |

File names always carry the resolution (`_dt1`, `_dt0.1`, ...).

Options: `--dt` simulation resolution (ms, default 1), `--jin`, `--jres` other weight matrices, `--window <csv>` another input window,
`--T` window length (ms), `--bin` raster bin (ms), `--out_dir`.

## Inputs

| file | shape | content |
|---|---|---|
| `inputs/W0000.csv` | 1323 rows | input events `neuron,time_ms,count`: channel 0–20, time from the window start on a 2 ms grid |
| `inputs/Jin_hardware_exact_nA.csv` | 21 × 64 | input → recurrent |
| `inputs/Jres_hardware_exact_nA.csv` | 64 × 64 | recurrent → recurrent (already includes the 0.20 recurrent scaling, the projection onto the hardware levels and the measured calibration) |

Matrices are in nA, comma-separated, `W[source, destination]`: row = presynaptic input
channel or neuron, column = postsynaptic recurrent neuron. The channel index in the window
file is the row of the input matrix.

## Model

| | |
|---|---|
| neuron | `iaf_psc_exp`: C_m 1000 pF, τ_m 8 ms, E_L = V_reset −65 mV, V_th −50 mV, t_ref 0, I_e 0 |
| synapse | static, exponential current: a spike adds its weight to the synaptic current, which decays with τ_syn 2 ms (charge per event = weight × 2 ms, i.e. 2 pC for 1 nA) |
| delay | 1 ms on every input and recurrent connection |
| resolution | 1 ms by default (`--dt` to change it; the delay stays 1 ms). A neuron fires at most once per step, and spike times are on the step grid |
| reset | all neurons start at rest with zero synaptic current; after a spike only V_m is reset (to −65 mV), the synaptic current keeps decaying |
| recurrence | no self-connections; zero weights are not connected |
| input | the window events replayed by `spike_generator`s; an event at 0 ms is moved to the first step, `dt` (NEST cannot emit at t = 0) |

Weights are used as given, only converted nA → pA for NEST. No other scaling or
quantization is applied.
