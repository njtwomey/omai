# Machine learning engineering (part 8, practice)

Training and serving mechanics simulated at small scale: training at scale (data and model parallelism simulated on a tiny model, gradient accumulation, mixed precision's effect on a loss, checkpointing), efficient inference (quantisation's error, pruning's sparsity against accuracy, distillation from a larger toy model, caching in decoding), and deployment and monitoring (drift monitors on a stream, canary comparisons), as sessions over `nn`, `optim` and the stream base.

## Capabilities

- Parallelism simulated: the same steps split across simulated workers, the synchronisation and the straggler effect on throughput.
- Precision: the loss and gradients in reduced precision against full, the loss scale as a knob.
- Quantisation: a trained network quantised to k bits with the accuracy and the weight histograms as k changes; pruning by magnitude step by step, the sparsity against the accuracy.
- Distillation: a student trained on a teacher's soft labels, the temperature as a knob.
- Monitoring: drift and performance monitors on a stream (with `shift-and-drift`).

## Examples to make

- Magnitude pruning one round at a time, the accuracy and the weight histogram.
- Quantisation to 8, 4 and 2 bits of the same network, the decision surfaces side by side.
- Distillation with the temperature as the knob, the student's boundary approaching the teacher's.
- Gradient accumulation against a larger batch: the same trajectory, the memory as a bar.

## Rendering

- Histograms, decision surfaces side by side, throughput bars, curves.
- Missing: the comparison panel.

## Shared

- `shared/`: the trainer base; the stream base.

## Problems

- New chapter, small by design; the engineering is simulated, which the pages say.
