# Transfer and meta-learning (part 5, learning paradigms)

Learning across tasks and domains: fine-tuning and feature transfer, domain adaptation (importance weighting, feature alignment, adversarial alignment), multi-task and meta-learning (MAML, prototypical networks), active and Bayesian transfer, online transfer, and federated averaging across clients, as sessions over `nn`, `optim`, `learn` and the shift modifiers of `shift-and-drift`.

## Capabilities

- Fine-tuning: a pretrained small network's layers frozen or not as knobs, the target accuracy against the target size.
- Domain adaptation: source and target with a planted shift; importance weighting, CORAL, DANN's adversarial alignment, the two embeddings overlapping step by step.
- Multi-task: shared and task-specific parameters, the task weights as knobs, negative transfer shown.
- Meta-learning: MAML's inner and outer loops on sinusoid regression, the few-shot fit after k inner steps as a probe; prototypical networks' prototypes.
- Federated: clients with non-IID data, federated averaging rounds, client sampling, the global and local models' accuracies.

## Examples to make

- DANN on two shifted moons: the feature embedding of both domains aligning round by round.
- MAML on sinusoids: after meta-training, one task fitted in five gradient steps, drawn step by step.
- Federated averaging with the heterogeneity as the knob: the global model's accuracy and the clients' drift.

## Rendering

- Embedding steps with domain as marker, curves per task or client, the few-shot fit as a probe over the function.
- Missing: a per-client small-multiples panel.

## Shared

- `shared/`: the trainer base, the source-target data slot, the task distribution slot for meta-learning.

## Problems

- `transfer/` and `federated/` move here (named for consistency with the encyclopedia's chapter; federated as a section).
