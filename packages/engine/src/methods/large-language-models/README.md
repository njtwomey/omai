# Large language models (part 6, neural architectures)

Language models at toy scale and as black boxes: n-gram and small neural language models with decoding (greedy, beam, sampling, temperature, nucleus), pretraining and scaling curves at tiny sizes, working with a model (prompting, in-context learning on synthetic tasks, retrieval augmentation), adaptation (fine-tuning, LoRA, preference optimisation on a toy reward), reasoning and test-time compute (sampling many, verifying), agents (a loop of calls and tools, simulated), and decision models, as sessions over `nn`, `text`, `transformers`' pieces and simulated models where a real one is out of reach.

## Capabilities

- n-gram models: counts, smoothing and perplexity on a small corpus, generation token by token.
- A tiny transformer language model trained (with `transformers`): perplexity on the tape, samples at the viewed step.
- Decoding as a slot: every strategy with its knobs on the same model, the chosen token's probability drawn.
- Scaling: loss against parameters and data at toy sizes on a log-log axis.
- Adaptation: LoRA's low-rank update, a preference-based objective (DPO) on a planted preference.
- Agents: a simulated model calling tools, the loop's state as a sequence of steps.

## Examples to make

- Nucleus sampling's cut on the next-token distribution, p and temperature as knobs.
- A bigram model generating, with the counts table as a probe.
- A toy scaling law: three model sizes, three data sizes, the loss surface.
- DPO on a toy preference: the policy's probabilities moving towards the preferred completions.

## Rendering

- Token strips with probabilities, bars over a vocabulary, log-log curves, step lists for agents.
- Missing: a chat-like transcript panel for agent loops; the token-probability strip.

## Shared

- `shared/`: the corpus slot (tiny corpora in `datasets`), the decoding slot table, the simulated-model stub for pages that cannot run a real model.

## Problems

- `llm/` moves here. Real models are out of scope for the engine; the pages are about mechanisms at toy scale, and the encyclopedia says so.
