# Multimodal learning (part 7, application domains)

Learning across modalities at toy scale: vision-language models (contrastive image-text alignment on synthetic pairs, captioning a shape scene with a tiny decoder) and cross-modal retrieval (a shared embedding queried from either side), as sessions over `nn`, `text`, `vision` and the trainer base.

## Capabilities

- Contrastive alignment: images of shapes and their descriptions, the shared embedding forming, retrieval accuracy both ways on the tape.
- Captioning: a tiny decoder conditioned on an image embedding, the generated caption at the viewed step.
- Cross-modal retrieval: a query from either side, the ranked results as a probe.

## Examples to make

- CLIP-style training on shape scenes and their sentences: the embedding with both modalities, the pairs drawing together.
- Captioning a scene as training proceeds, the caption's tokens with their probabilities.

## Rendering

- Embedding steps with modality as marker, image grids with captions, ranking strips.
- Missing: an image-with-caption grid panel.

## Shared

- `shared/`: the paired data slot (scenes with generated sentences from `datasets`), the retrieval probes.

## Problems

- New chapter. Everything is synthetic and tiny; its value is the mechanism, and the pages say so.
