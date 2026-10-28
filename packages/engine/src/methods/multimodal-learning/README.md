# Multimodal learning (part 7, application domains)

Learning across modalities at toy scale: contrastive image-text alignment (CLIP-style) on shape scenes and their template sentences, zero-shot classification and cross-modal retrieval over the joint embedding, captioning and visual question answering with a tiny decoder, text-conditioned generation, fusion strategies (early, late, intermediate, co-attention) with a modality missing, and the analysis of a joint space (the modality gap, cross-attention maps, probes), as sessions over `nn`, `losses` (contrastive), `text`, `metrics` (ranking, text, CLIPScore), the image generators of `datasets` and the trainer base. Every model here is a mechanism shown on synthetic pairs, not a real vision-language model, and every page says so.

## Structure

Everything here trains a network on a problem (a joint embedding, a caption, an answer), so it lives under `train/multimodal/` (section 3 of `methods/README.md`), one folder per kind of objective. The image side reuses `train/networks/` blocks and the text side `process/text/` tokenisers; the retrieval index and the ranking metrics are `recommendation-and-retrieval`'s, the representation-quality measures `representation-learning`'s.

```
train/multimodal/
  alignment/            contrastive alignment with the temperature and the batch as knobs, zero-shot classification
  generation/           captioning with the conditioning as a mode, visual question answering, text-conditioned generation
  retrieval/            cross-modal retrieval over the joint embedding, from either side
  fusion/               early, late and intermediate fusion, co-attention, a modality missing
  analysis/             the modality gap, cross-attention maps, probes per modality
  shared/               the paired-data slot (scenes with sentences and questions), the encoders, the retrieval probes (below)
```

Nothing of this chapter is built; the paired data it needs (scenes with sentences from a template grammar) is the first library piece.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of every training here is one minibatch (the trainer base); the step of a retrieval or an analysis is one query or one probe. Every method logs `training/loss` and the held-out metrics at a schedule (`eval/<metric>`), and answers `model/params` and `model/embedding` (the joint space with the modality as marker).

### alignment/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `contrastive-alignment` | A tiny image encoder (patches through a small transformer, or a convolution stack) and a tiny text encoder (a bag of token embeddings, or a small transformer) trained with the symmetric InfoNCE over batches of scene-sentence pairs: the joint embedding forming (both modalities in one projection, the pairs drawing together), retrieval accuracy both ways on the tape, the temperature (learned or fixed) and the batch size as knobs (the number of negatives), the similarity matrix of a batch as a probe (the diagonal brightening); the encoders as modes; the attribute the sentences name (shape, colour, count, position) as the planted truth the embedding must separate. A mechanism, not CLIP. | `training/loss` s, `training/temperature` s, `eval/recallAtK` s (image to text and text to image), `eval/zeroShotAccuracy` s, `model/similarity` p (the batch's matrix), `model/embedding` p, `model/params` p, `model/attribute` p (the embedding coloured by each planted attribute) | `losses/contrastive.ts`: `clipLoss`, `infoNce`; `nn/functional/vision.ts`: `patchEmbedding`; `nn/module/vision.ts`: `PatchEmbedding`; `nn/blocks/transformer.ts`: `TransformerStack`; `nn/module/layers.ts`: `Embedding`, `EmbeddingBag`; `nn/module/conv.ts` (the convolution stack); `nn/training/fit.ts`: `fitAlgorithm`; `metrics/ranking.ts`: `recallAtK`; `datasets/images.ts`: `shapes` (labels, boxes, masks as the attributes); TODO `datasets/captions.ts`: `captionedShapes` (scenes with sentences from a template grammar and questions with answers, the attributes as truth) and a paired schema in `data/schema/` | |
| `zero-shot-classification` | Class names as prompts through the text encoder, an image classified by its nearest prompt embedding; the prompt template as the knob ("a circle" against "a picture of a circle"), accuracy per class against a linear probe trained on the same embeddings (the probe wins with labels, the prompt needs none); the confusion between classes the sentences describe alike. | `eval/zeroShotAccuracy` s, `eval/probeAccuracy` s, `model/promptEmbeddings` t!, `model/confusion` p, `model/embedding` p | the aligned model above; `explain/concepts.ts`: `linearProbe`; `metrics/classification.ts`: `confusionMatrix`, `accuracyScore` | |

### generation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `captioning` | A decoder conditioned on the image: the conditioning as a mode (the image embedding as a prefix token, cross-attention over the patch embeddings), teacher forcing during training, the caption generated at the viewed step with per-token probabilities (the decoding as a slot: greedy, beam, sampling with temperature), BLEU and ROUGE against the template sentence, the exposure bias when sampling is used in training as the comparison; the cross-attention per generated word on the image (in `analysis/`). A mechanism at toy scale. | `training/loss` s, `eval/bleu` s, `eval/rougeL` s, `model/caption` p (a `tokens` record with probabilities), `model/params` p, `decode/token` m (the beam or sample per step at decode time) | `nn/blocks/transformer.ts`: `EncoderDecoder`, `TransformerLM`; `nn/module/transformer.ts`: `TransformerDecoder`; `nn/decoding/generate.ts`: `decodingAlgorithm`; `nn/decoding/beam.ts`: `beamSearchAlgorithm`; `nn/decoding/processors.ts`: `temperatureProcessor`, `topKProcessor`; `losses/classification.ts`: `crossEntropy`; `metrics/text.ts`: `bleuScore`, `rougeL`; `nn/language-model.ts`: `languageModelFromModule`, `characterTokenizer`; the paired data TODO as above | |
| `visual-question-answering` | Questions about a scene ("what colour is the circle", "how many squares") answered from the image and question embeddings fused (the fusion as a mode: concatenation, a product, co-attention) through a classifier over the answer vocabulary; accuracy by question type on the tape (counting is the hard one), the question-only baseline as the comparison (language priors answer more than they should); the attention over the image per question as a probe. | `training/loss` s, `eval/accuracy` s, `eval/accuracyByType` t, `eval/questionOnly` s, `model/attention` p, `model/params` p | `nn/functional/attention.ts`: `multiHeadAttention`, `scaledDotProductAttention`; `nn/module/containers.ts`: `Parallel`, `Sequential`, `mlp`; `nn/training/fit.ts`: `fitAlgorithm`; the questions and answers from `datasets/captions.ts` (TODO, above) | |
| `text-conditioned-generation` | A conditional generator (a VAE or a small diffusion model, the sampler as `generative-models`') with the sentence embedding as its condition: samples for a held-out sentence at the viewed step, classifier-free guidance's scale as the knob (fidelity against diversity), the attribute named in the sentence checked by a classifier on the samples; the mechanism of conditioning, not an image model. | `training/loss` s, `eval/attributeAccuracy` s, `samples/images` t! (for a fixed set of sentences), `samples/picture` p, `model/params` p | `losses/generative.ts`: `diffusionLoss`, `diffusionTarget`, `negativeElbo`; `nn/blocks/generative.ts`: `TimeEmbedding`, `DiffusionResBlock`; the samplers of `generative-models` (see that chapter); classifier-free guidance a few lines in the method | |

### retrieval/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `cross-modal-retrieval` | A query from either side over the joint embedding: the ranked results with the modality as marker, recall at k and MRR both ways at the viewed training step, the index as a slot (brute force, HNSW from `recommendation-and-retrieval`), CLIPScore of the pairs; the asymmetry of the two directions on the tape and what causes it (several sentences per scene). | `query/ranking` p (a ranking strip of images or sentences), `eval/recallAtK` s, `eval/mrr` s, `eval/clipScore` s, `model/embedding` p | `numerics/spatial/brute.ts`: `kNearest`; `numerics/spatial/graphs.ts`: `hnswIndex`; `metrics/ranking.ts`: `recallAtK`, `meanReciprocalRank`; `metrics/generative.ts`: `clipScore`; `distances/similarities.ts`: `cosineSimilarity` | |

### fusion/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `fusion-strategies` | A classifier over two modalities (an image and a sentence, or an image and a tabular row) with the fusion as a mode: early (concatenated inputs), late (one classifier per modality, their outputs combined), intermediate (concatenated embeddings), co-attention (each modality attending to the other); accuracy on the tape, each modality's contribution by ablation (the drop when it is masked), a modality missing at test time (the knob: the share of rows without it) and how each fusion degrades; modality dropout in training as the cure. | `training/loss` s, `eval/accuracy` s, `eval/accuracyMissing` s, `eval/contribution` t (per modality), `model/attention` p (co-attention), `model/params` p | `nn/module/containers.ts`: `Parallel`, `Sequential`, `mlp`; `nn/functional/attention.ts`: `multiHeadAttention`; `nn/functional/regularisation.ts`: `dropout`; `explain/perturbation.ts`: `featureAblation`; `nn/training/fit.ts`: `fitAlgorithm`; the paired data TODO as above | |

### analysis/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `modality-gap` | The two modalities' embeddings sit on separate cones of the sphere even when aligned: the gap (the distance between the modality centroids) on the tape from initialisation (where it is largest), the temperature's effect on it, alignment and uniformity of the pairs; the joint space projected with the modality as marker, the gap drawn as the vector between centroids; shifting one modality by the gap and what it does to retrieval. | `gap/distance` s, `gap/alignment` s, `gap/uniformity` s, `model/embedding` p, `gap/centroids` p | `distances/distances.ts`: `cosineDistance`, `euclideanDistance`; `losses/contrastive.ts` (the alignment and uniformity terms a few lines from `infoNce`'s pieces; `representation-learning` owns the measures); the aligned model above | |
| `cross-attention-maps` | The decoder's attention over the image patches per generated word (which patch "circle" looks at), attention rollout through the layers, the maps as an image grid under the caption's tokens; a caption that names an absent object (hallucination) and where its attention falls. | `attention/maps` t! (per word), `attention/picture` p (an `image` layout per token), `caption/tokens` p | `explain/internal.ts`: `attentionRollout`, `attentionFlow`; `nn/functional/attention.ts`: `multiHeadAttention` (the weights returned); the captioning model above | |
| `probes-per-modality` | Linear probes on each encoder for each planted attribute (shape, colour, count, position): what each modality encodes and at which layer, against what the joint space keeps after alignment (an attribute the sentences never mention fades from the image tower); the probe accuracies as a table over layers. | `probe/accuracy` t! (attribute by layer), `probe/table` p, `model/embedding` p | `explain/concepts.ts`: `linearProbe`, `conceptActivationVector`; `nn/module/module.ts`: `tap` (the layer activations) | |
| `multimodal-representations` | see `representation-learning` (the self-supervised objectives in general and the evaluation of representations; here only the two-modality case) | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `pairs.ts` | The paired-data slot: scenes of shapes with sentences from a template grammar (one or several per scene), questions with answers, the planted attributes as truth, held-out splits by scene; the image-and-tabular pairs for fusion; built on `datasets/images.ts` and the captions generator to add | every group |
| `encoders.ts` | The encoder slot: image encoders (patches through a small transformer, a convolution stack, HOG features as the frozen baseline) and text encoders (a bag of embeddings, a small transformer, the n-gram vectors of `process/text/` as the frozen baseline), each with its space | `alignment/`, `generation/`, `fusion/` |
| `probes.ts` | The joint-space probes: the embedding with the modality as marker, the batch similarity matrix, a ranking strip of images or sentences for a query, the image grid with captions and per-token probabilities, attention maps under tokens | every group |

## Examples to make

- CLIP-style training on shape scenes and their sentences: the joint embedding with both modalities, the pairs drawing together, the batch similarity matrix's diagonal brightening.
- Captioning a scene as training proceeds, the caption's tokens with their probabilities, the cross-attention of each word on the image.
- Zero-shot classification by prompt against a linear probe, accuracy per class.
- Fusion with a modality missing at test time: early against late fusion degrading differently.
- The modality gap from initialisation through training, the temperature as the knob.

## Rendering

- Embedding steps with modality as marker, image grids with captions, ranking strips, attention maps as images under tokens, similarity matrices as heatmaps.
- Missing: an image-with-caption grid panel (images, their captions with per-token probabilities, the attention map on hover); a two-sided ranking strip (a query on one side, results of the other modality).

## Problems

- The only library gap is the data: `datasets/captions.ts` (scenes with template sentences, questions and answers, the attributes as truth) and a paired schema in `data/schema/`; every model piece exists.
- Everything is synthetic and tiny: the encoders are a few layers on 16-pixel scenes, the sentences come from a grammar of a dozen words; the pages say the mechanism is the point, and a comparison with a pretrained model is out of reach in the browser.
- Audio as a third modality (spoken digits against their text) waits on an audio dataset (section 7.11 of `methods/README.md`); `signal/audio` has the features.
- Three chapters touch the joint space: `representation-learning` (the objectives and the measures), `recommendation-and-retrieval` (the index and the metrics) and this one (the two-modality case); the `see` rows keep each in one place.
