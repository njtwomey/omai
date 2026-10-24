import type { Params } from '../../testing/suite.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { type TensorLike, tensor } from '../tensor/index.ts'
import * as M from './index.ts'

type Refs = M.References
const refs = (p: Params): Refs => p['yTrue'] as Refs
const hyps = (p: Params): readonly string[] => p['yPred'] as readonly string[]
const truths = (p: Params): readonly string[] => p['yTrue'] as readonly string[]
const embeddings = (p: Params, key: string): TensorLike[] | undefined => p[key] as TensorLike[] | undefined
const field = (r: { readonly breakdown?: Readonly<Record<string, number>> }, p: Params): number =>
  r.breakdown?.[p['field'] as string] ?? Number.NaN

goldenSuite('text metrics against jiwer, sacrebleu, rouge-score and torchmetrics', 'metrics/text', {
  wordErrorRate: (_i, p) => M.wordErrorRate(truths(p), hyps(p)).estimate.value,
  matchErrorRate: (_i, p) => M.matchErrorRate(truths(p), hyps(p)).estimate.value,
  wordInformationPreserved: (_i, p) => M.wordInformationPreserved(truths(p), hyps(p)).estimate.value,
  wordInformationLost: (_i, p) => M.wordInformationLost(truths(p), hyps(p)).estimate.value,
  characterErrorRate: (_i, p) => M.characterErrorRate(truths(p), hyps(p)).estimate.value,
  editCounts: (_i, p) => {
    const b = M.wordErrorRate(truths(p), hyps(p)).breakdown ?? {}
    return tensor([b['hits'] ?? 0, b['substitutions'] ?? 0, b['deletions'] ?? 0, b['insertions'] ?? 0])
  },
  bleuScore: (_i, { yTrue, yPred, ...o }) => M.bleuScore(yTrue as Refs, yPred as string[], o).estimate.value,
  sentenceBleu: (_i, p) => M.sentenceBleu(p['yTrue'] as string[], p['yPred'] as string).estimate.value,
  chrfScore: (_i, { yTrue, yPred, ...o }) => M.chrfScore(yTrue as Refs, yPred as string[], o).estimate.value,
  translationEditRate: (_i, { yTrue, yPred, ...o }) =>
    M.translationEditRate(yTrue as Refs, yPred as string[], o).estimate.value,
  rougeN: (_i, p) => field(M.rougeN(refs(p), hyps(p), { n: p['n'] as number }), p),
  rougeL: (_i, p) => field(M.rougeL(refs(p), hyps(p)), p),
  rougeLsum: (_i, p) => field(M.rougeLsum(refs(p), hyps(p)), p),
  exactMatch: (_i, p) => M.exactMatch(refs(p), hyps(p)).estimate.value,
  tokenF1: (_i, p) => M.tokenF1(refs(p), hyps(p)).estimate.value,
  bertScore: (_i, p) => {
    const tw = embeddings(p, 'targetWeights')
    const pw = embeddings(p, 'predWeights')
    const r = M.bertScore(embeddings(p, 'targets') ?? [], embeddings(p, 'preds') ?? [], {
      ...(tw === undefined ? {} : { targetWeights: tw }),
      ...(pw === undefined ? {} : { predWeights: pw }),
    })
    return field(r, p)
  },
  perplexity: ([l, t], o) => M.perplexity(l, t, o).estimate.value,
  distinctN: (_i, p) => M.distinctN(hyps(p), { n: p['n'] as number }).estimate.value,
  tokenize13a: (_i, p) => M.tokenize13a(p['line'] as string),
})
