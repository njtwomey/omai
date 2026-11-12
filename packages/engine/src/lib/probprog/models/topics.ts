/**
 * A tiny latent Dirichlet allocation model (LDA-lite): three documents of five words over a vocabulary of four, two
 * topics, with Dirichlet priors on the document-topic mixtures and the topic-word distributions and a categorical topic
 * for every token. Small enough that the posterior is exact: with the mixtures and topics integrated out in closed form
 * (the Dirichlet-multinomial marginal), the joint of the data and the topic assignments depends on counts only, and the sum
 * over all 2^15 assignments gives the evidence and the posterior of every assignment. Given the assignments the
 * mixtures and topics are Dirichlet, so their marginals are exact mixtures of Betas over the counts. The topic priors
 * are asymmetric (topic 0 favours words 0 and 1, topic 1 words 2 and 3), which breaks the label symmetry softly: the
 * swapped labelling is in the truth with its small mass, and a sampler that stays in the main mode is right to within
 * that mass.
 */

import { betainc, lgamma } from '../../numerics/special/index.ts'
import { uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { arange, div, full, item, toArray, toFlat, zeros } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import type { MarginalOracle } from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { pmfOracle } from './discrete-truth.ts'
import { gibbsPlan, laplacePlan, nutsPlan, sviPlan } from './plans.ts'

const DOCS = 3
const TOKENS = 5
const TOPICS = 2
const WORDS = 4
/** The document-topic Dirichlet concentration (symmetric). */
const ALPHA = 0.5
/** The topic-word concentrations of each topic (asymmetric: each topic favours two words). */
const BETA = [
  [2, 2, 0.3, 0.3],
  [0.3, 0.3, 2, 2],
]
/** The word distributions the corpus is drawn from, and the mixtures of its documents. */
const TRUE_PHI = [
  [0.4, 0.4, 0.1, 0.1],
  [0.1, 0.1, 0.4, 0.4],
]
const TRUE_THETA = [
  [0.9, 0.1],
  [0.5, 0.5],
  [0.15, 0.85],
]

const pick = (probs: readonly number[], u: number): number => {
  let acc = 0
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i] as number
    if (u < acc) return i
  }
  return probs.length - 1
}

function generateCorpus(key: Key): BenchmarkData {
  const u = toFlat(drawUniform(foldIn(key, 0), [DOCS * TOKENS * 2]))
  const z: number[][] = []
  const w: number[][] = []
  for (let dd = 0; dd < DOCS; dd++) {
    z.push([])
    w.push([])
    for (let n = 0; n < TOKENS; n++) {
      const topic = pick(TRUE_THETA[dd] as number[], u[(dd * TOKENS + n) * 2] ?? 0.5)
      z[dd]?.push(topic)
      w[dd]?.push(pick(TRUE_PHI[topic] as number[], u[(dd * TOKENS + n) * 2 + 1] ?? 0.5))
    }
  }
  return { data: { w }, generating: { theta: TRUE_THETA, phi: TRUE_PHI, z } }
}

const lgCache = new Map<number, number>()
const lg = (x: number): number => {
  let v = lgCache.get(x)
  if (v === undefined) {
    v = item(lgamma(x))
    lgCache.set(x, v)
  }
  return v
}

/** A mixture of Betas as a marginal oracle: exact moments, and a CDF tabulated once on a fine grid of the unit interval. */
function betaMixtureOracle(
  name: string,
  parts: readonly { readonly weight: number; readonly a: number; readonly b: number }[],
): MarginalOracle {
  const total = parts.reduce((s, p) => s + p.weight, 0)
  const w = parts.map((p) => p.weight / total)
  let mean = 0
  let second = 0
  parts.forEach((p, i) => {
    const m = p.a / (p.a + p.b)
    mean += (w[i] as number) * m
    second += (w[i] as number) * ((p.a * p.b) / ((p.a + p.b) ** 2 * (p.a + p.b + 1)) + m * m)
  })
  const N = 2000
  const xs = div(arange(N + 1), N)
  // the cumulative distribution, summed over the parts in place
  const F = toFlat(zeros([N + 1]))
  parts.forEach((p, i) => {
    const c = toFlat(betainc(p.a, p.b, xs))
    for (let j = 0; j <= N; j++) F[j] = (F[j] as number) + (w[i] as number) * (c[j] as number)
  })
  for (let j = 1; j <= N; j++) F[j] = Math.max(F[j] as number, F[j - 1] as number)
  const locate = (target: number): number => {
    let lo = 0
    let hi = N
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if ((F[mid] as number) <= target) lo = mid
      else hi = mid
    }
    return lo
  }
  return {
    name,
    mean,
    sd: Math.sqrt(Math.max(second - mean * mean, 0)),
    cdf: (x) => {
      if (x <= 0) return 0
      if (x >= 1) return 1
      const j = Math.min(Math.floor(x * N), N - 1)
      const t = x * N - j
      return (F[j] as number) + t * ((F[j + 1] as number) - (F[j] as number))
    },
    quantile: (p) => {
      if (p <= (F[0] as number)) return 0
      if (p >= (F[N] as number)) return 1
      const j = locate(p)
      const span = (F[j + 1] as number) - (F[j] as number)
      return ((j + (span === 0 ? 0 : (p - (F[j] as number)) / span)) / N) as number
    },
  }
}

/**
 * The exact posterior of the LDA-lite model: enumerate the `2^15` topic assignments with the mixtures and topics integrated
 * out (the Dirichlet-multinomial marginal), weight each by its collapsed joint with the data, and read off the marginal of
 * every assignment, the evidence, and the marginals of the mixtures and topics as mixtures of Betas over the counts.
 *
 * @param w - The words, `[doc][token]`.
 * @returns The truth: the marginals of `z` (flat index `d * 5 + n`), `theta` (`d * 2 + k`, the weight of topic 1) and `phi` (`k * 4 + v`), and the log evidence.
 * @concept topic-models
 * @example
 * ```ts
 * import { ldaLiteTruth } from 'omai-engine/probprog/models'
 * const t = ldaLiteTruth([[0, 1, 0, 1, 0], [0, 2, 1, 3, 0], [2, 3, 3, 2, 3]])
 * [Object.keys(t.marginals).length, t.kind] // => [26, 'exact']
 * ```
 */
export function ldaLiteTruth(w: readonly (readonly number[])[]): BenchmarkTruth {
  const T = DOCS * TOKENS
  const logPrior = (counts: readonly number[], conc: readonly number[]): number => {
    const sumC = conc.reduce((a, c) => a + c, 0)
    const n = counts.reduce((a, c) => a + c, 0)
    return (
      lg(sumC) - lg(sumC + n) + counts.reduce((a, c, i) => a + lg((conc[i] as number) + c) - lg(conc[i] as number), 0)
    )
  }
  const configs = 2 ** T
  const logw = toFlat(zeros([configs]))
  const docCount = (c: number, dd: number): number => {
    let n = 0
    for (let t = 0; t < TOKENS; t++) n += (c >> (dd * TOKENS + t)) & 1
    return n
  }
  const topicWord = (c: number): number[][] => {
    const m = toArray(zeros([2, WORDS])) as number[][]
    for (let dd = 0; dd < DOCS; dd++)
      for (let t = 0; t < TOKENS; t++) {
        const topic = (c >> (dd * TOKENS + t)) & 1
        const word = (w[dd] as readonly number[])[t] as number
        ;(m[topic] as number[])[word] = ((m[topic] as number[])[word] as number) + 1
      }
    return m
  }
  let max = Number.NEGATIVE_INFINITY
  for (let c = 0; c < configs; c++) {
    let s = 0
    for (let dd = 0; dd < DOCS; dd++) {
      const n1 = docCount(c, dd)
      s += logPrior([TOKENS - n1, n1], [ALPHA, ALPHA])
    }
    const m = topicWord(c)
    for (let k = 0; k < TOPICS; k++) s += logPrior(m[k] as number[], BETA[k] as number[])
    logw[c] = s
    if (s > max) max = s
  }
  const weights = logw.map((v) => Math.exp(v - max))
  const total = weights.reduce((a, v) => a + v, 0)
  const logEvidence = max + Math.log(total)
  const marginals: Record<string, MarginalOracle> = {}
  // Marginal of each assignment, and the distribution of each document's count of topic-1 tokens.
  const p1 = toFlat(zeros([T]))
  const docPmf = toArray(zeros([DOCS, TOKENS + 1])) as number[][]
  // banned-ok: non-numeric: one map of terms per cell
  const phiParts = Array.from({ length: TOPICS * WORDS }, () => new Map<string, number>())
  for (let c = 0; c < configs; c++) {
    const q = (weights[c] as number) / total
    for (let t = 0; t < T; t++) if (((c >> t) & 1) === 1) p1[t] = (p1[t] as number) + q
    for (let dd = 0; dd < DOCS; dd++) {
      const pmf = docPmf[dd] as number[]
      const n1 = docCount(c, dd)
      pmf[n1] = (pmf[n1] as number) + q
    }
    const m = topicWord(c)
    for (let k = 0; k < TOPICS; k++) {
      const mk = (m[k] as number[]).reduce((a, v) => a + v, 0)
      for (let v = 0; v < WORDS; v++) {
        const key = `${(m[k] as number[])[v]},${mk}`
        const map = phiParts[k * WORDS + v] as Map<string, number>
        map.set(key, (map.get(key) ?? 0) + q)
      }
    }
  }
  for (let dd = 0; dd < DOCS; dd++)
    for (let n = 0; n < TOKENS; n++) {
      const name = `z[${dd * TOKENS + n}]`
      marginals[name] = pmfOracle(name, [0, 1], [1 - (p1[dd * TOKENS + n] as number), p1[dd * TOKENS + n] as number])
    }
  for (let dd = 0; dd < DOCS; dd++) {
    const name = `theta[${dd * TOPICS + 1}]`
    marginals[name] = betaMixtureOracle(
      name,
      (docPmf[dd] as number[]).map((weight, n1) => ({ weight, a: ALPHA + n1, b: ALPHA + TOKENS - n1 })),
    )
  }
  for (let k = 0; k < TOPICS; k++)
    for (let v = 0; v < WORDS; v++) {
      const name = `phi[${k * WORDS + v}]`
      const conc = BETA[k] as number[]
      const sum = conc.reduce((a, x) => a + x, 0)
      marginals[name] = betaMixtureOracle(
        name,
        [...(phiParts[k * WORDS + v] as Map<string, number>)].map(([key, weight]) => {
          const [mv, mk] = key.split(',').map(Number) as [number, number]
          return { weight, a: (conc[v] as number) + mv, b: sum - (conc[v] as number) + mk - mv }
        }),
      )
    }
  return {
    kind: 'exact',
    source:
      'enumeration of the 2^15 topic assignments with the mixtures and topics integrated out (Dirichlet-multinomial marginals, written out here): the evidence is their sum, each assignment marginal a weighted count, and each mixture or topic probability a mixture of Betas over the counts',
    logEvidence,
    marginals,
  }
}

/**
 * LDA-lite: three documents of five words over four, two topics, `theta_d ~ Dirichlet(0.5, 0.5)`, `phi_k ~ Dirichlet(beta_k)`
 * with each topic's prior favouring two words, a topic `z_dn ~ Categorical(theta_d)` and a word `w_dn ~ Categorical(phi_{z_dn})`
 * for every token. The posterior has 2^15 assignments, so the truth is exact; NUTS sums the topics out, Gibbs draws the
 * Dirichlet and categorical conditionals exactly.
 *
 * @concept topic-models
 * @index dataset LDA-lite: three documents, two topics, four words; the exact posterior by enumeration of the assignments.
 * @example
 * ```ts
 * import { ldaLite } from 'omai-engine/probprog/models'
 * [ldaLite.bound().sizes, ldaLite.truth().kind] // => [{ doc: 3, token: 5, topic: 2, word: 4 }, 'exact']
 * ```
 */
export const ldaLite: BenchmarkModel = defineBenchmark({
  name: 'lda-lite',
  title: 'LDA on a tiny corpus',
  description:
    'Latent Dirichlet allocation with three documents, two topics and four words: Dirichlet and categorical plates whose exact posterior comes from enumerating the topic assignments.',
  tags: ['topic-model', 'lda', 'dirichlet', 'discrete-latent', 'plates', 'exact', 'enumeration'],
  references: [
    'Blei, Ng and Jordan (2003), Latent Dirichlet allocation',
    'Griffiths and Steyvers (2004), Finding scientific topics (the collapsed sampler)',
  ],
  definition: defineModel('lda-lite', (b) => {
    const doc = b.range('doc')
    const token = b.range('token')
    const topic = b.range('topic', { size: TOPICS })
    const word = b.range('word', { size: WORDS })
    const w = b.data('w', { dims: [doc, token], support: 'index', role: 'response', indexes: word })
    const theta = b.latent('theta', d.dirichlet(full([TOPICS], ALPHA)), {
      dims: [doc],
      doc: 'the topic mixture of each document',
    })
    const phi = b.latent('phi', d.dirichlet(b.const(BETA, [topic])), {
      dims: [topic],
      doc: 'the word distribution of each topic',
    })
    const z = b.latent('z', d.categorical({ probs: theta }), { dims: [doc, token], doc: 'the topic of each token' })
    b.observe('w', d.categorical({ probs: phi.at(z) }), w)
  }),
  generate: (key) => generateCorpus(key),
  truth: (data) => ldaLiteTruth(data.data['w'] as number[][]),
  plans: (scale) => {
    // banned-ok: non-numeric: the names of the sites
    const mixtures = Array.from({ length: DOCS }, (_, dd) => `theta[${dd * TOPICS + 1}]`)
    const topics = Array.from({ length: TOPICS * WORDS }, (_, i) => `phi[${i}]`)
    return [
      nutsPlan(scale, {
        note: 'NUTS on the simplex-constrained mixtures and topics with the assignments summed out: consistent, within the main mode',
        tolerances: { meanError: 0.3, meanZ: 6, sdRatio: [0.6, 1.4], ks: 0.25 },
        quantities: [...mixtures, ...topics],
      }),
      gibbsPlan(scale, {
        // The four tokens of the third document that share a word move together, so a sweep's autocorrelation time is near a hundred:
        // 2000 sweeps leave a mean error of the assignments of 0.1 to 0.2 posterior sds, 8000 about 0.07 (20000 sweeps: 0.03 to 0.06).
        draws: 8000,
        note: 'exact Dirichlet and categorical conditionals in a sweep: consistent, correlated between sweeps (the Kolmogorov distance of a two-valued assignment is not small for a correct sampler, so it is not checked)',
        tolerances: { meanError: 0.2, meanZ: 6, sdRatio: [0.8, 1.25], ks: 1 },
      }),
      laplacePlan(scale, {
        note: 'a Gaussian at one mode of the unconstrained density, with the assignments summed out: the simplex coordinates are far from normal at this sample size',
        quantities: [...mixtures, ...topics],
      }),
      sviPlan(scale, {
        note: 'a mean-field normal on the unconstrained simplex coordinates: biased means and spreads too small',
        quantities: [...mixtures, ...topics],
      }),
    ]
  },
})
