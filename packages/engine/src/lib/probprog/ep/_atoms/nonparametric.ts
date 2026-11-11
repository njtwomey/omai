/**
 * The setups of expectation propagation with discrete assignments under Dirichlet-type weights: a truncated Dirichlet-process
 * mixture, whose weights are the stick-breaking of beta breaks (one `stick-breaking` factor per assignment), and a topic model,
 * whose documents each have a Dirichlet mixture of topics and whose topics each a Dirichlet word distribution (the word read
 * through the topic of its token: one `gate` per token over the topics' conjugate `discrete` factors). The exact answers
 * sum the assignments out: a grid over the breaks, and for the topic model an enumeration of the topic assignments with the
 * Dirichlets integrated out, which makes every posterior a mixture of betas.
 */

import { dict, real } from '../../../space/index.ts'
import type { Config } from '../../../trace/index.ts'
import { configNumber } from '../../inference/method.ts'
import * as d from '../../model/dists/index.ts'
import { defineModel, ops } from '../../model/index.ts'
import { type ExactMarginal, gridMarginals, lg, logWeights, normalPdf } from '../../validation/index.ts'
import { defineAtom } from '../_define.ts'
import { tolerance } from '../_tolerance.ts'
import type { AtomParts, EpAtom } from '../atom-types.ts'

// ----- the stick-breaking mixture -----

const COMPONENT_MEANS = [-3, 0, 3]
const STICK_DATA = [-3.2, -2.8, -0.4, -3.1, 2.9, 0.3, -2.7, 3.3, -0.2, -3.4]

function stickBreakingProblem(k: Config): AtomParts {
  const alpha = configNumber(k, 'concentration')
  const noise = configNumber(k, 'noise')
  // region stick-breaking
  const definition = defineModel('stick-breaking', (m) => {
    const row = m.range('row')
    const stick = m.range('stick', { size: 2 })
    const component = m.range('component', { size: 3 })
    const means = m.const(COMPONENT_MEANS, [component])
    // each break is the share of what is left that the next component takes
    const v = m.latent('v', d.beta(1, alpha), { dims: [stick] })
    const weights = m.derived('weights', ops.stickBreaking(v))
    const z = m.latent('z', d.categorical({ probs: weights }), { dims: [row] })
    const y = m.data('y', { dims: [row], role: 'response' })
    m.observe('y', d.normal(ops.at(means, z), noise), y)
  })
  // endregion
  const logDensity = (v: readonly number[]): number => {
    const [v0, v1] = [v[0] ?? 0, v[1] ?? 0]
    const pi = [v0, v1 * (1 - v0), (1 - v0) * (1 - v1)]
    let lp = (alpha - 1) * (Math.log(1 - v0) + Math.log(1 - v1))
    for (const y of STICK_DATA)
      lp += Math.log(pi.reduce((s, p, c) => s + p * normalPdf(y, COMPONENT_MEANS[c] as number, noise), 0))
    return lp
  }
  const g = gridMarginals(
    logDensity,
    [
      [0, 1],
      [0, 1],
    ],
    301,
  )
  return {
    build: {
      definition,
      data: { y: STICK_DATA },
      quantities: [
        { name: 'v[0]', label: 'v_1 (the share taken by the first component)', support: 'unit' },
        { name: 'v[1]', label: 'v_2 (the share of the rest taken by the second)', support: 'unit' },
      ],
    },
    reference: {
      exact: { 'v[0]': g.marginal(0), 'v[1]': g.marginal(1) },
      exactJoint: { a: 'v[0]', b: 'v[1]', ...g.joint(0, 1) },
      referenceSource: 'a grid over the two breaks with the assignments summed out',
    },
    present: {
      equations: [
        `v_1, v_2 ~ Beta(1, ${alpha})   (the breaks)`,
        'pi = (v_1, v_2 (1 - v_1), (1 - v_1)(1 - v_2))   (stick-breaking, truncated at three components)',
        `z_i ~ Categorical(pi),  y_i ~ N(mean[z_i], ${noise}²), means (${COMPONENT_MEANS.join(', ')})`,
        `observe y = (${STICK_DATA.join(', ')})`,
      ],
      pair: ['v[0]', 'v[1]'],
    },
  }
}

/**
 * A truncated Dirichlet-process mixture of three known Gaussians: two Beta breaks whose stick-breaking weights are the
 * probabilities of the assignments. The exact posterior of the breaks is a two-dimensional grid with the assignments summed out.
 */
const stickBreakingAtom: EpAtom = defineAtom({
  name: 'stick-breaking',
  title: 'Stick-breaking mixture (truncated Dirichlet process)',
  what: 'Ten observations from three known Gaussians whose weights are the stick-breaking of two Beta breaks, with a categorical assignment per observation: EP reads the weights through one stick-breaking factor per assignment, against the grid over the breaks.',
  factors: [
    'prior (beta) x2',
    'stick-breaking (an assignment with beta breaks)',
    'table (the component mean chosen by the assignment)',
  ],
  family: 'nonparametric',
  reference: 'grid',
  tolerance: tolerance(
    'gamma-projected',
    'the assignments are summed out exactly in each factor and the breaks are matched to one Beta each, so what EP approximates is the loop over the breaks that every assignment shares: measured within 0.02 reference sds',
  ),
  knobs: dict({
    concentration: real(1, 6, { default: 1, step: 0.25, label: 'concentration of the Dirichlet process (alpha)' }),
    noise: real(0.3, 1.5, { default: 1.3, step: 0.05, label: 'spread of each component (large: they overlap)' }),
  }),
  parts: stickBreakingProblem,
})

// ----- the topic model -----

const DOCS = 2
const TOKENS = 4
const WORD_DATA = [
  [0, 0, 1, 0],
  [1, 1, 0, 1],
]

/** The density of a mixture of betas, as an exact marginal on the unit interval. */
function betaMixture(
  parts: readonly { readonly weight: number; readonly a: number; readonly b: number }[],
): ExactMarginal {
  const total = parts.reduce((s, p) => s + p.weight, 0)
  const w = parts.map((p) => p.weight / total)
  let mean = 0
  let second = 0
  for (const [i, p] of parts.entries()) {
    const m = p.a / (p.a + p.b)
    mean += (w[i] as number) * m
    second += (w[i] as number) * ((p.a * p.b) / ((p.a + p.b) ** 2 * (p.a + p.b + 1)) + m * m)
  }
  return {
    mean,
    sd: Math.sqrt(Math.max(second - mean * mean, 0)),
    pdf: (x) => {
      if (x <= 0 || x >= 1) return 0
      return parts.reduce(
        (s, p, i) =>
          s +
          (w[i] as number) *
            Math.exp(lg(p.a + p.b) - lg(p.a) - lg(p.b) + (p.a - 1) * Math.log(x) + (p.b - 1) * Math.log(1 - x)),
        0,
      )
    },
    range: [0, 1],
  }
}

function topicModelProblem(k: Config): AtomParts {
  const alpha = configNumber(k, 'mixture')
  const sharp = configNumber(k, 'topicPrior')
  // each topic's prior leans to its own word: topic 0 to word 0, topic 1 to word 1 (the labels are softly fixed)
  const beta = [
    [sharp, 1],
    [1, sharp],
  ]
  // region topic-model
  const definition = defineModel('topic-model', (m) => {
    const doc = m.range('doc', { size: DOCS })
    const token = m.range('token', { size: TOKENS })
    const topic = m.range('topic', { size: 2 })
    const word = m.range('word', { size: 2 })
    // the topic mixture of each document and the word distribution of each topic
    const theta = m.latent('theta', d.dirichlet([alpha, alpha]), { dims: [doc] })
    const phi = m.latent('phi', d.dirichlet(m.const(beta, [topic])), { dims: [topic] })
    // the topic of each token, then the word it draws from that topic
    const z = m.latent('z', d.categorical({ probs: theta }), { dims: [doc, token] })
    const w = m.data('w', { dims: [doc, token], support: 'index', indexes: word })
    m.observe('w', d.categorical({ probs: phi.at(z) }), w)
  })
  // endregion
  // Every assignment of the eight tokens to topics, the Dirichlets integrated out: the posterior of an assignment is a product of
  // Dirichlet-multinomial terms, each posterior of a probability a mixture of betas over the counts the assignment makes.
  const configs = 2 ** (DOCS * TOKENS)
  const logPrior = (counts: readonly number[], conc: readonly number[]): number => {
    const sum = conc.reduce((a, c) => a + c, 0)
    const n = counts.reduce((a, c) => a + c, 0)
    return (
      lg(sum) - lg(sum + n) + counts.reduce((a, c, i) => a + lg((conc[i] as number) + c) - lg(conc[i] as number), 0)
    )
  }
  const assignments: { logw: number; doc: number[][]; topic: number[][] }[] = []
  for (let c = 0; c < configs; c++) {
    const doc = Array.from({ length: DOCS }, () => [0, 0])
    const topic = Array.from({ length: 2 }, () => [0, 0])
    for (let t = 0; t < DOCS * TOKENS; t++) {
      const z = (c >> t) & 1
      const dd = Math.floor(t / TOKENS)
      const word = (WORD_DATA[dd] as number[])[t % TOKENS] as number
      ;((doc[dd] as number[])[z] as number) += 1
      ;((topic[z] as number[])[word] as number) += 1
    }
    let logw = 0
    for (const dc of doc) logw += logPrior(dc, [alpha, alpha])
    for (let kk = 0; kk < 2; kk++) logw += logPrior(topic[kk] as number[], beta[kk] as number[])
    assignments.push({ logw, doc, topic })
  }
  const { weights } = logWeights(assignments.map((a) => a.logw))
  const exact: Record<string, ExactMarginal> = {}
  for (let dd = 0; dd < DOCS; dd++)
    exact[`theta[${dd * 2}]`] = betaMixture(
      assignments.map((a, i) => ({
        weight: weights[i] as number,
        a: alpha + ((a.doc[dd] as number[])[0] as number),
        b: alpha + ((a.doc[dd] as number[])[1] as number),
      })),
    )
  for (let kk = 0; kk < 2; kk++)
    exact[`phi[${kk * 2}]`] = betaMixture(
      assignments.map((a, i) => ({
        weight: weights[i] as number,
        a: ((beta[kk] as number[])[0] as number) + ((a.topic[kk] as number[])[0] as number),
        b: ((beta[kk] as number[])[1] as number) + ((a.topic[kk] as number[])[1] as number),
      })),
    )
  return {
    build: {
      definition,
      data: { w: WORD_DATA },
      quantities: [
        { name: 'theta[0]', label: 'theta_1 (document 1: the share of topic 0)', support: 'unit' },
        { name: 'theta[2]', label: 'theta_2 (document 2: the share of topic 0)', support: 'unit' },
        { name: 'phi[0]', label: 'phi_0 (topic 0: the probability of word 0)', support: 'unit' },
        { name: 'phi[2]', label: 'phi_1 (topic 1: the probability of word 0)', support: 'unit' },
      ],
    },
    reference: {
      exact,
      referenceSource: 'the 256 assignments of the eight tokens to topics, the Dirichlets integrated out',
    },
    present: {
      equations: [
        `theta_d ~ Dirichlet(${alpha}, ${alpha}) for the two documents   (the topic mixture of each)`,
        `phi_0 ~ Dirichlet(${beta[0]?.join(', ')}),  phi_1 ~ Dirichlet(${beta[1]?.join(', ')})   (the word distribution of each topic)`,
        'z_dn ~ Categorical(theta_d),  w_dn ~ Categorical(phi[z_dn])   (a topic per token, the word read through it)',
        `observe w = (${WORD_DATA.map((r) => r.join(' ')).join(' | ')})`,
      ],
    },
  }
}

/**
 * A topic model of two documents of four words over a vocabulary of two, with two topics: a Dirichlet mixture per document, a Dirichlet
 * word distribution per topic and a categorical topic per token, with each word read through its token's topic. The exact posterior
 * enumerates the assignments of the tokens to topics.
 */
const topicModelAtom: EpAtom = defineAtom({
  name: 'topic-model',
  title: 'Topic model (Dirichlet mixtures, a word read through its topic)',
  what: 'Two documents of four words, two topics: a Dirichlet mixture per document, a Dirichlet word distribution per topic, a topic per token and the word drawn from that topic: one gate per token over the topics’ conjugate discrete factors, against the enumeration of the assignments.',
  factors: [
    'prior (dirichlet)',
    'discrete (a topic with a Dirichlet mixture)',
    'gate (the topic chooses which word distribution reads the word)',
  ],
  family: 'nonparametric',
  reference: 'exact',
  tolerance: tolerance(
    'close',
    'the exact posteriors are mixtures of betas over the assignments (the label-swapped reading too), which one Dirichlet each cannot be; measured within 0.06 reference sds and 4 percent of the spread',
  ),
  knobs: dict({
    mixture: real(0.3, 3, { default: 1, step: 0.1, label: 'concentration of each document’s topic mixture' }),
    topicPrior: real(1, 5, { default: 2, step: 0.25, label: 'how strongly each topic’s prior leans to its own word' }),
  }),
  parts: topicModelProblem,
})

/** The atoms of the `nonparametric` file, basic and learned twins together. */
export const nonparametricAtoms: readonly EpAtom[] = [stickBreakingAtom, topicModelAtom]
