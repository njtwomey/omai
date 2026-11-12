/**
 * A small Bayesian neural network: one input, three tanh hidden units and a linear output, `N(0, 1)` priors on every weight
 * and bias, sixteen noisy samples of `sin(1.5 x)` with a gap in the middle of the input range, and the network's output at five
 * test inputs recorded as a derived quantity (`fstar`). The weights are not identified (permuting the hidden units or
 * flipping a unit's signs gives the same function), so the raw weights have no single-mode marginals; the function values
 * are what a comparison reads, and there is no closed form for them. The truth is a stored reference: the moments of the
 * function values from a long NUTS run (`bnnReferenceRun` repeats it), so every plan is reported, not required to agree.
 * What the reference shows, and the plans are read against: the predictive uncertainty is small near the data, larger in the
 * gap and largest far outside the data.
 */

import { OmaiError } from '../../base/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { add, mul, tanh, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { mcmc } from '../engines/nuts.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops, toProgram } from '../model/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'
import { mixtureOracle } from './truth.ts'

const HIDDEN = 3
const NOISE = 0.15
/** The inputs at which the network's output is compared: inside the data, in the gap, and outside it. */
export const BNN_TEST_INPUTS: readonly number[] = [-3, -1.25, 0, 1.25, 3]

function generateRegression(key: Key, size = 16): BenchmarkData {
  const u = toFlat(drawUniform(foldIn(key, 0), [size]))
  const e = toFlat(drawNormal(foldIn(key, 1), [size]))
  // Inputs on [-2, -0.5] and [0.5, 2]: a gap in the middle.
  const x = Array.from(u, (v, i) => (i % 2 === 0 ? -2 + 1.5 * v : 0.5 + 1.5 * v)).sort((a, b) => a - b)
  return {
    data: { x, y: x.map((v, i) => Math.sin(1.5 * v) + NOISE * (e[i] ?? 0)), xs: [...BNN_TEST_INPUTS] },
    generating: { noise: NOISE, hidden: HIDDEN },
  }
}

/** The stored reference: the mean and standard deviation of the network output at each test input (see {@link bnnReferenceRun}). */
const REFERENCE: readonly { readonly mean: number; readonly sd: number }[] = [
  { mean: -0.2106, sd: 0.4499 },
  { mean: -0.9035, sd: 0.06646 },
  { mean: 0.2953, sd: 0.2635 },
  { mean: 0.8281, sd: 0.08393 },
  { mean: 0.1441, sd: 0.3761 },
]

/**
 * The network's output at each test input for each draw of the weights (`fstar[j]`), computed from the weight draws, so a
 * method whose posterior holds the weights and not the derived output (Laplace, a guide) can be compared as well.
 */
function networkOutputs(draws: Readonly<Record<string, ArrayLike<number>>>): Record<string, ArrayLike<number>> {
  const n = draws['b2']?.length
  if (n === undefined) return {}
  const out: Record<string, number[]> = {}
  const column = (name: string) => tensor(Array.from(draws[name] as ArrayLike<number>))
  BNN_TEST_INPUTS.forEach((x, j) => {
    // the output of the network at x for every draw at once: the output bias plus the hidden units, added in order
    let f: Value = column('b2')
    for (let h = 0; h < HIDDEN; h++)
      f = add(f, mul(column(`w2[${h}]`), tanh(add(mul(column(`w1[${h}]`), x), column(`b1[${h}]`)))))
    out[`fstar[${j}]`] = toFlat(f)
  })
  return out
}

/**
 * The long NUTS run behind the stored reference: the mean and standard deviation of the network output at each test input on
 * the canonical data. The stored numbers come from `bnnReferenceRun()` with its defaults; it takes about ten minutes, so
 * nothing runs it by itself (the benchmark's periodic campaign compares a shorter run with the stored numbers).
 *
 * @param options - Chains (default 4), warm-up (1000) and draws (2500) per chain, and the key (default `key(7)`).
 * @returns The mean and sd of `fstar[j]` for each test input, and the sampler's divergences and the largest R-hat among them.
 * @concept bayesian-neural-networks
 */
export function bnnReferenceRun(
  options: { readonly chains?: number; readonly warmup?: number; readonly draws?: number; readonly key?: Key } = {},
): {
  readonly moments: readonly { readonly mean: number; readonly sd: number }[]
  readonly divergences: number
  readonly maxRhat: number
} {
  const p = mcmc(toProgram(bayesianNeuralNetwork.bound()), undefined, {
    key: options.key ?? makeKey(7),
    numChains: options.chains ?? 4,
    numWarmup: options.warmup ?? 1000,
    numDraws: options.draws ?? 2500,
    targetAcceptance: 0.9,
  })
  const rows = BNN_TEST_INPUTS.map((_, j) => p.summary().find((r) => r.name === `fstar[${j}]`))
  return {
    moments: rows.map((r) => ({ mean: r?.mean ?? Number.NaN, sd: r?.sd ?? Number.NaN })),
    divergences: Number(p.diagnostics['divergences'] ?? 0),
    maxRhat: Math.max(...rows.map((r) => r?.rhat ?? Number.NaN)),
  }
}

/**
 * The reference moments of the Bayesian neural network benchmark.
 *
 * @param data - The data (only the canonical data have a stored reference).
 * @returns The truth, of kind `reference`: a normal oracle with the stored mean and sd of each `fstar[j]`.
 * @throws OmaiError with code `value` for data other than the canonical data.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { bnnReferenceTruth, bayesianNeuralNetwork } from 'omai-engine/probprog/models'
 * Object.keys(bnnReferenceTruth(bayesianNeuralNetwork.data).marginals) // => ['fstar[0]', 'fstar[1]', 'fstar[2]', 'fstar[3]', 'fstar[4]']
 * ```
 */
export function bnnReferenceTruth(data: BenchmarkData): BenchmarkTruth {
  if (JSON.stringify(data.data) !== JSON.stringify(generateRegression(makeKey(0)).data))
    throw new OmaiError('value', 'the neural network benchmark has a stored reference for its canonical data only')
  const marginals = Object.fromEntries(
    REFERENCE.map((r, j) => [`fstar[${j}]`, mixtureOracle(`fstar[${j}]`, [{ weight: 1, mean: r.mean, sd: r.sd }])]),
  )
  return {
    kind: 'reference',
    source:
      'the mean and standard deviation of the network output at five test inputs from a long NUTS run (four chains, 1000 warm-up and 2500 draws each, key(7), repeated by `bnnReferenceRun`; no divergences, R-hat at most 1.0014 and ESS above 3000 for the five function values); stored, not draws; the weights have no single-mode marginals (hidden-unit permutation and sign symmetries), so the function values are compared',
    marginals,
  }
}

/**
 * A Bayesian neural network on a small regression: one input, three tanh units, `N(0, 1)` weights and biases, noise 0.15
 * known, sixteen points with a gap. The output at five test inputs (`fstar`) is the quantity compared with a stored NUTS
 * reference. Every plan is reported: the uncertainty should grow away from the data, a mean-field guide is overconfident, and
 * the Laplace fit sees one mode of a posterior with many symmetric ones.
 *
 * @concept bayesian-neural-networks
 * @index dataset Bayesian neural network (three tanh units) on a small regression; stored NUTS reference for the output.
 * @example
 * ```ts
 * import { bayesianNeuralNetwork } from 'omai-engine/probprog/models'
 * [bayesianNeuralNetwork.bound().sizes, bayesianNeuralNetwork.truth().kind] // => [{ point: 16, hidden: 3, test: 5 }, 'reference']
 * ```
 */
export const bayesianNeuralNetwork: BenchmarkModel = defineBenchmark({
  name: 'bayesian-neural-network',
  title: 'Bayesian neural network',
  description:
    'A one-hidden-layer tanh network with Gaussian priors on a noisy sine with a gap in the inputs: no closed form, a stored long-NUTS reference for the output, and the predictive uncertainty that should grow away from the data.',
  tags: ['neural-network', 'regression', 'non-identifiable', 'reference', 'uncertainty'],
  references: [
    'Neal (1996), Bayesian Learning for Neural Networks',
    'MacKay (1992), A practical Bayesian framework for backpropagation networks',
    'Izmailov et al. (2021), What are Bayesian neural network posteriors really like?',
  ],
  definition: defineModel('bayesian-neural-network', (m) => {
    const point = m.range('point')
    const hidden = m.range('hidden', { size: HIDDEN })
    const test = m.range('test')
    const x = m.data('x', { dims: [point], role: 'covariate' })
    const y = m.data('y', { dims: [point], role: 'response' })
    const xs = m.data('xs', { dims: [test], role: 'covariate' })
    const w1 = m.latent('w1', d.normal(0, 1), { dims: [hidden], doc: 'input weights of the hidden units' })
    const b1 = m.latent('b1', d.normal(0, 1), { dims: [hidden], doc: 'biases of the hidden units' })
    const w2 = m.latent('w2', d.normal(0, 1), { dims: [hidden], doc: 'output weights' })
    const b2 = m.latent('b2', d.normal(0, 1), { doc: 'output bias' })
    const output = (input: typeof x): ReturnType<typeof x.mul> =>
      ops.tanh(input.mul(w1).add(b1)).mul(w2).sum(hidden).add(b2)
    m.observe('y', d.normal(output(x), NOISE), y)
    m.derived('fstar', output(xs), { doc: 'the network output at the test inputs' })
  }),
  generate: (key, size) => generateRegression(key, size),
  summarise: networkOutputs,
  truth: (data) => bnnReferenceTruth(data),
  plans: (scale) => [
    nutsPlan(scale, {
      expect: 'report',
      note: 'multimodal in the weights (unit permutations) but the function values are well mixed: close to the long run it is compared with (a consistency check only, since the reference is itself NUTS)',
      quantities: [...BNN_TEST_INPUTS.map((_, j) => `fstar[${j}]`)],
    }),
    temperedPlan(scale, {
      expect: 'report',
      note: 'tempered particles over ten weights: finds several symmetric modes',
      quantities: BNN_TEST_INPUTS.map((_, j) => `fstar[${j}]`),
    }),
    laplacePlan(scale, {
      note: 'a Gaussian at one mode of a posterior with many symmetric ones: overconfident away from the data',
      quantities: BNN_TEST_INPUTS.map((_, j) => `fstar[${j}]`),
    }),
    sviPlan(scale, {
      note: 'a mean-field normal on the weights: overconfident, the output spread too small',
      quantities: BNN_TEST_INPUTS.map((_, j) => `fstar[${j}]`),
    }),
  ],
})
