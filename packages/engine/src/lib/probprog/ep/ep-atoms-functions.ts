/**
 * The setups of expectation propagation with a function of a random value: an exponential, a hyperbolic tangent, a
 * Gaussian whose scale is the exponential of a latent (the stochastic volatility step), a product of two Gaussians and the
 * maximum of two. Each is a latent Gaussian pushed through the function and observed through noise, so the exact posterior
 * is a one- or two-dimensional quadrature (the model written as a program, which runs whether or not EP applies).
 */

import { dict, real } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type AtomParts, defineAtom, type EpAtom } from './ep-atoms.ts'
import { tolerance } from './ep-tolerance.ts'

const G = 61

function expProblem(k: Config): AtomParts {
  const y = configNumber(k, 'y')
  // region exp
  const definition = defineModel('exp-of-gaussian', (m) => {
    const x = m.latent('x', d.normal(0, 1))
    m.observe('y', d.normal(ops.exp(x), 0.3), m.data('y'))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y },
      quantities: [{ name: 'x', label: 'x', support: 'real' }],
    },
    present: {
      equations: [
        'x ~ N(0, 1)',
        'g = exp(x)   (a unary function of a random value)',
        'y ~ N(g, 0.3²)',
        `observe y = ${y}`,
      ],
    },
  }
}

/**
 * A Gaussian pushed through the exponential and observed: the unary-function rule (`apply`), whose message back to the
 * input is the tilted moment of the input under the cavity of the output.
 *
 * @index estimator EP atom: exp of a Gaussian (unary function of a random value).
 * @example
 * ```ts
 * import { expAtom } from 'omai-engine/probprog/ep'
 * expAtom.name // => 'exp'
 * ```
 */
export const expAtom: EpAtom = defineAtom({
  name: 'exp',
  title: 'Exp of a Gaussian',
  what: 'A Gaussian x, the value g = exp(x) observed through Gaussian noise: the unary-function factor, whose exact posterior is skewed (log-normal likelihood against a Gaussian prior).',
  factors: ['prior (gaussian)', 'apply: exp of a random value', 'gaussian likelihood'],
  family: 'functions',
  reference: 'grid',
  tolerance: tolerance('coupled', 'the posterior is skewed; EP matches its mean and variance with a Gaussian'),
  evidence: { nats: 1e-3, reason: 'the apply rule (exp) by quadrature' },
  knobs: dict({ y: real(0.2, 8, { default: 2.5, step: 0.1, label: 'observed y' }) }),
  parts: expProblem,
})

function tanhProblem(k: Config): AtomParts {
  const y = configNumber(k, 'y')
  // region tanh
  const definition = defineModel('tanh-of-gaussian', (m) => {
    const x = m.latent('x', d.normal(0, 1))
    m.observe('y', d.normal(ops.tanh(x), 0.2), m.data('y'))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y },
      quantities: [{ name: 'x', label: 'x', support: 'real' }],
    },
    present: {
      equations: ['x ~ N(0, 1)', 'g = tanh(x)   (a squashing unit)', 'y ~ N(g, 0.2²)', `observe y = ${y}`],
    },
  }
}

/**
 * A Gaussian pushed through tanh and observed: the neural-network unit of probabilistic backpropagation.
 *
 * @index estimator EP atom: tanh of a Gaussian (a squashing unit).
 * @example
 * ```ts
 * import { tanhAtom } from 'omai-engine/probprog/ep'
 * tanhAtom.family // => 'functions'
 * ```
 */
export const tanhAtom: EpAtom = defineAtom({
  name: 'tanh',
  title: 'Tanh of a Gaussian',
  what: 'A Gaussian x, the value g = tanh(x) observed through Gaussian noise: the squashing unit of a Bayesian neural network, saturating at plus and minus one.',
  factors: ['prior (gaussian)', 'apply: tanh of a random value', 'gaussian likelihood'],
  family: 'functions',
  reference: 'grid',
  tolerance: tolerance(
    'lumpy',
    'the posterior is skewed by the saturation; EP matches its mean and variance with a Gaussian',
  ),
  evidence: { nats: 1e-3, reason: 'the apply rule (tanh) by quadrature' },
  knobs: dict({ y: real(-0.95, 0.95, { default: 0.8, step: 0.05, label: 'observed y' }) }),
  parts: tanhProblem,
})

function logScaleProblem(k: Config): AtomParts {
  const y = configNumber(k, 'y')
  // region log-scale
  const definition = defineModel('log-scale', (m) => {
    const h = m.latent('h', d.normal(0, 1))
    m.observe('y', d.normal(0, ops.exp(ops.div(h, 2))), m.data('y'))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y },
      quantities: [{ name: 'h', label: 'h (log variance)', support: 'real' }],
    },
    present: {
      equations: [
        'h ~ N(0, 1)',
        'y ~ N(0, exp(h / 2)²)   (the scale is the exponential of a Gaussian state)',
        `observe y = ${y}`,
      ],
    },
  }
}

/**
 * An observation whose scale is the exponential of a latent Gaussian: the stochastic-volatility step.
 *
 * @index estimator EP atom: a Gaussian with the exponential of a latent as its scale (stochastic volatility step).
 * @example
 * ```ts
 * import { logScaleAtom } from 'omai-engine/probprog/ep'
 * logScaleAtom.name // => 'log-scale'
 * ```
 */
export const logScaleAtom: EpAtom = defineAtom({
  name: 'log-scale',
  title: 'Gaussian with an exp(h) scale',
  what: 'A latent h and one observation y ~ N(0, exp(h/2)²): the observation says how large the variance is, and the likelihood of h is not Gaussian (the stochastic volatility step).',
  factors: ['prior (gaussian)', 'gaussian-log-scale'],
  family: 'functions',
  reference: 'grid',
  tolerance: tolerance(
    'skew-normal',
    'the likelihood of h is a skewed, non-Gaussian bump that EP projects onto a Gaussian',
  ),
  evidence: { nats: 1e-3, reason: 'the Gaussian with a log scale by quadrature' },
  knobs: dict({ y: real(-4, 4, { default: 1.5, step: 0.1, label: 'observed y' }) }),
  parts: logScaleProblem,
})

function productProblem(k: Config): AtomParts {
  const y = configNumber(k, 'y')
  // region product
  const definition = defineModel('product', (m) => {
    const a = m.latent('a', d.normal(2, 1))
    const b = m.latent('b', d.normal(1, 1))
    m.observe('y', d.normal(ops.mul(a, b), 0.3), m.data('y'))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'a', label: 'a', support: 'real' },
        { name: 'b', label: 'b', support: 'real' },
      ],
    },
    reference: {
      grid: { points: G, maxRefinements: 0 },
    },
    present: {
      equations: [
        'a ~ N(2, 1²)',
        'b ~ N(1, 1²)',
        'y ~ N(a b, 0.3²)   (a product of two random values)',
        `observe y = ${y}`,
      ],
      pair: ['a', 'b'],
    },
  }
}

/**
 * The product of two Gaussians observed: a banana-shaped posterior (a b is fixed by the observation, so a and b trade off),
 * and the numeric product factor.
 *
 * @index estimator EP atom: the product of two Gaussians (numeric product rule, a banana posterior).
 * @example
 * ```ts
 * import { productAtom } from 'omai-engine/probprog/ep'
 * productAtom.name // => 'product'
 * ```
 */
export const productAtom: EpAtom = defineAtom({
  name: 'product',
  title: 'Product of two Gaussians',
  what: 'Two Gaussians a and b and an observation of their product a b: the product factor, a banana posterior along a b = y that a belief per variable cannot follow.',
  factors: ['prior (gaussian) x2', 'product (numeric)', 'gaussian likelihood'],
  family: 'functions',
  reference: 'grid',
  tolerance: tolerance(
    'lumpiest',
    'the posterior is curved and EP keeps one Gaussian per variable, so it can match the margins and not the bend',
  ),
  evidence: { nats: 0.02, reason: 'the product of two Gaussians: loopy, moment matched' },
  knobs: dict({ y: real(-3, 4, { default: 2, step: 0.1, label: 'observed y' }) }),
  ep: { damping: 0.5 },
  parts: productProblem,
})

function maxProblem(k: Config): AtomParts {
  const y = configNumber(k, 'y')
  // region max
  const definition = defineModel('max-of-two', (m) => {
    const a = m.latent('a', d.normal(0, 1))
    const b = m.latent('b', d.normal(0.5, 1))
    m.observe('y', d.normal(ops.maximum(a, b), 0.3), m.data('y'))
  })
  // endregion
  return {
    build: {
      definition,
      data: { y },
      quantities: [
        { name: 'a', label: 'a', support: 'real' },
        { name: 'b', label: 'b', support: 'real' },
      ],
    },
    reference: {
      grid: { points: G, maxRefinements: 0 },
    },
    present: {
      equations: [
        'a ~ N(0, 1²)',
        'b ~ N(0.5, 1²)',
        'y ~ N(max(a, b), 0.3²)   (the maximum of two random values)',
        `observe y = ${y}`,
      ],
      pair: ['a', 'b'],
    },
  }
}

/**
 * The maximum of two Gaussians observed: the max factor, whose posterior has an L shape (one of the two is at the observed
 * value and the other is below it).
 *
 * @index estimator EP atom: the maximum of two Gaussians (max factor, an L-shaped posterior).
 * @example
 * ```ts
 * import { maxAtom } from 'omai-engine/probprog/ep'
 * maxAtom.name // => 'max'
 * ```
 */
export const maxAtom: EpAtom = defineAtom({
  name: 'max',
  title: 'Max of two Gaussians',
  what: 'Two Gaussians a and b and an observation of max(a, b): the max factor; the posterior is an L, one of the two sits at the observation and the other is free below it.',
  factors: ['prior (gaussian) x2', 'max', 'gaussian likelihood'],
  family: 'functions',
  reference: 'grid',
  tolerance: tolerance(
    'lumpy',
    'the posterior is L-shaped and EP matches the mean and variance of each margin with a Gaussian',
  ),
  evidence: { nats: 0.01, reason: 'the max of two Gaussians: moment matched' },
  knobs: dict({ y: real(-2, 3, { default: 1.5, step: 0.1, label: 'observed y' }) }),
  parts: maxProblem,
})
