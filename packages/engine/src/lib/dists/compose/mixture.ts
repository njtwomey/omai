/**
 * Finite mixture distributions, p(x) = sum_k w_k p_k(x): a mixture of arbitrary components (scipy.stats.Mixture), a
 * mixture of one batched family (torch.distributions.MixtureSameFamily), and the posterior component probabilities
 * (responsibilities) that form the E step of expectation maximisation.
 */

import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  argmax,
  broadcastTo,
  cast,
  exp,
  expandDims,
  log,
  logSoftmax,
  logsumexp,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  shapesEqual,
  sliceAxis,
  softmax,
  sqrt,
  square,
  squeeze,
  stack,
  sub,
  sum,
  type TensorLike,
  takeAlongAxis,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { type Bulk, numericEntropy, numericMode, numericPpf } from '../_numeric.ts'
import { implicitReparam } from '../_sampling.ts'
import { categorical } from '../discrete/categorical.ts'
import {
  type Distribution,
  type DistributionSpec,
  defineDistribution,
  type Support,
  unavailable,
} from '../distribution.ts'

/** Mixture weights: probabilities or logits along the last axis (one entry per component). */
export type MixtureWeights = { readonly probs: TensorLike } | { readonly logits: TensorLike }

/** A finite mixture: a distribution that also exposes its log weights and per-component log densities. */
export interface Mixture extends Distribution {
  /** Normalised log weights `[..., K]`. */
  readonly logWeights: Value
  /** log p_k(x) for every component, `[..., K]`. */
  componentLogProbs(x: TensorLike): Value
}

function logWeightsOf(weights: MixtureWeights): Value {
  const raw =
    'logits' in weights ? (promoteFloat(weights.logits)[0] as Value) : log(promoteFloat(weights.probs)[0] as Value)
  if (raw.shape.length === 0) throw new WhiteboxError('shape', 'mixture: weights need a component axis')
  return logSoftmax(raw, -1)
}

/** Concrete scalar ends of a support (where a component's density may have a kink). */
function endsOf(support: Support): number[] {
  const scalar = (v: Value): number[] => {
    const values = toFlat(primalOf(v))
    return values.length === 1 && Number.isFinite(values[0]) ? [values[0] as number] : []
  }
  switch (support.kind) {
    case 'positive':
    case 'nonNegative':
      return [0]
    case 'unitInterval':
      return [0, 1]
    case 'interval':
      return [...scalar(support.low), ...scalar(support.high)]
    case 'greaterThan':
      return scalar(support.low)
    case 'lessThan':
      return scalar(support.high)
    default:
      return []
  }
}

/** The quadrature bulk for a univariate mixture's entropy, from its support, its components' ends and moments. */
function bulkOf(support: Support, ends: readonly number[], mean: Value, sd: Value): Bulk {
  if (support.kind === 'real' && ends.length > 0) {
    const breaks = [...new Set(ends)].sort((a, b) => a - b)
    return { kind: 'segments', breaks, scale: sd }
  }
  switch (support.kind) {
    case 'positive':
    case 'nonNegative':
      return { kind: 'halfLine', low: 0, scale: mean }
    case 'unitInterval':
      return { kind: 'finite', low: 0, high: 1 }
    case 'interval':
      return { kind: 'finite', low: support.low, high: support.high }
    default:
      return { kind: 'real', center: mean, scale: sd }
  }
}

/**
 * The parts every mixture shares, given log weights `[..., K]`, per-component log densities, CDFs, draws, means and
 * variances with the component axis last before the event axes.
 */
interface Parts {
  readonly name: string
  readonly params: Readonly<Record<string, Value>>
  readonly support: Support
  readonly batchShape: readonly number[]
  readonly eventShape: readonly number[]
  readonly logWeights: Value
  /** Ends of the components' supports, where the mixture density may have kinks. */
  readonly ends: readonly number[]
  componentLogProbs(x: Value): Value
  componentCdfs(x: Value): Value
  componentSfs(x: Value): Value
  /** Draws of every component, `[...shape, ...batch, K, ...event]`. */
  componentSamples(key: Key, shape: readonly number[]): Value
  componentMeans(): Value
  componentVariances(): Value
  componentModes(): Value
}

function buildMixture(parts: Parts): Mixture {
  const eventRank = parts.eventShape.length
  const scalar = eventRank === 0
  const lw = parts.logWeights
  const k = lw.shape[lw.shape.length - 1] ?? 0
  // Weights lifted to broadcast against `[..., K, ...event]`.
  const lifted = (): Value => reshape(exp(lw), [...lw.shape, ...parts.eventShape.map(() => 1)])
  const componentAxis = -1 - eventRank
  const mean = (): Value => sum(mul(lifted(), parts.componentMeans()), componentAxis)
  const variance = (): Value => {
    const m = parts.componentMeans()
    const second = sum(mul(lifted(), add(parts.componentVariances(), square(m))), componentAxis)
    return sub(second, square(mean()))
  }
  const draw = (key: Key, shape: readonly number[]): Value => {
    const all = parts.componentSamples(foldIn(key, 0), shape)
    // One component per draw and batch member (weights broadcast to the full batch first).
    const choice = categorical({ logits: broadcastTo(lw, [...parts.batchShape, k]) }).sample(foldIn(key, 1), shape)
    const index = reshape(cast(choice, 'int32'), [...choice.shape, 1, ...parts.eventShape.map(() => 1)])
    const target = [...choice.shape, 1, ...parts.eventShape]
    return squeeze(takeAlongAxis(all, broadcastIndex(index, target), componentAxis), componentAxis)
  }
  const spec: DistributionSpec = {
    name: parts.name,
    params: parts.params,
    support: parts.support,
    batchShape: parts.batchShape,
    eventShape: parts.eventShape,
    logProb: (x) => logsumexp(add(lw, parts.componentLogProbs(x)), -1),
    sample: draw,
    mean,
    variance,
    mode: () => {
      // The component mode with the highest mixture density, refined by a local search for scalar events.
      const candidates = parts.componentModes()
      const dens = densityAtModes(candidates)
      const best = argmax(dens, -1)
      const pick = (v: Value): Value =>
        squeeze(
          takeAlongAxis(
            v,
            broadcastIndex(reshape(cast(best, 'int32'), [...best.shape, 1, ...parts.eventShape.map(() => 1)]), [
              ...best.shape,
              1,
              ...parts.eventShape,
            ]),
            componentAxis,
          ),
          componentAxis,
        )
      const start = pick(candidates)
      if (!scalar) return start
      const spread = sqrt(pick(parts.componentVariances()))
      return numericMode(self, () => self, sub(start, spread), add(start, spread))
    },
    entropy: () =>
      scalar
        ? numericEntropy(self, bulkOf(parts.support, parts.ends, mean(), sqrt(variance())))
        : unavailable(parts.name, 'entropy (no closed form for a multivariate mixture; estimate it by Monte Carlo)'),
  }
  // Mixture log density at each component's mode: [..., K].
  const densityAtModes = (modes: Value): Value => {
    const pieces: Value[] = []
    for (let j = 0; j < k; j++) pieces.push(self.logProb(sliceComponent(modes, j, componentAxis)))
    return stack(pieces, -1)
  }
  const withScalar: DistributionSpec = scalar
    ? {
        ...spec,
        cdf: (x) => sum(mul(exp(lw), parts.componentCdfs(x)), -1),
        sf: (x) => sum(mul(exp(lw), parts.componentSfs(x)), -1),
        ppf: (q) => numericPpf(self, () => self, q),
        // Implicit reparameterisation through the mixture CDF (Graves, 2016; Figurnov et al., 2018).
        rsample: (key, shape) => implicitReparam(draw(key, shape), self.cdf, self.prob),
      }
    : spec
  const base = defineDistribution(withScalar)
  const self: Mixture = {
    ...base,
    logWeights: lw,
    componentLogProbs: (x) => parts.componentLogProbs(promoteFloat(x)[0] as Value),
  }
  return self
}

function sliceComponent(v: Value, j: number, axis: number): Value {
  return squeeze(sliceAxis(v, axis, j, j + 1), axis)
}

function broadcastIndex(index: Value, shape: readonly number[]): Value {
  return shapesEqual(index.shape, shape) ? index : broadcastTo(index, shape)
}

/** The support of a mixture: the components' common support, or the real line when they differ. */
function widest(supports: readonly Support[]): Support {
  const first = supports[0] ?? { kind: 'real' }
  return supports.every((s) => s.kind === first.kind) ? first : { kind: 'real' }
}

/**
 * A finite mixture of distributions with the same batch and event shapes (scipy.stats.Mixture): p(x) =
 * sum_k w_k p_k(x). Univariate mixtures have a CDF, numerical quantiles and mode, an entropy by quadrature, and an
 * `rsample` by implicit reparameterisation through the mixture CDF, so draws differentiate in the weights and the
 * component parameters.
 *
 * @param weights - `{ probs }` or `{ logits }`, `[..., K]`.
 * @param components - The K components.
 * @returns The mixture (with `logWeights` and `componentLogProbs`).
 * @throws WhiteboxError with code `shape` when the components disagree in shape or number.
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { mixture } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * const m = mixture({ probs: [0.3, 0.7] }, [normal(0, 1), normal(3, 0.5)])
 * Math.abs(item(m.mean()) - 2.1) < 1e-14 // => true
 * ```
 */
export function mixture(weights: MixtureWeights, components: readonly Distribution[]): Mixture {
  const lw = logWeightsOf(weights)
  const k = lw.shape[lw.shape.length - 1] ?? 0
  const first = components[0]
  if (first === undefined || components.length !== k)
    throw new WhiteboxError('shape', `mixture: ${k} weights but ${components.length} components`)
  for (const c of components)
    if (!shapesEqual(c.eventShape, first.eventShape))
      throw new WhiteboxError('shape', 'mixture: components must share an event shape')
  const eventShape = [...first.eventShape]
  const along = (f: (c: Distribution) => Value): Value => stack(components.map(f), -1 - eventShape.length)
  const lastAxis = (f: (c: Distribution) => Value): Value => stack(components.map(f), -1)
  return buildMixture({
    name: 'mixture',
    params: { logits: lw },
    support: widest(components.map((c) => c.support)),
    ends: components.flatMap((c) => endsOf(c.support)),
    batchShape: [...first.batchShape],
    eventShape,
    logWeights: lw,
    componentLogProbs: (x) => lastAxis((c) => c.logProb(x)),
    componentCdfs: (x) => lastAxis((c) => c.cdf(x)),
    componentSfs: (x) => lastAxis((c) => c.sf(x)),
    componentSamples: (key, shape) => {
      const draws = components.map((c, j) => c.sample(foldIn(key, j), shape))
      return stack(draws, -1 - eventShape.length)
    },
    componentMeans: () => along((c) => c.mean()),
    componentVariances: () => along((c) => c.variance()),
    componentModes: () => along((c) => c.mode()),
  })
}

/**
 * A mixture whose components are one batched distribution (torch.distributions.MixtureSameFamily): the component's
 * last batch axis indexes the K components, mixed by `weights`.
 *
 * @param weights - `{ probs }` or `{ logits }`, `[..., K]`.
 * @param component - A distribution with batch shape `[..., K]`.
 * @returns The mixture, with batch shape `[...]`.
 * @throws WhiteboxError with code `shape` when the component axis disagrees with the weights.
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { mixtureSameFamily } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * const m = mixtureSameFamily({ probs: [0.5, 0.5] }, normal([-1, 1], [1, 1]))
 * item(m.mean()) // => 0
 * ```
 */
export function mixtureSameFamily(weights: MixtureWeights, component: Distribution): Mixture {
  const lw = logWeightsOf(weights)
  const k = lw.shape[lw.shape.length - 1] ?? 0
  const cb = component.batchShape
  if ((cb[cb.length - 1] ?? -1) !== k)
    throw new WhiteboxError('shape', `mixtureSameFamily: the component batch must end in ${k}`)
  const eventShape = [...component.eventShape]
  const eventRank = eventShape.length
  // Insert the component axis into x, before its event axes.
  const lift = (x: Value): Value => expandDims(x, x.shape.length - eventRank)
  return buildMixture({
    name: 'mixtureSameFamily',
    params: { ...component.params, logits: lw },
    support: component.support,
    ends: [],
    batchShape: cb.slice(0, -1),
    eventShape,
    logWeights: lw,
    componentLogProbs: (x) => component.logProb(lift(x)),
    componentCdfs: (x) => component.cdf(lift(x)),
    componentSfs: (x) => component.sf(lift(x)),
    componentSamples: (key, shape) => component.sample(key, shape),
    componentMeans: () => component.mean(),
    componentVariances: () => component.variance(),
    componentModes: () => component.mode(),
  })
}

/**
 * The posterior probabilities of the components given observations (the E step of EM): softmax over k of
 * log w_k + log p_k(x).
 *
 * @param mix - A mixture from {@link mixture} or {@link mixtureSameFamily}.
 * @param x - Observations.
 * @returns Responsibilities `[..., K]`, summing to one over the last axis.
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { mixture, responsibilities } from 'whitebox-engine/dists/compose'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(responsibilities(mixture({ probs: [0.5, 0.5] }, [normal(-1, 1), normal(1, 1)]), 0)).map((v) => Math.round(v * 1e12) / 1e12) // => [0.5, 0.5]
 * ```
 */
export function responsibilities(mix: Mixture, x: TensorLike): Value {
  return softmax(add(mix.logWeights, mix.componentLogProbs(x)), -1)
}
