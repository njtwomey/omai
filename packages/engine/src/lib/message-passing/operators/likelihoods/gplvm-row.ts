/**
 * The latent-variable row: one observed point `y` of a latent variable model written with random Fourier features, as ONE factor over the
 * point's latent coordinates and the weights of every output. Each of the `D` outputs is `y_d = w_d . phi(x) + e` with the features
 * `phi_m(x) = cos(omega_m . x + b_m)` of fixed frequencies and phases; the coordinates `x` are one or two scalar Gaussian ports and the
 * weights of output `d` are one Gaussian vector port `w_d` of `M` weights each. Given `x` the row is linear and Gaussian in the weights,
 * so they are integrated out in closed form at each value of `x`; what is left of the tilted distribution is a distribution over `x`
 * alone, which the rule lays on a zoomed grid of nodes (`numerics/_node-grid.ts`). The message to each coordinate is the moment match of
 * the weighted nodes (the marginal mean and variance); the message to each output's weights is the moment match of the mixture over the
 * nodes of that output's conditional Gaussian (its mean and its full `M x M` covariance).
 *
 * What it keeps and what it drops. The point and its `D` outputs are one factor, so there is no loop through the features of a point
 * (the scalar route has a product, a cosine and a sum per output and feature, and its messages cycle); with a Gaussian prior on each
 * coordinate and on each output's weights the graph is a star around the weights and a tree. What the messages drop is the dependence
 * between ports that the shared coordinate induces: the weights of two outputs are correlated given the data of a point (both move with
 * `x`), and each output's message carries only its own marginal; the two coordinates of a point are two scalar ports, so their joint is
 * reduced to the two marginals. Every port needs a proper cavity: with no prior there is nothing to place the grid on.
 */

import { errorText, keyOf, memoised } from '../../core/_messages.ts'
import { allProper, etaOf, gaussianOf, massOf, numberParam, vectorParam } from '../../core/args.ts'
import { defineOperator, siteRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gaussianMvMoments } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, Params, Port } from '../../core/types.ts'
import { zeroArray, zeroFloat64 } from '../../numerics/_arrays.ts'
import { spd } from '../../numerics/_dense.ts'
import { zoomNodeGrid } from '../../numerics/_node-grid.ts'

/** The most coordinates per point the grid lays nodes over. */
export const GPLVM_MOST_COORDS = 2
/** The most weights (`outputs * features`) one row may have: the cost of a row is the number of nodes times `outputs * features^2`. */
export const GPLVM_MOST_WEIGHTS = 1024
/** Nodes per coordinate of each grid, by the number of coordinates (the zoom lays a second grid of the same size over the mass). */
const POINTS = [0, 241, 61] as const
/** The width of the window of a coordinate's cavity in standard deviations, either side of the mean, by the number of coordinates. */
const REACH = [0, 8, 6] as const
/** The nodes whose normalised weight is below this are left out of the mixture of the weights. */
const NEGLIGIBLE = 1e-12

/** What the factor reads from its parameters, checked. */
interface Form {
  readonly coords: number
  readonly features: number
  readonly outputs: number
  readonly omega: Float64Array
  readonly phase: Float64Array
  readonly noise: number
  readonly y: Float64Array
}

/** The parameters of a factor, checked against each other. */
function formOf(params: Params): Form {
  const coords = numberParam(params, 'coords')
  const features = numberParam(params, 'features')
  const outputs = numberParam(params, 'outputs')
  const omega = vectorParam(params, 'omega')
  const phase = vectorParam(params, 'phase')
  const y = vectorParam(params, 'y')
  const noise = numberParam(params, 'noise')
  if (!(coords >= 1 && coords <= GPLVM_MOST_COORDS))
    throw new ParameterError(
      `gplvm-row: ${coords} coordinates per point; the grid over the coordinates is built for 1 to ${GPLVM_MOST_COORDS}`,
    )
  if (omega.length !== coords * features)
    throw new ParameterError(`gplvm-row: omega must have ${coords} x ${features} entries (a row per coordinate)`)
  if (phase.length !== features) throw new ParameterError(`gplvm-row: phase must have ${features} entries`)
  if (y.length !== outputs) throw new ParameterError(`gplvm-row: y must have ${outputs} entries`)
  if (!(noise > 0)) throw new ParameterError('gplvm-row: the noise standard deviation must be positive')
  if (outputs * features > GPLVM_MOST_WEIGHTS)
    throw new ParameterError(
      `gplvm-row: ${outputs} outputs of ${features} features are ${outputs * features} weights, past the ${GPLVM_MOST_WEIGHTS} a row takes`,
    )
  return {
    coords,
    features,
    outputs,
    omega: Float64Array.from(omega),
    phase: Float64Array.from(phase),
    noise,
    y: Float64Array.from(y),
  }
}

/** The cavities of the weights of every output: the means (`outputs x features`) and the covariances (`outputs x features x features`) side by side. */
interface Cavities {
  readonly means: Float64Array
  readonly covariances: Float64Array
}

/** The cavity of the weights of each output: the mean and covariance of the vector port of that output. */
function cavitiesOf(args: readonly Arg[], coords: number, outputs: number, features: number): Cavities {
  const means = zeroFloat64(outputs * features)
  const covariances = zeroFloat64(outputs * features * features)
  for (let d = 0; d < outputs; d++) {
    try {
      const { mean, covariance } = gaussianMvMoments(etaOf(args[coords + d] as Arg), features)
      means.set(mean, d * features)
      covariances.set(covariance, d * features * features)
    } catch (e) {
      if (e instanceof ImproperMessageError)
        throw new ImproperMessageError(`gplvm-row: the cavity of the weights of output ${d} is not positive definite`)
      throw e
    }
  }
  return { means, covariances }
}

/** What one tilt found: the log mass of the factor under the normalised cavities, and the messages' targets. */
interface Tilt {
  readonly logMass: number
  /** The tilted mean and variance of each coordinate. */
  readonly mean: readonly number[]
  readonly variance: readonly number[]
  /** The tilted natural parameters of the weights of each output, `[h, J]` of dimension `features`. */
  readonly natural: readonly Eta[]
}

/** The features of a point at the coordinates `x`: `phi_m = cos(omega_m . x + b_m)`, written into `phi`. */
function featuresAt(form: Form, x: Float64Array, phi: Float64Array): void {
  const { coords, features, omega, phase } = form
  for (let m = 0; m < features; m++) {
    let angle = phase[m] as number
    for (let i = 0; i < coords; i++) angle += (omega[i * features + m] as number) * (x[i] as number)
    phi[m] = Math.cos(angle)
  }
}

/** The tilted distribution of the factor against the cavities in the context. */
function tiltOf(c: OperatorContext): Tilt {
  const form = formOf(c.params)
  const { coords, features, outputs, noise, y } = form
  const cavity = c.args.slice(0, coords).map((a) => gaussianOf(a))
  const { means, covariances } = cavitiesOf(c.args, coords, outputs, features)
  const noiseVariance = noise * noise
  const phi = zeroFloat64(features)
  const logConstant = -0.5 * Math.log(2 * Math.PI)
  const block = features * features
  // The log of the cavities of the coordinates times the marginal likelihood of the point at the coordinates `x`: each output's
  // observation is Gaussian with the predictive mean `phi . m` and variance `phi' V phi + noise^2` of its cavity weights.
  const logw = (x: Float64Array): number => {
    let total = 0
    for (let i = 0; i < coords; i++) {
      const g = cavity[i] as { readonly mean: number; readonly variance: number }
      total += logConstant - 0.5 * Math.log(g.variance) - (0.5 * ((x[i] as number) - g.mean) ** 2) / g.variance
    }
    featuresAt(form, x, phi)
    for (let d = 0; d < outputs; d++) {
      const mAt = d * features
      const vAt = d * block
      let mu = 0
      let quad = 0
      for (let a = 0; a < features; a++) {
        const pa = phi[a] as number
        mu += pa * (means[mAt + a] as number)
        // the quadratic form of the symmetric covariance from its lower triangle
        let row = 0
        const at = vAt + a * features
        for (let b = 0; b < a; b++) row += (covariances[at + b] as number) * (phi[b] as number)
        quad += pa * (2 * row + (covariances[at + a] as number) * pa)
      }
      const s = quad + noiseVariance
      const e = (y[d] as number) - mu
      total += logConstant - 0.5 * Math.log(s) - (0.5 * e * e) / s
    }
    return total
  }
  const points = POINTS[coords] as number
  const reach = REACH[coords] as number
  const grid = zoomNodeGrid(
    cavity.map((g) => ({ lo: g.mean - reach * Math.sqrt(g.variance), hi: g.mean + reach * Math.sqrt(g.variance) })),
    points,
    logw,
  )
  const { nodes, weights: w, total, count } = grid
  // The marginal mean and variance of each coordinate under the weighted nodes.
  const mean = zeroArray(coords)
  const variance = zeroArray(coords)
  for (let i = 0; i < coords; i++) {
    let m = 0
    for (let k = 0; k < count; k++) m += (w[k] as number) * (nodes[k * coords + i] as number)
    m /= total
    let v = 0
    for (let k = 0; k < count; k++) v += (w[k] as number) * ((nodes[k * coords + i] as number) - m) ** 2
    mean[i] = m
    variance[i] = v / total
  }
  // The mixture over the nodes of the weights' conditionals: for output d at a node, the cavity updated by one Gaussian observation of
  // `w_d . phi` is `N(m + g r, V - g g' / s)` with `g = V phi`, `s = phi' V phi + noise^2` and `r = (y - phi . m) / s`; the mixture has
  // mean `m + sum p g r` and covariance `V + sum p (r^2 - 1/s) g g' - (sum p g r)(sum p g r)'`.
  const shift = zeroFloat64(outputs * features)
  const spread = zeroFloat64(outputs * block)
  const g = zeroFloat64(features)
  const x = zeroFloat64(coords)
  for (let k = 0; k < count; k++) {
    const p = (w[k] as number) / total
    if (p < NEGLIGIBLE) continue
    for (let i = 0; i < coords; i++) x[i] = nodes[k * coords + i] as number
    featuresAt(form, x, phi)
    for (let d = 0; d < outputs; d++) {
      const mAt = d * features
      const vAt = d * block
      let mu = 0
      let s = noiseVariance
      for (let a = 0; a < features; a++) {
        mu += (phi[a] as number) * (means[mAt + a] as number)
        let row = 0
        const at = vAt + a * features
        for (let b = 0; b < features; b++) row += (covariances[at + b] as number) * (phi[b] as number)
        g[a] = row
        s += (phi[a] as number) * row
      }
      const r = ((y[d] as number) - mu) / s
      const weight = p * (r * r - 1 / s)
      for (let a = 0; a < features; a++) {
        const ga = g[a] as number
        shift[mAt + a] = (shift[mAt + a] as number) + p * r * ga
        const at = vAt + a * features
        for (let b = 0; b <= a; b++) spread[at + b] = (spread[at + b] as number) + weight * ga * (g[b] as number)
      }
    }
  }
  const natural: Eta[] = []
  for (let d = 0; d < outputs; d++) {
    const mAt = d * features
    const vAt = d * block
    const tilted = zeroFloat64(block)
    const centre = zeroFloat64(features)
    for (let a = 0; a < features; a++) {
      centre[a] = (means[mAt + a] as number) + (shift[mAt + a] as number)
      for (let b = 0; b <= a; b++) {
        const value =
          (covariances[vAt + a * features + b] as number) +
          (spread[vAt + a * features + b] as number) -
          (shift[mAt + a] as number) * (shift[mAt + b] as number)
        tilted[a * features + b] = value
        tilted[b * features + a] = value
      }
    }
    const factor = spd(tilted, features)
    if (factor === undefined)
      throw new ImproperMessageError(
        `gplvm-row: the tilted covariance of the weights of output ${d} is not positive definite`,
      )
    const precision = factor.inverse()
    const h = zeroArray(features)
    for (let a = 0; a < features; a++)
      for (let b = 0; b < features; b++)
        h[a] = (h[a] as number) + (precision[a * features + b] as number) * (centre[b] as number)
    natural.push([...h, ...precision])
  }
  return { logMass: grid.top + Math.log(total * grid.volume), mean, variance, natural }
}

/** The tilt for the context, kept in the engine's cache for the ports of the factor and its evidence. */
function tilt(c: OperatorContext): Tilt {
  const inputs = [c.args.map((a) => a.eta), c.params]
  return memoised(c, `gplvm-row:${keyOf(inputs)}`, () => tiltOf(c))
}

/** The ports of a row: one scalar Gaussian per coordinate, then one Gaussian vector of the weights per output. */
function portsOf(params: Params): readonly Port[] {
  const coords = Math.max(1, Math.floor(numberParam(params, 'coords', 1)))
  const outputs = Math.max(1, Math.floor(numberParam(params, 'outputs', 1)))
  return [
    ...Array.from({ length: coords }, (_, i): Port => ({ name: `x${i}`, accepts: ['gaussian'] })),
    ...Array.from({ length: outputs }, (_, d): Port => ({ name: `w${d}`, accepts: ['gaussian-mv'] })),
  ]
}

/**
 * One point of a latent variable model with random Fourier features: `y_d = w_d . cos(omega x + b) + e` for `D` outputs, as one factor on the
 * point's coordinates `x` (one or two Gaussian variables) and the weights of each output (`D` Gaussian vectors of `M`). The weights are
 * integrated out in closed form at each node of a zoomed grid over `x`; the message to a coordinate is the moment match of the weighted
 * nodes, the message to the weights of an output the moment match of the mixture of that output's conditional Gaussian (the correlation
 * of two outputs' weights through the shared coordinate is dropped). The evidence is the log of the node-averaged likelihood. The
 * parameters are `coords`, `features`, `outputs`, the frequencies `omega` (row-major, a row per coordinate), the phases `phase`, the noise
 * standard deviation `noise` and the observed point `y`. It has no variational rule (the mean field of the coordinates and the weights
 * has no closed-form expected log of the sum over features per row; the scalar route has one) and no power.
 *
 * @concept factor-operators
 * @index structure One GPLVM point with Fourier features: its coordinate and all outputs' weights.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, gaussianMv, gaussianMvEta, gplvmRow, messageArg } from 'omai-engine/message-passing'
 * const x = messageArg(gaussian, gaussianEta(0, 1))
 * const w = messageArg(gaussianMv(2), gaussianMvEta([0, 0], [1, 0, 0, 1]))
 * const params = { coords: 1, features: 2, outputs: 1, omega: [1, 2], phase: [0, 0], noise: 0.1, y: [0.5] }
 * gplvmRow.ep?.({ mode: 'ep', to: 0, args: [x, w], params, power: 1 }).message.length // => 2
 * ```
 */
export const gplvmRow: FactorOperator = defineOperator({
  name: 'gplvm-row',
  description:
    'a point y_d = w_d . cos(omega x + b) + noise (d = 1..D) of a latent variable model with Fourier features: one joint EP site over its coordinates and the weights of every output',
  ports: portsOf,
  params: {
    coords: param.integer({ min: 1, max: GPLVM_MOST_COORDS, example: 1 }),
    features: param.integer({ min: 1, example: 2 }),
    outputs: param.integer({ min: 1, example: 1 }),
    omega: param.vector({ example: [1, 2] }),
    phase: param.vector({ example: [0, 0] }),
    noise: param.positive({ example: 0.1 }),
    y: param.vector({ example: [0.5] }),
  },
  check: (params, families) => {
    try {
      const form = formOf(params)
      if (families.length !== form.coords + form.outputs)
        return 'gplvm-row: the coordinates and the weights of each output are one port each'
      for (let d = 0; d < form.outputs; d++) {
        const size = families[form.coords + d]?.dim ?? 0
        if (size !== form.features)
          return `gplvm-row: the weights of output ${d} have ${size} entries, not ${form.features}`
      }
      return undefined
    } catch (e) {
      return errorText(e)
    }
  },
  rules: {
    ep: siteRule((c) => {
      const t = tilt(c)
      const coords = formOf(c.params).coords
      if (c.to < coords) return { mean: t.mean[c.to] as number, variance: t.variance[c.to] as number }
      return { natural: t.natural[c.to - coords] as Eta }
    }),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      let total = tilt(c).logMass
      for (const a of c.args) total += massOf(a)
      return total
    },
  },
  noRule: {
    vmp: 'the mean field of the coordinates and the weights has no closed-form expected log of the sum over features per row: compile the features as scalar variables (packScalars off) for the variational rules of apply and product',
  },
  traits: { readsTarget: true },
})
