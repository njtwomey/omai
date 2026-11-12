/**
 * Non-conjugate variational message passing (Knowles and Minka 2011) for the factors that have no closed-form expected log:
 * a numeric likelihood of a Gaussian predictor, a density over two random scalars, a softmax over class scores, a
 * deterministic function that is not affine. The mean field of such a factor still has a fixed point (a stationary point of the
 * variational bound), and its message to a port is the natural gradient of `E_q[log f]` with respect to that port's belief:
 * `F^-1 Cov_q(T, log f)`, with `T` the sufficient statistics of the port's family and `F = Cov_q(T, T)` their Fisher information.
 * Both covariances are found by quadrature under the beliefs of the ports (Gauss-Hermite for a Gaussian, a grid in the log
 * for a gamma, a grid in the logit for a beta), so a factor needs only its log density. The expected log itself, the bound's
 * term, is the same quadrature. Where the log density is log-concave in the port the message has a positive precision; where it
 * is not (a heavy-tailed likelihood in its tails) the message can have a negative one, which the belief tolerates while the
 * other factors keep it proper.
 */

import { OmaiError } from '../../../base/index.ts'
import { gaussianMessage, gaussianOf, scalarOf } from '../../core/args.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gammaShapeRate } from '../../core/families.ts'
import type { Arg, Eta } from '../../core/types.ts'
import { shiftedSum } from '../../numerics/_moments.ts'
import { gaussianGridOn, gaussianNodes } from '../../numerics/_quadrature.ts'
import { betaModeWindow, gammaModeWindow, uniformNodes } from '../../numerics/_windows.ts'

/** The quadrature of one port under its belief: the nodes (the coordinate the log density reads), their weights and the statistics. */
export interface QuadPort {
  /** The value each node gives the log density. */
  readonly coords: Float64Array
  /** The weights of the nodes, summing to one. */
  readonly weights: Float64Array
  /** The sufficient statistics of the port's family at each node (empty for an observed or constant port). */
  readonly stats: readonly (readonly [number, number])[]
}

/** What {@link quadraturePort} is told. */
export interface QuadOptions {
  /** The number of nodes (Gaussian) or grid points (gamma, beta). */
  readonly points?: number
  /** For a gamma port: the log density reads the log of the variable, not the variable (a positive coordinate carried as its log). */
  readonly log?: boolean
  /** For a Gaussian port: where the log density has a kink (the outcome of an absolute-value likelihood); the panels are split there, so the kink is resolved. */
  readonly kinks?: readonly number[]
}

const log1pExp = (x: number): number => (x > 30 ? x : x < -30 ? Math.exp(x) : Math.log1p(Math.exp(x)))

/** A uniform grid of `points` on `[lo, hi]` with weights proportional to `exp(logDensity)`, normalised to sum to one. */
function gridWeights(
  window: { readonly lo: number; readonly hi: number },
  points: number,
  logDensity: (t: number) => number,
): { t: Float64Array; w: Float64Array } {
  const t = uniformNodes(window, points)
  const { weights: w, total } = shiftedSum(Float64Array.from(t, logDensity))
  for (let i = 0; i < points; i++) w[i] = (w[i] as number) / total
  return { t, w }
}

/**
 * The quadrature of a port under its belief: an observed or constant value is one node; a Gaussian belief is Gauss-Hermite;
 * a gamma belief is a grid in `t = log x` with the density of `t` (`x` is `e^t`, and the log density reads `x`, or `t` with
 * `log`); a beta belief is a grid in the logit with the density of the logit.
 *
 * @param a - The argument: under variational message passing, the belief of the port or its value.
 * @param options - The number of points and the coordinate.
 * @returns The nodes, weights and sufficient statistics.
 * @throws ImproperMessageError when the belief has no density.
 * @throws OmaiError with code `value` for a family that is not Gaussian, gamma or beta.
 * @concept variational-message-passing
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, quadraturePort } from 'omai-engine/message-passing'
 * const port = quadraturePort(messageArg(gaussian, gaussianEta(1, 4)))
 * const mean = port.coords.reduce((s, x, i) => s + x * (port.weights[i] ?? 0), 0)
 * mean.toFixed(6) // => '1.000000'
 * ```
 */
export function quadraturePort(a: Arg, options: QuadOptions = {}): QuadPort {
  if (a.value !== undefined) return { coords: Float64Array.of(scalarOf(a)), weights: Float64Array.of(1), stats: [] }
  const kind = a.family.kind
  if (kind === 'gaussian') {
    const { mean, variance } = gaussianOf(a)
    const stats = (x: Float64Array): [number, number][] => Array.from(x, (v): [number, number] => [v, -0.5 * v * v])
    const sd = Math.sqrt(variance)
    const inside = (options.kinks ?? []).filter((k) => k > mean - 10 * sd && k < mean + 10 * sd).sort((p, q) => p - q)
    if (inside.length > 0) {
      // Composite Gauss-Legendre on [mean - 10 sd, mean + 10 sd] cut at the kinks, with the density in the weights.
      const lo = mean - 10 * sd
      const hi = mean + 10 * sd
      const cuts = [lo, ...inside, hi]
      const xs: number[] = []
      const ws: number[] = []
      for (let k = 0; k + 1 < cuts.length; k++) {
        const left = cuts[k] as number
        const right = cuts[k + 1] as number
        const panels = Math.max(6, Math.round((40 * (right - left)) / (hi - lo)))
        const grid = gaussianGridOn(mean, variance, left, right, panels, 8)
        xs.push(...grid.x)
        ws.push(...grid.w)
      }
      const total = ws.reduce((p, q) => p + q, 0)
      const x = Float64Array.from(xs)
      return { coords: x, weights: Float64Array.from(ws, (v) => v / total), stats: stats(x) }
    }
    const { x, w } = gaussianNodes(mean, variance, options.points ?? 24)
    return { coords: x, weights: w, stats: stats(x) }
  }
  const eta = a.eta as Eta
  if (kind === 'gamma') {
    const { shape, rate } = gammaShapeRate(eta)
    if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
      throw new ImproperMessageError('a gamma belief has no density (its shape and rate are not positive)')
    const { t, w } = gridWeights(
      gammaModeWindow(shape, rate),
      options.points ?? 161,
      (u) => shape * u - rate * Math.exp(u),
    )
    return {
      coords: options.log === true ? t : Float64Array.from(t, Math.exp),
      weights: w,
      stats: Array.from(t, (u): [number, number] => [u, Math.exp(u)]),
    }
  }
  if (kind === 'beta') {
    const p = (eta[0] ?? Number.NaN) + 1
    const q = (eta[1] ?? Number.NaN) + 1
    if (!(p > 0 && q > 0 && Number.isFinite(p) && Number.isFinite(q)))
      throw new ImproperMessageError('a beta belief has no density (its shapes are not positive)')
    // In u = logit(x) the density is x^p (1 - x)^q.
    const logOf = (u: number): number => -p * log1pExp(-u) - q * log1pExp(u)
    const { t, w } = gridWeights(betaModeWindow(p, q), options.points ?? 161, logOf)
    const probs = Float64Array.from(t, (u) => 1 / (1 + Math.exp(-u)))
    return {
      coords: probs,
      weights: w,
      stats: Array.from(t, (u): [number, number] => [-log1pExp(-u), -log1pExp(u)]),
    }
  }
  throw new ParameterError(`non-conjugate variational message passing has no quadrature for a ${kind} variable`)
}

/**
 * The quadratures of all the ports of a factor, with a number of nodes that keeps the product grid at some thousands of
 * evaluations whatever the number of free ports.
 *
 * @param args - The arguments of the ports.
 * @param options - For each port, whether a gamma belief's coordinate is its log (default: the variable itself), and the kinks of a
 * Gaussian port's log density.
 * @returns One quadrature per port.
 * @concept variational-message-passing
 * @example
 * ```ts
 * import { gamma, gammaEta, messageArg, quadraturePorts, valueArg, gaussian } from 'omai-engine/message-passing'
 * const ports = quadraturePorts([valueArg(gaussian, 2), messageArg(gamma, gammaEta(3, 2))])
 * [ports[0]?.coords.length, ports[1]?.stats.length] // => [1, 161]
 * ```
 */
export function quadraturePorts(
  args: readonly Arg[],
  options: {
    readonly log?: readonly boolean[]
    readonly kinks?: readonly (readonly number[] | undefined)[]
  } = {},
): QuadPort[] {
  const free = args.filter((a) => a.value === undefined).length
  const gauss = free <= 1 ? 48 : free === 2 ? 24 : free === 3 ? 12 : free === 4 ? 8 : 6
  const grid = free <= 1 ? 161 : free === 2 ? 81 : 31
  return args.map((a, i) =>
    quadraturePort(a, {
      points: a.family.kind === 'gaussian' ? gauss : grid,
      log: options.log?.[i] === true,
      ...(options.kinks?.[i] === undefined ? {} : { kinks: options.kinks[i] as readonly number[] }),
    }),
  )
}

/** The average of `logf` over the product of the ports' quadratures, the coordinate of `fixed` set to a node. */
function average(
  ports: readonly QuadPort[],
  skip: number,
  x: number[],
  logf: (x: readonly number[]) => number,
): number {
  const others: number[] = []
  for (let i = 0; i < ports.length; i++) if (i !== skip) others.push(i)
  const counter = new Array<number>(others.length).fill(0)
  let total = 0
  for (;;) {
    let weight = 1
    for (const [k, i] of others.entries()) {
      const p = ports[i] as QuadPort
      x[i] = p.coords[counter[k] as number] as number
      weight *= p.weights[counter[k] as number] as number
    }
    const v = logf(x)
    if (!Number.isFinite(v)) {
      if (weight > 1e-14) throw new ImproperMessageError('the log density is not finite on the support of the beliefs')
    } else total += weight * v
    let k = 0
    for (; k < others.length; k++) {
      const p = ports[others[k] as number] as QuadPort
      counter[k] = (counter[k] as number) + 1
      if ((counter[k] as number) < p.coords.length) break
      counter[k] = 0
    }
    if (k === others.length) break
  }
  return total
}

/**
 * The expected log density under the beliefs of the ports, `E_q[log f]`: the factor's term of the variational bound.
 *
 * @param ports - The quadratures of the ports.
 * @param logf - The log density, one coordinate per port.
 * @returns The expectation.
 * @concept variational-message-passing
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, numericExpectedLog, quadraturePort } from 'omai-engine/message-passing'
 * // E[-x^2 / 2] under N(0, 3) is -3 / 2.
 * const port = quadraturePort(messageArg(gaussian, gaussianEta(0, 3)))
 * numericExpectedLog([port], ([x]) => -0.5 * (x as number) ** 2).toFixed(6) // => '-1.500000'
 * ```
 */
export function numericExpectedLog(ports: readonly QuadPort[], logf: (x: readonly number[]) => number): number {
  return average(ports, -1, new Array<number>(ports.length).fill(0), logf)
}

/**
 * The non-conjugate variational message to a port: the natural gradient of `E_q[log f]` in the port's own belief, `F^-1
 * Cov(T, g)` with `g(x_to) = E_{q of the others}[log f]` evaluated at each node of the port. For a Gaussian port it is the message
 * `[h, J]` with `J = -E[g'']` and `h = E[g'] + J mean`, without a derivative of `log f`.
 *
 * @param ports - The quadratures of the ports.
 * @param to - The port to send to (a free port of the Gaussian, gamma or beta family).
 * @param logf - The log density, one coordinate per port.
 * @returns The message in the natural parameters of the port's family.
 * @throws ImproperMessageError when the density is not finite under the beliefs or the port's statistics are degenerate.
 * @concept variational-message-passing
 * @index function The non-conjugate variational message to a port by quadrature: natural gradient of the expected log.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, numericMessage, quadraturePort } from 'omai-engine/message-passing'
 * // log f = -(x - 2)^2 is a Gaussian likelihood of mean 2 and precision 2: the message is [4, 2].
 * const port = quadraturePort(messageArg(gaussian, gaussianEta(0, 1)))
 * numericMessage([port], 0, ([x]) => -(((x as number) - 2) ** 2)).map((v) => v.toFixed(6)) // => ['4.000000', '2.000000']
 * ```
 */
export function numericMessage(ports: readonly QuadPort[], to: number, logf: (x: readonly number[]) => number): Eta {
  const target = ports[to] as QuadPort
  if (target.stats.length === 0) throw new ParameterError('a variational message is not sent to an observed port')
  const n = target.coords.length
  const x = new Array<number>(ports.length).fill(0)
  const g = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    x[to] = target.coords[i] as number
    g[i] = average(ports, to, x, logf)
  }
  let g0 = 0
  let t0 = 0
  let t1 = 0
  for (let i = 0; i < n; i++) {
    const w = target.weights[i] as number
    g0 += w * (g[i] as number)
    t0 += w * ((target.stats[i] as readonly [number, number])[0] as number)
    t1 += w * ((target.stats[i] as readonly [number, number])[1] as number)
  }
  let c00 = 0
  let c01 = 0
  let c11 = 0
  let s0 = 0
  let s1 = 0
  for (let i = 0; i < n; i++) {
    const w = target.weights[i] as number
    const [a, b] = target.stats[i] as readonly [number, number]
    const da = a - t0
    const db = b - t1
    const dg = (g[i] as number) - g0
    c00 += w * da * da
    c01 += w * da * db
    c11 += w * db * db
    s0 += w * da * dg
    s1 += w * db * dg
  }
  const det = c00 * c11 - c01 * c01
  if (!(det > 1e-300 * (c00 * c11 + 1e-300)) || !Number.isFinite(det))
    throw new ImproperMessageError('the statistics of the belief are degenerate')
  return [(c11 * s0 - c01 * s1) / det, (c00 * s1 - c01 * s0) / det]
}

/**
 * The variational messages of a deterministic factor that defines its output as a function of Gaussian inputs, `y = f(x)`
 * (ports `y` then the inputs): to the output the Gaussian that has the mean and variance of `f(x)` under the beliefs of the
 * inputs (the pushforward, which is what its readers' expected log needs when they are Gaussian); to an input the
 * non-conjugate message of `E[h f(x) - J f(x)^2 / 2]`, `(h, J)` the cavity of the output (the messages its readers send it).
 *
 * @param args - The arguments (the output first, then the inputs).
 * @param to - The port to send to.
 * @param f - The function of the inputs.
 * @returns The message.
 * @throws OmaiError with code `not-implemented` for an observed output.
 * @concept variational-message-passing
 * @example
 * ```ts
 * import { definedFunctionMessage, gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * // y = x with x ~ N(1, 1/4): the output is told N(1, 1/4).
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(1, 0.25))]
 * definedFunctionMessage(args, 0, ([x]) => x as number).map((v) => v.toFixed(4)) // => ['4.0000', '4.0000']
 * ```
 */
export function definedFunctionMessage(args: readonly Arg[], to: number, f: (x: readonly number[]) => number): Eta {
  const inputs = args.slice(1)
  const ports = quadraturePorts(inputs)
  if (to === 0) {
    const coords = new Array<number>(inputs.length).fill(0)
    const mean = average(ports, -1, coords, f)
    const second = average(ports, -1, coords, (x) => f(x) ** 2)
    return gaussianMessage(mean, Math.max(second - mean * mean, 1e-300))
  }
  const out = (args[0] as Arg).eta
  if (out === undefined)
    throw new OmaiError('not-implemented', 'a deterministic factor with an observed output has no variational message')
  const h = out[0] ?? 0
  const lambda = out[1] ?? 0
  return numericMessage(ports, to - 1, (x) => {
    const v = f(x)
    return h * v - 0.5 * lambda * v * v
  })
}
