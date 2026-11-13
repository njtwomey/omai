/**
 * The computation of 'How long until it fails': forty units on test, some of which failed and some of which were still working when
 * their follow-up ended (they are censored). The lifetime of a unit is Weibull, with an unknown scale and an unknown shape, each on
 * a log scale so that a bell curve can describe it. Every unit is one factor of a factor graph that connects the two unknowns: a
 * unit that failed at a time contributes the Weibull density there, and a unit that was censored at a time contributes the
 * survival function there (the chance of lasting at least that long). The library's expectation propagation (EP) passes messages
 * between the factors and the unknowns until they settle. The same graph is run again without the censored units, to show what
 * dropping them does. One update of the first sweep is taken apart for a unit taken as a failure and as a censoring at the same
 * time, from the same cavity. The exact posterior of the two unknowns, which tensor operations work out on a grid, is read for
 * comparison. The survival curve and its band come from draws of the two unknowns. The result is plain data and a pure function of
 * the seed (which decides the draws), so it runs in the story's worker. The regions are the code shown.
 */

// region imports
import { normal as bell } from 'omai-engine/dists'
import { compileGraph, defaultOperators, epMethod, gaussianPrior, marginalsOf } from 'omai-engine/message-passing'
import { foldIn, key, normal } from 'omai-engine/random'
import {
  add,
  div,
  exp,
  item,
  linspace,
  log,
  mul,
  neg,
  pow,
  reshape,
  sliceAxis,
  softmax,
  sort,
  square,
  sub,
  sum,
  take,
  tensor,
  toFlat,
  type Value,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import type { CompiledGraph, EpPath, FactorGraphSpec, MpState } from 'omai-engine/message-passing'
import type { Config, Session } from 'omai-engine/trace'
import type { Say } from '../../kit/protocol.ts'
import { HORIZON, lifetimes } from './data.ts'

/** A bell curve: its mean and its standard deviation. */
export interface Bell {
  readonly mean: number
  readonly sd: number
}

/** What EP concludes about the two unknowns, the log of the scale and the log of the shape. */
export interface Fit {
  readonly scale: Bell
  readonly shape: Bell
  /** The sweeps of messages it took, and whether they settled. */
  readonly sweeps: number
  readonly settled: boolean
}

/** One kind of observation at one time, as what it does to a cavity. The curves are over the scale, in units of time. */
export interface Case {
  readonly kind: 'failed' | 'censored'
  /** The scales the curves are drawn at, the cavity, the cavity times the unit's factor (the tilted curve) and its bell curve. */
  readonly scale: readonly number[]
  readonly cavity: readonly number[]
  readonly tilted: readonly number[]
  readonly matched: readonly number[]
  /** The log of the scale before (the cavity) and after (the bell curve that matches the tilted curve). */
  readonly before: Bell
  readonly after: Bell
  /** What the new message adds to the precision of each unknown (the precision is one over the variance). */
  readonly precisionScale: number
  readonly precisionShape: number
  /** For the kind the data say: the largest difference between this message and the one the engine sent. */
  readonly engineGap: number | undefined
}

/** One update of the first sweep, with the unit's time taken as a failure and as a censoring. */
export interface Update {
  /** The unit (counting from 1 in the order the messages visit them), its time, and what the data say it was. */
  readonly unit: number
  readonly time: number
  readonly actual: 'failed' | 'censored'
  readonly failed: Case
  readonly censored: Case
  /** The range of scales the figure draws. */
  readonly range: readonly [number, number]
}

/** The curves over the same times: the true one, the fit with every unit (its middle and a band) and the fit without the censored ones. */
export interface Curves {
  readonly time: readonly number[]
  readonly truth: readonly number[]
  readonly middle: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  readonly dropped: readonly number[]
  /** The share of the times at which the true curve lies inside the band. */
  readonly covered: number
}

/** What the figures and the prose read. All of it is plain data. */
export interface Result {
  readonly time: readonly number[]
  readonly event: readonly number[]
  /** The fit with every unit, and the fit with the censored units left out. */
  readonly all: Fit
  readonly dropped: Fit
  /** The spread of the log of the scale and of the shape in the first sweep, after this many units. */
  readonly sharpening: {
    readonly units: readonly number[]
    readonly scale: readonly number[]
    readonly shape: readonly number[]
  }
  readonly update: Update
  /** The exact posterior of the two unknowns, from a grid: a bell curve for each, and their correlation. */
  readonly exact: { readonly scale: Bell; readonly shape: Bell; readonly correlation: number }
  readonly curves: Curves
}

/** How much to do. */
export interface Size {
  readonly units: number
  readonly draws: number
  readonly times: number
  readonly grid: number
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { units: 40, draws: 2000, times: 141, grid: 161 }

/** The sweeps the messages are allowed. */
const MAX_SWEEPS = 100

/** The unit whose update is taken apart (counting from 0): the first sweep, after this many units have sent their first messages. */
const UNIT_SHOWN = 9

/** The units after which the spread of each unknown is read, in the first sweep. */
const SHARPENING_UNITS = [0, 5, 10, 20, 40] as const

/** The priors: the log of the scale has a mean of 1 and a variance of 4, the log of the shape a mean of 0 and a variance of 1. */
const PRIOR = { scale: { mean: 1, variance: 4 }, shape: { mean: 0, variance: 1 } } as const

// region factor
/**
 * The log of a unit's factor. With scale s = exp(u) and shape k = exp(t), the chance of
 * lasting past time y is exp(-(y / s)^k): the survival. A unit that failed at y also says
 * that it failed then and not later. That multiplies the survival by the hazard, the rate
 * of failing at y among the units that have lasted until then.
 */
const logFactor = (y: Value | number, censored: Value | number, u: Value, t: Value): Value => {
  const k = exp(t)
  const logSurvival = neg(exp(mul(k, sub(log(y), u))))
  const logHazard = add(sub(t, u), mul(sub(k, 1), sub(log(y), u)))
  return add(logSurvival, mul(sub(1, censored), logHazard))
}
// endregion

/** A session of EP on the graph of the units, and the graph. */
interface Solved {
  readonly fit: Fit
  readonly graph: CompiledGraph
  readonly run: Session<MpState, Config, EpPath>
}

/**
 * Pass messages on the graph of some units until they settle.
 *
 * @param time - When each unit was last seen.
 * @param event - Whether it failed then (1) or was still working (0).
 * @returns The bell curves for the two unknowns, and the session to read the messages from.
 */
function solve(time: readonly number[], event: readonly number[]): Solved {
  // region graph
  // the unknowns are the log of the scale (u) and the log of the shape (t)
  // each has a bell curve for a prior
  const scalePrior = gaussianPrior('u', PRIOR.scale.mean, PRIOR.scale.variance)
  const shapePrior = gaussianPrior('t', PRIOR.shape.mean, PRIOR.shape.variance)
  // each unit is one factor on both unknowns: a failure time is a density, a censoring time a survival
  const units = time.map((y, i) => ({
    operator: 'likelihood-pair',
    args: ['u', 't'],
    params: { likelihood: 'weibull-shape', y, censored: 1 - (event[i] ?? 0) },
  }))
  const spec: FactorGraphSpec = {
    variables: [
      { name: 'u', family: { kind: 'gaussian' } },
      { name: 't', family: { kind: 'gaussian' } },
    ],
    factors: [scalePrior, shapePrior, ...units],
  }
  const graph = compileGraph(spec, defaultOperators())
  // endregion
  // region infer
  // expectation propagation: one step is one factor's update, and a sweep is every factor once
  const run = session(epMethod(graph), {
    steps: MAX_SWEEPS * graph.factors.length + 2,
    mode: 'eager',
    logs: 'off',
    config: { granularity: 'factor', tol: 1e-6, restarts: 0, maxSweeps: MAX_SWEEPS },
  })
  // the bell curve for each unknown: its mean and its variance
  const beliefs = marginalsOf(graph, run.state)
  // endregion
  const belief = (name: string): Bell => ({
    mean: Number(beliefs[name]?.['mean']),
    sd: Math.sqrt(Number(beliefs[name]?.['variance'])),
  })
  return {
    fit: {
      scale: belief('u'),
      shape: belief('t'),
      sweeps: Math.floor(run.t / graph.factors.length),
      settled: run.status === 'converged',
    },
    graph,
    run,
  }
}

/** The messages of a step as a tensor with a row for each edge: (precision times mean, precision). */
const messagesAt = (solved: Solved, step: number): Value => tensor(solved.run.stateAt(step).messages)

/** The edges that end at a variable (0 for the scale, 1 for the shape), and the edge a factor sends a variable its message on. */
const edgesOf = (graph: CompiledGraph, variable: number): number[] =>
  graph.edges.flatMap((e, i) => (e.variable === variable ? [i] : []))
const edgeOf = (graph: CompiledGraph, factor: number, variable: number): number =>
  graph.edges.findIndex((e) => e.factor === factor && e.variable === variable)

/** A bell curve from its natural parameters: (precision times mean, precision). */
const bellOf = ([shift, precision]: readonly number[]): Bell => ({
  mean: (shift as number) / (precision as number),
  sd: 1 / Math.sqrt(precision as number),
})

/** The belief of a variable at a step: the sum of the messages it receives. */
const beliefAt = (solved: Solved, step: number, variable: number): Bell =>
  bellOf(toFlat(sum(take(messagesAt(solved, step), edgesOf(solved.graph, variable), 0), 0)))

/**
 * One update of the first sweep, worked out from the messages before it, for a unit taken as a failure and as a censoring.
 *
 * @param solved - The run on the graph.
 * @param unit - The unit, counting from 0.
 * @param time - Its time.
 * @param actual - What the data say it was.
 * @param size - How many points the curves have.
 * @returns The two cases and the range of scales the figure draws.
 */
function takeApart(
  solved: Solved,
  unit: number,
  time: number,
  actual: Case['kind'],
  size: Size,
): Pick<Update, 'failed' | 'censored' | 'range'> {
  const factor = 2 + unit
  const before = messagesAt(solved, factor)
  const sent = messagesAt(solved, factor + 1)
  // the cavity of an unknown: its belief before the update, without this unit's own message
  const cavityOf = (variable: number): Bell => {
    const belief = sum(take(before, edgesOf(solved.graph, variable), 0), 0)
    return bellOf(toFlat(sub(belief, take(before, edgeOf(solved.graph, factor, variable), 0))))
  }
  const [cavityU, cavityT] = [cavityOf(0), cavityOf(1)]
  const caseOf = (kind: Case['kind']): Case => {
    // region update
    // a grid over both unknowns: a row for each value of u, a column for each value of t
    const span = (c: Bell): Value => linspace(c.mean - 6 * c.sd, c.mean + 6 * c.sd, size.grid)
    const [rows, columns] = [span(cavityU), span(cavityT)]
    const u = reshape(rows, [size.grid, 1])
    const t = reshape(columns, [1, size.grid])
    // the cavity is a bell curve for each unknown
    const logCavityU = bell(cavityU.mean, cavityU.sd).logProb(u)
    const logCavityT = bell(cavityT.mean, cavityT.sd).logProb(t)
    // the tilted curve is the cavity times the unit's factor, here as weights that sum to one
    const logFactors = logFactor(time, kind === 'failed' ? 0 : 1, u, t)
    const tilted = softmax(reshape(add(add(logCavityU, logCavityT), logFactors), [-1]))
    const weights = reshape(tilted, [size.grid, size.grid])
    // the mean and variance of each unknown under the tilted curve
    const weightU = sum(weights, 1)
    const weightT = sum(weights, 0)
    const meanU = item(sum(mul(weightU, rows)))
    const meanT = item(sum(mul(weightT, columns)))
    const varianceU = item(sum(mul(weightU, square(sub(rows, meanU)))))
    const varianceT = item(sum(mul(weightT, square(sub(columns, meanT)))))
    // the new message is that bell curve divided by the cavity
    // in natural parameters, a difference
    const message = (mean: number, variance: number, cavity: Bell): number[] => [
      mean / variance - cavity.mean / cavity.sd ** 2,
      1 / variance - 1 / cavity.sd ** 2,
    ]
    // endregion
    const toU = message(meanU, varianceU, cavityU)
    const toT = message(meanT, varianceT, cavityT)
    const engineU = toFlat(take(sent, edgeOf(solved.graph, factor, 0), 0))
    const engineT = toFlat(take(sent, edgeOf(solved.graph, factor, 1), 0))
    const gap = Math.max(
      ...toU.map((v, i) => Math.abs(v - (engineU[i] as number))),
      ...toT.map((v, i) => Math.abs(v - (engineT[i] as number))),
    )
    // a density over the log of the scale becomes a density over the scale by dividing by the scale
    const overScale = (density: Value): readonly number[] => toFlat(div(density, exp(rows)))
    const spacing = (12 * cavityU.sd) / (size.grid - 1)
    return {
      kind,
      scale: toFlat(exp(rows)),
      cavity: overScale(exp(reshape(logCavityU, [-1]))),
      tilted: overScale(div(weightU, spacing)),
      matched: overScale(bell(meanU, Math.sqrt(varianceU)).prob(rows)),
      before: cavityU,
      after: { mean: meanU, sd: Math.sqrt(varianceU) },
      precisionScale: toU[1] as number,
      precisionShape: toT[1] as number,
      engineGap: kind === actual ? gap : undefined,
    }
  }
  return {
    failed: caseOf('failed'),
    censored: caseOf('censored'),
    range: [Math.exp(cavityU.mean - 3.5 * cavityU.sd), Math.exp(cavityU.mean + 3.5 * cavityU.sd)],
  }
}

/**
 * The exact posterior of the two unknowns on a grid: the prior and every unit's log factor summed over a grid of the two unknowns,
 * laid around the bell curves EP found.
 *
 * @param time - When each unit was last seen.
 * @param event - Whether it failed then.
 * @param around - The fit the grid is laid around.
 * @param size - The grid.
 * @returns A bell curve for each unknown (the exact mean and spread) and their correlation.
 */
function exactPosterior(time: readonly number[], event: readonly number[], around: Fit, size: Size): Result['exact'] {
  const cells = size.grid + 40
  const span = (c: Bell): Value => linspace(c.mean - 8 * c.sd, c.mean + 8 * c.sd, cells)
  const [rows, columns] = [span(around.scale), span(around.shape)]
  // the units are laid out along a first axis, the grid of the two unknowns along the other two
  const u = reshape(rows, [1, cells, 1])
  const t = reshape(columns, [1, 1, cells])
  const censored = reshape(tensor(event.map((e) => 1 - e)), [-1, 1, 1])
  const logLikelihood = sum(logFactor(reshape(tensor(time), [-1, 1, 1]), censored, u, t), 0)
  const logPrior = add(
    bell(PRIOR.scale.mean, Math.sqrt(PRIOR.scale.variance)).logProb(reshape(rows, [cells, 1])),
    bell(PRIOR.shape.mean, Math.sqrt(PRIOR.shape.variance)).logProb(reshape(columns, [1, cells])),
  )
  const posterior = reshape(softmax(reshape(add(logLikelihood, logPrior), [-1])), [cells, cells])
  // the exact means, spreads and correlation are weighted sums over the cells
  const [weightU, weightT] = [sum(posterior, 1), sum(posterior, 0)]
  const [meanU, meanT] = [item(sum(mul(weightU, rows))), item(sum(mul(weightT, columns)))]
  const [offU, offT] = [sub(rows, meanU), sub(columns, meanT)]
  const [varianceU, varianceT] = [item(sum(mul(weightU, square(offU)))), item(sum(mul(weightT, square(offT))))]
  const covariance = item(sum(mul(posterior, mul(reshape(offU, [cells, 1]), reshape(offT, [1, cells])))))
  return {
    scale: { mean: meanU, sd: Math.sqrt(varianceU) },
    shape: { mean: meanT, sd: Math.sqrt(varianceT) },
    correlation: covariance / Math.sqrt(varianceU * varianceT),
  }
}

/**
 * The survival curves: draws of the two unknowns from their bell curves give one curve each, and the 5% and 95% points of the
 * curves at each time are the edges of the band.
 *
 * @param all - The fit with every unit.
 * @param dropped - The fit without the censored units.
 * @param truth - The true survival curve.
 * @param seed - Decides the draws.
 * @param size - How many draws and times.
 * @returns The curves.
 */
function survivalCurves(all: Fit, dropped: Fit, truth: (time: number) => number, seed: number, size: Size): Curves {
  // region curves
  const times = linspace(0, HORIZON, size.times)
  // draws of the log of the scale and of the log of the shape, one row for each draw
  const logScales = add(all.scale.mean, mul(all.scale.sd, normal(foldIn(key(seed), 0), [size.draws, 1])))
  const logShapes = add(all.shape.mean, mul(all.shape.sd, normal(foldIn(key(seed), 1), [size.draws, 1])))
  // one survival curve for each draw, exp(-(time / scale)^shape): a row for each draw, a column for each time
  const curves = exp(neg(pow(div(times, exp(logScales)), exp(logShapes))))
  // sort each column: its 5% and 95% rows are the edges of a 90% band
  const sorted = sort(curves, 0)
  const low = Math.round(0.05 * (size.draws - 1))
  const high = Math.round(0.95 * (size.draws - 1))
  const lower = sliceAxis(sorted, 0, low, low + 1)
  const upper = sliceAxis(sorted, 0, high, high + 1)
  // the curve at the middle of each belief: with every unit, and without the censored ones
  const atMiddle = (fit: Fit): Value => {
    const [scale, shape] = [Math.exp(fit.scale.mean), Math.exp(fit.shape.mean)]
    return exp(neg(pow(div(times, scale), shape)))
  }
  // endregion
  const lowerCurve = toFlat(lower)
  const upperCurve = toFlat(upper)
  const trueCurve = toFlat(times).map(truth)
  const inside = trueCurve.filter((v, i) => v >= (lowerCurve[i] as number) && v <= (upperCurve[i] as number))
  return {
    time: toFlat(times),
    truth: trueCurve,
    middle: toFlat(atMiddle(all)),
    lower: lowerCurve,
    upper: upperCurve,
    dropped: toFlat(atMiddle(dropped)),
    covered: inside.length / trueCurve.length,
  }
}

/**
 * The run of the story: pass messages on all the units, pass them again without the censored ones, take one update apart, read the
 * exact posterior and draw the survival curves.
 *
 * @param seed - Decides the draws of the band.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  const data = lifetimes(size.units)
  say('Passing messages between the units and the two unknowns')
  const all = solve(data.time, data.event)
  say('Passing messages again without the censored units')
  const kept = data.event.flatMap((e, i) => (e === 1 ? [i] : []))
  const dropped = solve(
    kept.map((i) => data.time[i] as number),
    kept.map(() => 1),
  )
  say('Taking one update apart')
  const unit = Math.min(UNIT_SHOWN, size.units - 1)
  const actual = data.event[unit] === 1 ? 'failed' : 'censored'
  const apart = takeApart(all, unit, data.time[unit] as number, actual, size)
  const sharpening = SHARPENING_UNITS.filter((n) => n <= size.units)
  say('Working out the exact answer on a grid')
  const exact = exactPosterior(data.time, data.event, all.fit, size)
  say('Drawing the survival curves')
  const curves = survivalCurves(all.fit, dropped.fit, data.survival, seed, size)
  return {
    time: data.time,
    event: data.event,
    all: all.fit,
    dropped: dropped.fit,
    sharpening: {
      units: sharpening,
      scale: sharpening.map((n) => beliefAt(all, 2 + n, 0).sd),
      shape: sharpening.map((n) => beliefAt(all, 2 + n, 1).sd),
    },
    update: { unit: unit + 1, time: data.time[unit] as number, actual, ...apart },
    exact,
    curves,
  }
}

/**
 * The run of the story.
 *
 * @param seed - Decides the draws of the band.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
