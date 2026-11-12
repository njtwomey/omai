/**
 * The computation of 'How expectation propagation works': Minka's clutter problem, the example expectation propagation (EP) was
 * first explained with. One unknown location has a wide prior, and six measurements each come from the location (a bell curve of
 * width one around it) or, with probability 0.4, are clutter (a much wider bell curve around zero). The library's own EP, in
 * `omai-engine/message-passing`, runs on the factor graph of that model one factor at a time, and the run is read three ways:
 * every update of the first sweep, the whole of one update taken apart (the cavity, the tilted curve, the bell that matches it and
 * the new stand-in, worked out here with tensor operations and checked against the engine's own message), and the beliefs and
 * stand-ins after every sweep. The exact posterior (a grid) is beside it. The result is plain data; the seed decides the six values.
 * It runs in the story's worker, so the engine is bundled there and not in the page. The regions are the code shown.
 */

// region imports
import { normal } from 'omai-engine/dists'
import { clutterGraph, clutterPosterior, clutterProblem, numericEpMethod } from 'omai-engine/message-passing'
import { key } from 'omai-engine/random'
import { add, div, item, linspace, log, mul, square, sub, sum, take, tensor, toFlat } from 'omai-engine/tensor'
import { metricSeries, session } from 'omai-engine/trace'
// endregion

import { type ModelRun, modelRunFromTape } from 'omai-engine/exposition'
import { type EpPath, graphStructure, type MpState } from 'omai-engine/message-passing'
import type { Config, Session } from 'omai-engine/trace'
import type { Say } from '../../kit/protocol.ts'
import { replayRecord } from '../../kit/replay.ts'
import { CLUTTER_CHANCE, CLUTTER_SD, MAX_SWEEPS, ROWS, SHOWN_SWEEP, TOLERANCE } from './data.ts'

/** A bell curve: its centre and its standard deviation, or `null` where the message has no bell (a flat or negative precision). */
export interface Bell {
  readonly mean: number
  readonly sd: number
}

/** A message as the engine keeps it: the precision, and the precision times the mean (the two natural parameters). */
export interface Message {
  readonly precision: number
  readonly shift: number
}

/** One update of the run: a factor, the bell it was given (its cavity), the stand-in it sent and the approximation afterwards. */
export interface Step {
  /** 0 for the prior, 1 to `rows` for the measurements. */
  readonly factor: number
  readonly cavity: Bell | null
  readonly standIn: Message
  readonly belief: Bell | null
}

/** One update taken apart. All the curves are on one grid of values of the location. */
export interface Taken {
  readonly sweep: number
  /** The factor (1 to `rows`) and the measurement it holds. */
  readonly factor: number
  readonly point: number
  /** The approximation just before the update, and this factor's own stand-in then. */
  readonly before: Message
  readonly own: Message
  readonly cavity: Message
  readonly tilted: Bell
  readonly matched: Message
  readonly standIn: Message
  /** The stand-in the engine's own update sent, to compare with `standIn`: the largest difference of the two numbers. */
  readonly engineGap: number
  /** How far the matched bell is from the tilted curve: the divergence from the tilted curve to the bell, in nats. */
  readonly divergence: number
  readonly grid: readonly number[]
  readonly cavityDensity: readonly number[]
  readonly factorValues: readonly number[]
  readonly tiltedDensity: readonly number[]
  readonly matchedDensity: readonly number[]
}

/** What the figures and the prose read. All of it is plain data. */
export interface Result {
  readonly rows: number
  /** The `rows` values, which of them the generator made clutter, and the location it drew the others around. */
  readonly points: readonly number[]
  readonly clutter: readonly boolean[]
  readonly truth: number
  /** The first sweep, update by update. */
  readonly first: readonly Step[]
  /** The sweeps the run made, whether it settled, and the tolerance it was given. */
  readonly sweeps: number
  readonly settled: boolean
  /** The belief after every sweep (sweep 1 is the first complete one), and the largest change of a belief over each. */
  readonly beliefs: {
    readonly sweep: readonly number[]
    readonly mean: readonly number[]
    readonly sd: readonly number[]
    readonly change: readonly number[]
  }
  /** The stand-ins of every factor after a few sweeps. */
  readonly standIns: readonly { readonly sweep: number; readonly messages: readonly Message[] }[]
  /** One update, taken apart (undefined when no factor had a proper cavity in the sweep). */
  readonly taken: Taken | undefined
  /** The whole run as a record the page can replay: the factor graph with the message of each update on its edge. */
  readonly record: ModelRun
  /** The answer EP ended with, its estimate of the log evidence, and the exact posterior on a grid. */
  readonly ep: Bell
  readonly logEvidence: number | undefined
  readonly exact: {
    readonly mean: number
    readonly sd: number
    readonly logEvidence: number
    readonly grid: readonly number[]
    readonly density: readonly number[]
    readonly epDensity: readonly number[]
  }
}

/** How much to use: the run uses `ROWS` measurements. */
export interface Size {
  readonly rows: number
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { rows: ROWS }

/** The bell of a message, or null when the message has no bell. */
const bellOf = (m: readonly number[]): Bell | null =>
  (m[1] ?? 0) > 0 ? { mean: (m[0] ?? 0) / (m[1] as number), sd: 1 / Math.sqrt(m[1] as number) } : null

const messageOf = (m: readonly number[]): Message => ({ shift: m[0] ?? 0, precision: m[1] ?? 0 })

/** A session of the library's EP on the clutter graph. */
type EpSession = Session<MpState, Config, EpPath>

/**
 * The run of the story: make the problem from the seed, run EP on it factor by factor, take one update apart and read the beliefs
 * and stand-ins of every sweep.
 *
 * @param seed - Decides the measurements.
 * @param say - Says what is happening, for the play button.
 * @param size - How many measurements.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Making six measurements, some of them clutter')
  // region model
  // the model as a factor graph: one unknown, a prior factor and one clutter factor for each measurement
  const problem = clutterProblem(key(seed), { n: size.rows, weight: CLUTTER_CHANCE })
  const graph = clutterGraph(problem)
  // endregion
  say('Passing messages, one factor at a time')
  // region infer
  // numeric EP: the clutter factors have no closed form, so the engine integrates them
  const method = numericEpMethod(graph)
  // one step is one factor's update, and a sweep is every factor once
  const run = session(method, {
    steps: MAX_SWEEPS * graph.factors.length + 2,
    mode: 'eager',
    logs: 'all',
    config: { granularity: 'factor', tol: TOLERANCE, restarts: 0, maxSweeps: MAX_SWEEPS },
  })
  // endregion
  const factors = graph.factors.length
  const sweeps = Math.floor(run.t / factors)
  say('Reading the first sweep and one update in detail')
  const messagesAt = (step: number): readonly (readonly number[])[] => run.stateAt(step).messages
  const beliefAt = (step: number): readonly number[] => {
    const ms = messagesAt(step)
    return [ms.reduce((a, m) => a + (m[0] ?? 0), 0), ms.reduce((a, m) => a + (m[1] ?? 0), 0)]
  }
  const first: Step[] = Array.from({ length: factors }, (_, j) => {
    const ms = messagesAt(j)
    const cavity = beliefAt(j).map((v, k) => v - (ms[j]?.[k] ?? 0))
    return {
      factor: j,
      cavity: bellOf(cavity),
      standIn: messageOf(messagesAt(j + 1)[j] as readonly number[]),
      belief: bellOf(beliefAt(j + 1)),
    }
  })
  // the update to take apart: the measurement in the shown sweep whose tilted curve is furthest from its bell
  const candidates =
    sweeps < SHOWN_SWEEP ? [] : problem.ys.map((_, k) => takeApart(run, problem.ys, SHOWN_SWEEP, k + 1))
  const taken = candidates.reduce<Taken | undefined>(
    (best, t) => (t !== undefined && (best === undefined || t.divergence > best.divergence) ? t : best),
    undefined,
  )
  say('Reading the beliefs sweep by sweep')
  const sweepSteps = Array.from({ length: sweeps }, (_, s) => s + 1)
  const changes = metricSeries(run.run(), 'mp/change')
  const changeAt = (s: number): number => {
    const at = changes.steps.indexOf(s * factors)
    const v = at < 0 ? Number.NaN : (changes.values[at] as number)
    return Number.isFinite(v) ? v : Number.NaN
  }
  const bells = sweepSteps.map((s) => bellOf(beliefAt(s * factors)))
  const shown = [...new Set([1, 2, 3, sweeps].filter((s) => s >= 1 && s <= sweeps))].sort((a, b) => a - b)
  const exact = clutterPosterior(problem)
  const final = bellOf(beliefAt(run.t)) ?? { mean: Number.NaN, sd: Number.NaN }
  // the exact posterior on a grid of about 400 values, over the part that holds its mass
  const peak = Math.max(...exact.density)
  const inside = exact.density.map((d, i) => (d > 1e-3 * peak ? i : -1)).filter((i) => i >= 0)
  const lo = Math.max(0, (inside[0] ?? 0) - 40)
  const hi = Math.min(exact.x.length - 1, (inside.at(-1) ?? 0) + 40)
  const every = Math.max(1, Math.round((hi - lo) / 400))
  const picks = Array.from({ length: Math.floor((hi - lo) / every) + 1 }, (_, k) => lo + k * every)
  const grid = picks.map((i) => exact.x[i] as number)
  const evidence = metricSeries(run.run(), 'mp/evidence').values.at(-1)
  return {
    rows: size.rows,
    points: problem.ys,
    clutter: problem.isClutter,
    truth: problem.truth,
    first,
    sweeps,
    settled: run.status === 'converged',
    beliefs: {
      sweep: sweepSteps,
      mean: bells.map((b) => b?.mean ?? Number.NaN),
      sd: bells.map((b) => b?.sd ?? Number.NaN),
      change: sweepSteps.map(changeAt),
    },
    standIns: shown.map((sweep) => ({ sweep, messages: messagesAt(sweep * factors).map(messageOf) })),
    taken,
    record: replayRecord(modelRunFromTape(graphStructure(graph.spec), { run: run.run() }, { maxMessages: 6 })),
    ep: final,
    logEvidence: evidence,
    exact: {
      mean: exact.mean,
      sd: exact.sd,
      logEvidence: exact.logEvidence,
      grid,
      density: picks.map((i) => exact.density[i] as number),
      epDensity: toFlat(normal(final.mean, final.sd).prob(grid)),
    },
  }
}

/**
 * The run of the story.
 *
 * @param seed - Decides the measurements.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)

/**
 * One update of EP, worked out step by step from the stand-ins before it: (a) the cavity, (b) the tilted curve on a grid, (c) the
 * bell with its mean and variance, (d) the new stand-in. The engine's own message for the same update is read for comparison.
 *
 * @param run - The session.
 * @param points - The measurements.
 * @param sweep - The sweep (1 is the first).
 * @param factor - The factor: 1 to rows for a measurement.
 * @returns The update taken apart, or undefined when the cavity has no bell.
 */
function takeApart(run: EpSession, points: readonly number[], sweep: number, factor: number): Taken | undefined {
  const factors = points.length + 1
  const before = (sweep - 1) * factors + factor
  const point = points[factor - 1] as number
  // region update
  // every factor's stand-in before this update, one row each: (precision times mean, precision)
  const standIns = tensor(run.stateAt(before).messages)
  // the approximation is the product of the stand-ins: in natural parameters, their sum
  const belief = sum(standIns, 0)
  const own = take(standIns, factor, 0)
  // (a) the cavity: the approximation without this factor's own stand-in
  const cavity = sub(belief, own)
  const [shift, precision] = toFlat(cavity) as [number, number]
  // a cavity with no positive precision is not a bell: this update cannot be worked out
  if (!(precision > 0)) return undefined
  const cavityMean = shift / precision
  const cavitySd = 1 / Math.sqrt(precision)
  // (b) the tilted curve: the cavity times the real factor, on a grid of locations
  const grid = linspace(cavityMean - 6 * cavitySd, cavityMean + 6 * cavitySd, 401)
  const spacing = (12 * cavitySd) / 400
  const cavityDensity = normal(cavityMean, cavitySd).prob(grid)
  // the real factor: the point is a measurement of the location, or it is clutter
  const asMeasurement = mul(1 - CLUTTER_CHANCE, normal(grid, 1).prob(point))
  const asClutter = mul(CLUTTER_CHANCE, normal(0, CLUTTER_SD).prob(point))
  const factorValues = add(asMeasurement, asClutter)
  const tilted = mul(cavityDensity, factorValues)
  const density = div(tilted, mul(sum(tilted), spacing))
  // (c) the bell with the same mean and variance as the tilted curve
  const tiltedMean = item(mul(sum(mul(grid, density)), spacing))
  const tiltedVariance = item(mul(sum(mul(square(sub(grid, tiltedMean)), density)), spacing))
  const matched = tensor([tiltedMean / tiltedVariance, 1 / tiltedVariance])
  // (d) the new stand-in: the matched bell divided by the cavity
  const standIn = sub(matched, cavity)
  // endregion
  const bell = normal(tiltedMean, Math.sqrt(tiltedVariance)).prob(grid)
  const floor = 1e-300
  const divergence = item(mul(sum(mul(density, sub(log(add(density, floor)), log(add(bell, floor))))), spacing))
  const engine = run.stateAt(before + 1).messages[factor] as readonly number[]
  const sent = toFlat(standIn)
  return {
    sweep,
    factor,
    point,
    before: messageOf(toFlat(belief)),
    own: messageOf(toFlat(own)),
    cavity: messageOf(toFlat(cavity)),
    tilted: { mean: tiltedMean, sd: Math.sqrt(tiltedVariance) },
    matched: messageOf(toFlat(matched)),
    standIn: messageOf(sent),
    engineGap: Math.max(Math.abs((engine[0] ?? 0) - (sent[0] ?? 0)), Math.abs((engine[1] ?? 0) - (sent[1] ?? 0))),
    divergence,
    grid: toFlat(grid),
    cavityDensity: toFlat(cavityDensity),
    factorValues: toFlat(factorValues),
    tiltedDensity: toFlat(density),
    matchedDensity: toFlat(bell),
  }
}
