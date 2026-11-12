/**
 * The computation of 'Gaussian mixtures': a mixture of Gaussian groups is fitted to the 272 eruptions of Old Faithful (the
 * length of each eruption and the wait before it), built the way the reader chose: the shape of each group (round, diagonal or
 * full) and the number of groups (2, 3, or as many as the data need, with stick-breaking weights cut off at 6 or 8 pieces). One
 * model is written for every choice. Where expectation propagation runs it and settles in a few seconds it runs it; the rest are
 * run by the library's Gibbs sampler, and the result says which. The result is plain data: the groups (weight, centre, spread,
 * tilt and an ellipse to draw), the group of each eruption and how sure the model is of it. It is a pure function of the seed
 * and the choice; the seed decides the order in which the eruptions are handed to the method (and the sampler's draws). It runs
 * in the story's worker, so the engine is bundled there and not in the page. The regions are the code shown.
 */

import { epRoute, gibbsMethod } from 'omai-engine/probprog/inference'
import { defineModel, ops, type Var } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, uniform } from 'omai-engine/random'
import {
  add,
  arange,
  argmax,
  argsort,
  broadcastTo,
  cast,
  cos,
  div,
  equal,
  full,
  linspace,
  max,
  mean,
  mul,
  reshape,
  sin,
  sliceAxis,
  sqrt,
  square,
  std,
  sub,
  take,
  toFlat,
  type Value,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { Say } from '../../kit/protocol.ts'
import { columns } from './data.ts'

/** The shape of each group: one spread for both directions, a spread in each direction, or a spread in each direction and a tilt. */
export type Shape = 'round' | 'diagonal' | 'full'

/** How many groups: a fixed number, or as many as the data need (stick-breaking weights cut off at 6 or 8 pieces). */
export type Groups = 'two' | 'three' | 'open6' | 'open8'

/** What the reader chose before pressing play. */
export interface Build {
  readonly shape: Shape
  readonly groups: Groups
}

/** The choice the page opens on. */
export const DEFAULT_BUILD: Build = { shape: 'round', groups: 'two' }

/** How many pieces the model has room for under a choice of groups. */
export const piecesOf = (groups: Groups): number => ({ two: 2, three: 3, open6: 6, open8: 8 })[groups]

/** Whether the number of groups is left to the data (stick-breaking weights). */
export const isOpen = (groups: Groups): boolean => groups === 'open6' || groups === 'open8'

/** A group holds real weight when it holds at least this share of the eruptions. */
export const REAL = 0.05

/** An eruption is clear when its group is at least this likely. */
export const CLEAR = 0.9

/**
 * The choices expectation propagation runs: the one it settles on in under about ten seconds. On the others it takes much longer
 * (a diagonal pair of groups: 10 to 32 sweeps of nearly a second each, depending on the machine) or does not settle in 200
 * sweeps (full groups), so the sampler runs them.
 */
const EP_BUILDS: readonly Build[] = [{ shape: 'round', groups: 'two' }]

/** Which method runs a choice: expectation propagation where it settles in a few seconds, else Gibbs sampling. */
export function engineOf(build: Build): 'ep' | 'gibbs' {
  return EP_BUILDS.some((b) => b.shape === build.shape && b.groups === build.groups) ? 'ep' : 'gibbs'
}

/** One group of the fit, in the data's own units (minutes). */
export interface GroupFit {
  /** The share of the eruptions it holds (0 to 1). */
  readonly weight: number
  /** The centre: the eruption length and the wait. */
  readonly centre: readonly [number, number]
  /** The spread in each direction (standard deviation). */
  readonly spread: readonly [number, number]
  /** The correlation inside the group that its shape allows (0 for a round or a diagonal group). */
  readonly tilt: number
  /** A closed curve two spreads out from the centre, as points to draw. */
  readonly ellipse: { readonly x: readonly number[]; readonly y: readonly number[] }
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  readonly build: Build
  /** Which method ran it. */
  readonly engine: 'ep' | 'gibbs'
  /** How many pieces the model had room for. */
  readonly pieces: number
  /** The groups, the heaviest first. */
  readonly groups: readonly GroupFit[]
  /** For each eruption (in the order of the data): the index in `groups` of the group it most likely belongs to. */
  readonly group: readonly number[]
  /** For each eruption: how likely that group is (0 to 1). */
  readonly sure: readonly number[]
  /** Whether the method finished cleanly (EP settled; Gibbs ran its sweeps). */
  readonly settled: boolean
  /** The sweeps of the method's last pass (EP) or the sweeps run (Gibbs). */
  readonly sweeps: number
}

/** How much to run: the real run, or a small one for a test. */
export interface Size {
  readonly rows: number
  readonly warmup: number
  readonly draws: number
  readonly ellipse: number
}

/**
 * The size of the real run of a choice. The sampler's sweeps cost more the more the model has to move: a group with a spread of
 * its own in each direction, with room for 6 or 8 groups, takes half the sweeps so that no choice runs for long.
 *
 * @param build - The reader's choice.
 * @returns The size.
 */
export function sizeOf(build: Build): Size {
  const heavy = build.shape !== 'round' && isOpen(build.groups)
  return { rows: 272, warmup: heavy ? 30 : 60, draws: heavy ? 40 : 80, ellipse: 48 }
}

// region model ep gibbs
/**
 * The model of a choice: each eruption belongs to one group, and its length and wait fall
 * around that group's centre. A spread is written as a precision with a gamma prior, which is
 * the form expectation propagation reads.
 */
export function mixtureModel(build: Build): ReturnType<typeof defineModel> {
  const pieces = piecesOf(build.groups)
  return defineModel(`mixture-${build.shape}-${build.groups}`, (m) => {
    const row = m.range('row')
    const group = m.range('group', { size: pieces })
    const length = m.data('length', { dims: [row] })
    const wait = m.data('wait', { dims: [row] })
    // the share of the eruptions in each group
    let shares: Var
    if (isOpen(build.groups)) {
      // stick-breaking: each piece takes a part of what is left, so the data can leave late pieces tiny
      const sticks = m.range('stick', { size: pieces - 1 })
      const breaks = m.latent('breaks', d.beta(1, 1), { dims: [sticks] })
      shares = m.derived('shares', ops.stickBreaking(breaks))
    } else {
      shares = m.latent('shares', d.dirichlet(full([pieces], 1)))
    }
    // centres in the length direction; for a fixed number of groups they are in order
    const inOrder = isOpen(build.groups) ? {} : { constraint: { kind: 'ordered' as const } }
    const centreL = m.latent('centreL', d.normal(0, 1.5), { dims: [group], ...inOrder })
    const which = m.latent('which', d.categorical({ probs: shares }), { dims: [row] })
    if (build.shape === 'round') {
      // one spread, for both directions and every group
      const precision = m.latent('precision', d.gamma(2, 0.5))
      const sd = ops.div(1, ops.sqrt(precision))
      const centreW = m.latent('centreW', d.normal(0, 1.5), { dims: [group] })
      m.observe('length', d.normal(centreL.at(which), sd), length)
      m.observe('wait', d.normal(centreW.at(which), sd), wait)
    } else if (build.shape === 'diagonal') {
      // a spread of its own in each direction, for each group
      const precisionL = m.latent('precisionL', d.gamma(2, 0.5), { dims: [group] })
      const precisionW = m.latent('precisionW', d.gamma(2, 0.5), { dims: [group] })
      const sdL = ops.div(1, ops.sqrt(precisionL.at(which)))
      const sdW = ops.div(1, ops.sqrt(precisionW.at(which)))
      const centreW = m.latent('centreW', d.normal(0, 1.5), { dims: [group] })
      m.observe('length', d.normal(centreL.at(which), sdL), length)
      m.observe('wait', d.normal(centreW.at(which), sdW), wait)
    } else {
      // the same, and inside each group the wait leans on the length: the lean is the tilt
      const precisionL = m.latent('precisionL', d.gamma(2, 0.5), { dims: [group] })
      const precisionW = m.latent('precisionW', d.gamma(2, 0.5), { dims: [group] })
      const sdL = ops.div(1, ops.sqrt(precisionL.at(which)))
      const sdW = ops.div(1, ops.sqrt(precisionW.at(which)))
      const lean = m.latent('lean', d.normal(0, 1.5), { dims: [group] })
      const offset = m.latent('offset', d.normal(0, 1.5), { dims: [group] })
      const line = ops.add(ops.mul(length, lean), offset)
      m.observe('length', d.normal(centreL.at(which), sdL), length)
      m.observe('wait', d.normal(ops.pick(line, which), sdW), wait)
    }
  })
}
// endregion

/** The beliefs of a method, and how it ended. */
interface Fitted {
  readonly beliefs: {
    mean(site: string): Value
    marginal(site: string): { params: Readonly<Record<string, Value>> }
    draws?: Readonly<Record<string, Value>>
  }
  readonly settled: boolean
  readonly sweeps: number
}

/** Expectation propagation on the bound model: one step is one sweep of messages from every eruption to the groups and back. */
function runEp(model: ReturnType<typeof mixtureModel>, data: Record<string, Value>): Fitted {
  // region ep
  // the model and the eruptions, put together
  const bound = model.bind(data)
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no message-passing route: ${route.verdict.reason}`)
  // sweep until no belief moves by more than a hundredth
  const run = session(route.method, { steps: 80, mode: 'eager', config: { tol: 1e-2 } })
  // the beliefs the messages settled on
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  return { beliefs, settled: run.status === 'converged', sweeps: run.t }
}

/** Gibbs sampling on the bound model: each sweep draws the group of every eruption, then the groups' parameters. */
function runGibbs(
  model: ReturnType<typeof mixtureModel>,
  data: Record<string, Value>,
  seed: number,
  size: Size,
): Fitted {
  // region gibbs
  // the model and the eruptions, put together
  const bound = model.bind(data)
  const method = gibbsMethod(bound)
  // the first sweeps are warm-up, the rest are kept
  const config = { numWarmup: size.warmup, numDraws: size.draws }
  const run = session(method, { steps: size.warmup + size.draws, mode: 'eager', seed, config })
  // the draws the sweeps kept
  const beliefs = method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  return { beliefs, settled: run.status === 'completed', sweeps: run.t }
}

/** How likely each eruption is to be in each group, as a tensor of rows by groups. */
function responsibilities(fitted: Fitted, pieces: number, rows: number): Value {
  const draws = fitted.beliefs.draws?.['which']
  if (draws === undefined) return fitted.beliefs.marginal('which').params['probs'] as Value
  const each = equal(reshape(draws, [-1, rows, 1]), reshape(arange(pieces), [1, 1, pieces]))
  return mean(cast(each, 'float64'), 0)
}

/** A belief about one value per group (a single value, when the model has one for all of them, is repeated). */
const perGroup = (fitted: Fitted, name: string, pieces: number): Value =>
  broadcastTo(fitted.beliefs.mean(name), [pieces])

/**
 * The run of a choice, from a seed: put the two columns on a scale of about one, hand the eruptions to the method in the seed's
 * order, fit, and turn the beliefs into groups (centre, spread, tilt, ellipse) in minutes, heaviest first.
 *
 * @param seed - Decides the order of the eruptions (and the sampler's draws).
 * @param say - Says what is happening, for the play button.
 * @param build - The reader's choice of shape and number of groups.
 * @param size - How much to run.
 * @returns The result.
 */
export function fit(
  seed: number,
  say: Say = () => {},
  build: Build = DEFAULT_BUILD,
  size: Size = sizeOf(build),
): Result {
  const engine = engineOf(build)
  const pieces = piecesOf(build.groups)
  say('Writing the model')
  const data = columns()
  const rawL = sliceAxis(data.length, 0, 0, size.rows)
  const rawW = sliceAxis(data.wait, 0, 0, size.rows)
  // both measures are put on a scale of about one: the model is written for that scale
  const [centreL, scaleL] = [mean(rawL), std(rawL)]
  const [centreW, scaleW] = [mean(rawW), std(rawW)]
  const order = argsort(uniform(foldIn(key(seed), 0), [size.rows]), 0)
  const length = take(div(sub(rawL, centreL), scaleL), order, 0)
  const wait = take(div(sub(rawW, centreW), scaleW), order, 0)
  const model = mixtureModel(build)
  say(
    engine === 'ep'
      ? 'Passing messages between the eruptions and the groups'
      : 'Sampling the groups and their eruptions',
  )
  const fitted = engine === 'ep' ? runEp(model, { length, wait }) : runGibbs(model, { length, wait }, seed, size)
  say('Reading off the groups')
  // how likely each eruption is to be in each group, back in the order of the data
  const chance = take(responsibilities(fitted, pieces, size.rows), argsort(order, 0), 0)
  const shares = mean(chance, 0)
  // the groups, heaviest first
  const rank = argsort(mul(shares, -1), 0)
  const sorted = (v: Value): Value => take(v, rank, 0)
  // the centre of each group, in standard units
  const cL = perGroup(fitted, 'centreL', pieces)
  const shape = build.shape
  const lean = shape === 'full' ? perGroup(fitted, 'lean', pieces) : mul(cL, 0)
  const cW =
    shape === 'full' ? add(mul(lean, cL), perGroup(fitted, 'offset', pieces)) : perGroup(fitted, 'centreW', pieces)
  // the spreads (variances) and the covariance of each group, in standard units
  const varL =
    shape === 'round' ? div(1, perGroup(fitted, 'precision', pieces)) : div(1, perGroup(fitted, 'precisionL', pieces))
  const varW0 = shape === 'round' ? varL : div(1, perGroup(fitted, 'precisionW', pieces))
  const cov = mul(lean, varL)
  const varW = add(varW0, mul(lean, cov))
  // into minutes
  const [sL, sW] = [toFlat(scaleL)[0] as number, toFlat(scaleW)[0] as number]
  const centreMinutes = [add(mul(sorted(cL), scaleL), centreL), add(mul(sorted(cW), scaleW), centreW)]
  const spreadMinutes = [mul(sqrt(sorted(varL)), scaleL), mul(sqrt(sorted(varW)), scaleW)]
  const tilt = div(sorted(cov), sqrt(mul(sorted(varL), sorted(varW))))
  // the curve two spreads from the centre: the first column of the Cholesky factor of the covariance, turned through an angle
  const angle = reshape(linspace(0, 2 * Math.PI, size.ellipse), [1, -1])
  const l11 = sqrt(sorted(varL))
  const l21 = div(sorted(cov), l11)
  const l22 = sqrt(sub(sorted(varW), square(l21)))
  const [cosA, sinA] = [cos(angle), sin(angle)]
  const column = (v: Value): Value => reshape(v, [-1, 1])
  const edgeL = add(column(centreMinutes[0] as Value), mul(mul(column(l11), cosA), 2 * sL))
  const edgeW = add(column(centreMinutes[1] as Value), mul(add(mul(column(l21), cosA), mul(column(l22), sinA)), 2 * sW))
  const weights = toFlat(sorted(shares))
  const centres = [toFlat(centreMinutes[0] as Value), toFlat(centreMinutes[1] as Value)]
  const spreads = [toFlat(spreadMinutes[0] as Value), toFlat(spreadMinutes[1] as Value)]
  const tilts = toFlat(tilt)
  const edges = [toFlat(edgeL), toFlat(edgeW)]
  const n = size.ellipse
  const groups = weights.map(
    (weight, i): GroupFit => ({
      weight,
      centre: [centres[0]?.[i] as number, centres[1]?.[i] as number],
      spread: [spreads[0]?.[i] as number, spreads[1]?.[i] as number],
      tilt: tilts[i] as number,
      ellipse: {
        x: (edges[0] as number[]).slice(i * n, (i + 1) * n),
        y: (edges[1] as number[]).slice(i * n, (i + 1) * n),
      },
    }),
  )
  // each eruption's group: the heaviest group has index 0, and so on
  const place = argsort(rank, 0)
  return {
    build,
    engine,
    pieces,
    groups,
    group: toFlat(take(place, argmax(chance, 1), 0)),
    sure: toFlat(max(chance, 1)),
    settled: fitted.settled,
    sweeps: fitted.sweeps,
  }
}

/**
 * The run of the story: the fit of the choice on all the eruptions.
 *
 * @param seed - Decides the order of the eruptions.
 * @param say - Says what is happening.
 * @param build - The reader's choice (default: round groups, two of them).
 * @returns The result.
 */
export const run = (seed: number, say?: Say, build: Build = DEFAULT_BUILD): Result =>
  fit(seed, say, build, sizeOf(build))
