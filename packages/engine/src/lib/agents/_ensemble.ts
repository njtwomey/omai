// Probabilistic ensembles of MLP dynamics models (PETS, MBPO): the network with normalised inputs and targets and
// learned log-variance bounds, bootstrap training by Gaussian negative log-likelihood, and batched trajectory
// sampling for planners. Private to methods/rl/agents.
import { meanOf, varianceOf } from '../base/index.ts'
import { gaussianNll } from '../losses/index.ts'
import { mlp, SiLU } from '../nn/index.ts'
import type { GradientTransformation, ParamTree } from '../optim/index.ts'
import { foldIn, type Key, randint } from '../random/index.ts'
import { type DynamicsModel, type Ensemble, ensembleSample } from '../rl/index.ts'
import {
  add,
  concat,
  exp,
  fromStorage,
  item,
  mul,
  sliceAxis,
  softplus,
  sub,
  sum,
  type Tensor,
  type Value,
} from '../tensor/index.ts'
import { gradientStep, rowsTensor, vectorTensor } from './_common.ts'

/** Per-feature mean and standard deviation of the model's inputs and targets (fitted to the data at each training). */
export interface Normaliser {
  readonly inMean: readonly number[]
  readonly inStd: readonly number[]
  readonly outMean: readonly number[]
  readonly outStd: readonly number[]
}

/** One member: network weights, the learned log-variance bounds, and its normaliser. */
export interface Member {
  readonly params: { readonly net: ParamTree; readonly maxLogVar: Tensor; readonly minLogVar: Tensor }
  readonly norm: Normaliser
}

/** A probabilistic MLP dynamics model of `[delta observation, reward]` and the means to initialise its members. */
export interface ProbabilisticNet {
  readonly model: DynamicsModel<Member>
  init(key: Key): Member
  /** The member's normalised mean and log-variance (what training fits), from normalised inputs. */
  raw(params: Member['params'], inputs: Value): { readonly mean: Value; readonly logVar: Value }
}

const fill = (n: number, v: number): Tensor => fromStorage(new Float64Array(n).fill(v), [n], 'float64')

/** Build the network: inputs `[obs, action]`, outputs a mean and a bounded log-variance per target (Chua et al.). */
export function probabilisticNet(
  observationSize: number,
  actionSize: number,
  hidden: readonly number[],
  key: Key,
): ProbabilisticNet {
  const d = observationSize + 1
  const inputs = observationSize + actionSize
  const net = mlp([inputs, ...hidden, 2 * d], { activation: () => new SiLU(), key })
  const raw = (p: Member['params'], x: Value): { mean: Value; logVar: Value } => {
    const out = net.apply(p.net as never, x)
    const mean = sliceAxis(out, -1, 0, d)
    const rawLv = sliceAxis(out, -1, d, 2 * d)
    // Soft bounds: maxLogVar - softplus(maxLogVar - lv), then minLogVar + softplus(lv - minLogVar).
    const upper = sub(p.maxLogVar, softplus(sub(p.maxLogVar, rawLv)))
    return { mean, logVar: add(p.minLogVar, softplus(sub(upper, p.minLogVar))) }
  }
  const model: DynamicsModel<Member> = {
    observationSize,
    actionSize,
    predict: (m, obs, act) => {
      const x = normaliseRows(concat([obs, act], -1), m.norm.inMean, m.norm.inStd)
      const r = raw(m.params, x)
      const std = vectorTensor(m.norm.outStd)
      return {
        mean: add(mul(r.mean, std), vectorTensor(m.norm.outMean)),
        logVar: add(r.logVar, vectorTensor(m.norm.outStd.map((s) => 2 * Math.log(s)))),
      }
    },
  }
  return {
    model,
    raw,
    init: (k) => ({
      params: {
        net: mlp([inputs, ...hidden, 2 * d], { activation: () => new SiLU(), key: k }).params,
        maxLogVar: fill(d, 0.5),
        minLogVar: fill(d, -10),
      },
      norm: {
        inMean: new Array<number>(inputs).fill(0),
        inStd: new Array<number>(inputs).fill(1),
        outMean: new Array<number>(d).fill(0),
        outStd: new Array<number>(d).fill(1),
      },
    }),
  }
}

function normaliseRows(x: Value, mean: readonly number[], std: readonly number[]): Value {
  return mul(sub(x, vectorTensor(mean)), vectorTensor(std.map((s) => 1 / s)))
}

function columnStats(rows: readonly (readonly number[])[], width: number): { mean: number[]; std: number[] } {
  const mean: number[] = []
  const std: number[] = []
  for (let j = 0; j < width; j++) {
    const col = rows.map((r) => r[j] ?? 0)
    mean.push(meanOf(col))
    std.push(Math.max(Math.sqrt(varianceOf(col)), 1e-6))
  }
  return { mean, std }
}

/** Supervised data for the ensemble: model inputs `[obs, encoded action]` and targets `[delta obs, reward]`. */
export interface ModelData {
  readonly inputs: readonly (readonly number[])[]
  readonly targets: readonly (readonly number[])[]
}

/** Options of one training round. */
export interface TrainOptions {
  readonly optimizer: GradientTransformation
  readonly gradientSteps: number
  readonly batchSize: number
}

/**
 * One training round: refit the normalisers to all the data, then give each member `gradientSteps` minibatch Adam
 * steps on its own bootstrap resample of the data (sampling with replacement), minimising the Gaussian negative
 * log-likelihood of the normalised targets plus `0.01 (sum maxLogVar - sum minLogVar)`.
 */
export function trainEnsemble(
  net: ProbabilisticNet,
  members: readonly Member[],
  optStates: readonly unknown[],
  data: ModelData,
  options: TrainOptions,
  key: Key,
): { readonly members: Member[]; readonly optStates: unknown[]; readonly loss: number } {
  const n = data.inputs.length
  const inW = data.inputs[0]?.length ?? 0
  const outW = data.targets[0]?.length ?? 0
  const inStats = columnStats(data.inputs, inW)
  const outStats = columnStats(data.targets, outW)
  const norm: Normaliser = { inMean: inStats.mean, inStd: inStats.std, outMean: outStats.mean, outStd: outStats.std }
  const xs = data.inputs.map((r) => r.map((v, j) => (v - (inStats.mean[j] ?? 0)) / (inStats.std[j] ?? 1)))
  const ys = data.targets.map((r) => r.map((v, j) => (v - (outStats.mean[j] ?? 0)) / (outStats.std[j] ?? 1)))
  let total = 0
  const out: Member[] = []
  const states: unknown[] = []
  for (const [i, member] of members.entries()) {
    const km = foldIn(key, i)
    // The member's bootstrap resample, drawn once per round.
    const boot = Array.from(randint(foldIn(km, 0), [n], 0, n).data)
    let params: ParamTree = member.params
    let opt = optStates[i]
    let loss = 0
    for (let s = 0; s < options.gradientSteps; s++) {
      const pick = Array.from(randint(foldIn(km, s + 1), [Math.min(options.batchSize, n)], 0, n).data).map(
        (j) => boot[j] ?? 0,
      )
      const x = rowsTensor(
        pick.map((j) => xs[j] ?? []),
        inW,
      )
      const y = rowsTensor(
        pick.map((j) => ys[j] ?? []),
        outW,
      )
      const lossFn = (p: ParamTree): Value => {
        const q = p as Member['params']
        const r = net.raw(q, x)
        const nll = gaussianNll(r.mean, y, exp(r.logVar))
        return add(nll, mul(sub(sum(q.maxLogVar), sum(q.minLogVar)), 0.01))
      }
      const step = gradientStep(lossFn, params, options.optimizer, opt)
      params = step.params
      opt = step.optState
      loss = step.loss
    }
    out.push({ params: params as Member['params'], norm })
    states.push(opt)
    total += loss / members.length
  }
  return { members: out, optStates: states, loss: total }
}

/**
 * The mean return of each action sequence under the ensemble, estimated with `particles` trajectory-sampling (TS-1)
 * particles per sequence, all sequences and particles stepped together in one batch per time step. A known reward
 * function replaces the predicted one; a known termination function stops a particle's rewards.
 */
export function particleReturns(
  ensemble: Ensemble<Member>,
  start: readonly number[],
  plans: readonly (readonly (readonly number[])[])[],
  particles: number,
  key: Key,
  options: {
    readonly encode: (action: readonly number[]) => readonly number[]
    readonly reward?: (obs: readonly number[], action: readonly number[], next: readonly number[]) => number
    readonly terminated?: (obs: readonly number[]) => boolean
    readonly gamma?: number
  },
): number[] {
  const P = plans.length
  const H = plans[0]?.length ?? 0
  const gamma = options.gamma ?? 1
  let obs: number[][] = Array.from({ length: P * particles }, () => [...start])
  const alive = new Array<boolean>(P * particles).fill(true)
  const ret = new Array<number>(P * particles).fill(0)
  let discount = 1
  for (let t = 0; t < H; t++) {
    const actions = obs.map((_, r) => plans[Math.floor(r / particles)]?.[t] ?? [])
    const next = ensembleSample(
      ensemble,
      obs,
      actions.map((a) => options.encode(a)),
      foldIn(key, t),
    )
    obs = next.map((p, r) => {
      if (alive[r] === true) {
        const reward = options.reward?.(obs[r] ?? [], actions[r] ?? [], p.observation) ?? p.reward
        ret[r] = (ret[r] ?? 0) + discount * reward
        if (options.terminated?.(p.observation) === true) alive[r] = false
      }
      return p.observation
    })
    discount *= gamma
  }
  return Array.from({ length: P }, (_, p) => {
    let s = 0
    for (let j = 0; j < particles; j++) s += ret[p * particles + j] ?? 0
    return s / particles
  })
}

/** The training log-likelihood of the targets under a member, for diagnostics. */
export function memberLoss(net: ProbabilisticNet, member: Member, data: ModelData): number {
  const x = normaliseRows(rowsTensor(data.inputs, data.inputs[0]?.length ?? 0), member.norm.inMean, member.norm.inStd)
  const y = normaliseRows(
    rowsTensor(data.targets, data.targets[0]?.length ?? 0),
    member.norm.outMean,
    member.norm.outStd,
  )
  const r = net.raw(member.params, x)
  return item(gaussianNll(r.mean, y, exp(r.logVar)))
}
