/**
 * Predictive distributions of probabilistic programs, NumPyro style: draws from the prior predictive and the posterior
 * predictive, and per-observation log-likelihoods under posterior draws, flattened into the matrix that WAIC and LOO
 * take.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  asTensor,
  concat,
  fromStorage,
  primalOf,
  reshape,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { maskedLogProb } from './_contract.ts'
import { handle, type ProgramTrace, seed, traceOf, uncondition } from './handlers.ts'
import type { Posterior } from './posterior.ts'
import type { Message, Model } from './primitives.ts'

/** Options of the predictive functions. */
export interface PredictiveOptions {
  readonly key: Key
  /** Draws (default 1000; for posterior predictives, the posterior is drawn this many times). */
  readonly numSamples?: number
  /** The sites to return (default every observed and deterministic site). */
  readonly sites?: readonly string[]
}

function stack(traces: readonly ProgramTrace[], names: readonly string[]): Record<string, Tensor> {
  const out: Record<string, Tensor> = {}
  for (const name of names) {
    const rows: number[] = []
    let shape: readonly number[] = []
    for (const t of traces) {
      const s = t.sites.get(name)
      if (s === undefined || s.type === 'plate' || s.type === 'markov') continue
      shape = s.value.shape
      rows.push(...toFlat(primalOf(s.value)))
    }
    out[name] = fromStorage(Float64Array.from(rows), [traces.length, ...shape], 'float64')
  }
  return out
}

function observedNames(t: ProgramTrace | undefined, deterministic: boolean): string[] {
  if (t === undefined) return []
  return [...t.sites.values()]
    .filter(
      (s) => (s.type === 'sample' && s.isObserved && !s.isFactor) || (deterministic && s.type === 'deterministic'),
    )
    .map((s) => s.name)
}

/** Fix the latent sample sites of a run at given values (deterministic sites are recomputed, observed sites untouched). */
function atLatents<D = void>(model: Model<D>, values: Readonly<Record<string, Value>>): Model<D> {
  return handle(model, () => ({
    process: (msg: Message) => {
      if (msg.type !== 'sample' || msg.isObserved) return
      const v = values[msg.name]
      if (v !== undefined) msg.value = v
    },
  }))
}

/**
 * Draws from the prior predictive distribution (NumPyro's `Predictive` without samples): run the program with its
 * observed sites turned latent, `numSamples` times.
 *
 * @param model - The program.
 * @param data - Its data (observations are ignored and resampled).
 * @param options - Key, number of draws and sites.
 * @returns Draws by site `[n, ...shape]` (observed and deterministic sites by default; latent sites when asked).
 * @concept predictive-checks
 * @example
 * ```ts
 * import { observe, plate, priorPredictive, sample } from 'mdml-engine/probprog'
 * import { normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * const model = () => { const m = sample('m', normal(0, 1)); plate('n', 3, () => observe('y', normal(m, 1), [0, 0, 0])) }
 * priorPredictive(model, undefined, { key: key(0), numSamples: 10 })['y']?.shape // => [10, 3]
 * ```
 */
export function priorPredictive<D = void>(
  model: Model<D>,
  data: D,
  options: PredictiveOptions,
): Record<string, Tensor> {
  const n = options.numSamples ?? 1000
  const traces = Array.from({ length: n }, (_, i) => traceOf(seed(uncondition(model), foldIn(options.key, i)), data))
  const first = traceOf(seed(model, foldIn(options.key, n)), data)
  const names = options.sites ?? observedNames(first, true)
  return stack(traces, names)
}

/**
 * Draws from the posterior predictive distribution (NumPyro's `Predictive(model, posterior_samples)`): for each
 * posterior draw, fix the latent sites and run the program with its observed sites resampled. Pass new data (new
 * covariates) to predict elsewhere.
 *
 * @param model - The program.
 * @param posterior - The posterior.
 * @param data - The data to predict at.
 * @param options - Key, number of draws and sites.
 * @returns Replicated draws by site `[n, ...shape]`.
 * @concept predictive-checks
 * @example
 * ```ts
 * import { conjugateInference, observe, plate, posteriorPredictive, sample } from 'mdml-engine/probprog'
 * import { normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * const model = (y: number[]) => { const m = sample('m', normal(0, 1)); plate('n', y.length, () => observe('y', normal(m, 1), y)) }
 * const post = conjugateInference(model, [1, 2, 3])
 * posteriorPredictive(model, post, [1, 2, 3], { key: key(0), numSamples: 20 })['y']?.shape // => [20, 3]
 * ```
 */
export function posteriorPredictive<D = void>(
  model: Model<D>,
  posterior: Posterior,
  data: D,
  options: PredictiveOptions,
): Record<string, Tensor> {
  const n = options.numSamples ?? 1000
  const draws = posterior.sample(foldIn(options.key, 0), n)
  const traces: ProgramTrace[] = []
  for (let i = 0; i < n; i++) {
    const values = rowOf(draws, i)
    traces.push(traceOf(seed(uncondition(atLatents(model, values)), foldIn(foldIn(options.key, 1), i)), data))
  }
  // The replicated sites: every sample site the posterior does not cover (observed in training, or new at these
  // data), and the deterministic sites.
  const first = traces[0]
  const names =
    options.sites ??
    (first === undefined
      ? []
      : [...first.sites.values()]
          .filter(
            (x) => (x.type === 'sample' && !x.isFactor && draws[x.name] === undefined) || x.type === 'deterministic',
          )
          .map((x) => x.name))
  return stack(traces, names)
}

// Draws for latent sites missing from the posterior draws (only used when no key is passed).
const fallbackKey: Key = makeKey(0)

function isPosterior(x: Readonly<Record<string, Tensor>> | Posterior): x is Posterior {
  const p = x as Partial<Posterior>
  return typeof p.form === 'string' && typeof p.engine === 'string' && typeof p.summary === 'function'
}

function rowOf(draws: Readonly<Record<string, Tensor>>, i: number): Record<string, Value> {
  const out: Record<string, Value> = {}
  for (const [name, t] of Object.entries(draws)) {
    const shape = t.shape.slice(1)
    const size = shape.reduce((a, b) => a * b, 1)
    out[name] = fromStorage(Float64Array.from(t.data).slice(i * size, (i + 1) * size), shape, 'float64')
  }
  return out
}

/**
 * Per-observation log-likelihoods under posterior draws (NumPyro's `log_likelihood`): for each draw, the log density of
 * every element of every observed site (plate axes kept), the input of WAIC and PSIS-LOO.
 *
 * @param model - The program.
 * @param draws - Posterior draws by site `[S, ...shape]` (latent sites), or a posterior to draw from.
 * @param data - The data.
 * @param options - For a posterior: the key and number of draws; and the observed sites (default all).
 * @returns Log-likelihoods by observed site `[S, ...batch]`.
 * @concept predictive-checks
 * @example
 * ```ts
 * import { logLikelihood, observe, plate, sample } from 'mdml-engine/probprog'
 * import { normal } from 'mdml-engine/dists'
 * import { tensor } from 'mdml-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); plate('n', 2, () => observe('y', normal(m, 1), [0, 1])) }
 * logLikelihood(model, { m: tensor([0, 0.5, 1]) }, undefined)['y']?.shape // => [3, 2]
 * ```
 */
export function logLikelihood<D = void>(
  model: Model<D>,
  draws: Readonly<Record<string, Tensor>> | Posterior,
  data: D,
  options: { readonly key?: Key; readonly numSamples?: number; readonly sites?: readonly string[] } = {},
): Record<string, Tensor> {
  let values: Readonly<Record<string, Tensor>>
  if (isPosterior(draws)) {
    const post = draws
    if (post.draws !== undefined && options.numSamples === undefined && post.logWeights === undefined)
      values = post.draws
    else {
      if (options.key === undefined) throw new MdmlError('value', 'logLikelihood: drawing from a posterior needs a key')
      values = post.sample(options.key, options.numSamples ?? 1000)
    }
  } else values = draws
  const S = Object.values(values)[0]?.shape[0] ?? 0
  const rows: Record<string, number[]> = {}
  const shapes: Record<string, readonly number[]> = {}
  for (let i = 0; i < S; i++) {
    const t = traceOf(seed(atLatents(model, rowOf(values, i)), foldIn(options.key ?? fallbackKey, i)), data)
    for (const s of t.sites.values()) {
      if (s.type !== 'sample' || !s.isObserved || s.isFactor) continue
      if (options.sites !== undefined && !options.sites.includes(s.name)) continue
      const lp = primalOf(maskedLogProb(s))
      shapes[s.name] = lp.shape
      rows[s.name] = [...(rows[s.name] ?? []), ...toFlat(lp)]
    }
  }
  const out: Record<string, Tensor> = {}
  for (const [name, r] of Object.entries(rows))
    out[name] = fromStorage(Float64Array.from(r), [S, ...(shapes[name] ?? [])], 'float64')
  return out
}

/**
 * Per-observation log-likelihoods of every observed site flattened and joined: the `[S, N]` matrix WAIC and LOO take.
 *
 * @param logLik - Log-likelihoods by site `[S, ...batch]` (from {@link logLikelihood}).
 * @returns The `[S, N]` matrix.
 * @concept predictive-checks
 * @example
 * ```ts
 * import { pointwiseLogLikelihood } from 'mdml-engine/probprog'
 * import { tensor } from 'mdml-engine/tensor'
 * pointwiseLogLikelihood({ a: tensor([[1, 2], [3, 4]]), b: tensor([5, 6]) }).shape // => [2, 3]
 * ```
 */
export function pointwiseLogLikelihood(logLik: Readonly<Record<string, TensorLike>>): Tensor {
  const parts = Object.values(logLik).map((v) => {
    const t = asTensor(v)
    const S = t.shape[0] ?? 0
    return reshape(t, [S, -1])
  })
  if (parts.length === 0) throw new MdmlError('value', 'pointwiseLogLikelihood: no observed sites')
  return primalOf(concat(parts, 1))
}
