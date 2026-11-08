/**
 * A dataset's generating program: `generatingProgram` wraps a probabilistic program over rows of features and targets
 * so a synthetic dataset can carry its ground truth, giving exact per-row log densities, draws of new rows and, where
 * it exists, the true regression function.
 */

import { OmaiError } from '../base/index.ts'
import { type GeneratingProgram, posteriorFromJoint } from '../data/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  add,
  concat,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  tensor,
  type Value,
} from '../tensor/index.ts'
import { maskedLogProb } from './_contract.ts'
import { logJoint } from './density.ts'
import { enumerate, handle, isDiscrete, type ProgramTrace, type SampleSite, seed, traceOf } from './handlers.ts'
import type { Message, Model } from './primitives.ts'

/** The data a generating program takes: features, targets (when known) and the number of rows. */
export interface RowsData {
  readonly x?: Value
  readonly y?: Value
  readonly n: number
}

/** Options of {@link generatingProgram}. */
export interface GeneratingProgramOptions {
  /** The site of the features (default `x`). */
  readonly features?: string
  /** The site of the targets (default `y`; `null` for programs without targets). */
  readonly target?: string | null
  /** The number of classes of a classification program (the posterior is then p(y | x) by Bayes' rule). */
  readonly classes?: number
}

/** A site's log density summed over every axis but the rows axis (the first axis of length n), scaled. */
function perRow(site: SampleSite, n: number): Value {
  const lp = maskedLogProb(site)
  const axis = lp.shape.indexOf(n)
  const r =
    axis < 0
      ? reshape(sum(lp), [1])
      : (() => {
          const others = lp.shape.map((_, a) => a).filter((a) => a !== axis)
          return others.length === 0 ? lp : sum(lp, others)
        })()
  return site.scale === 1 ? r : mul(r, site.scale)
}

/** The program with one site's log density dropped (a leaf that integrates to one). */
function withoutDensity<D>(model: Model<D>, site: string): Model<D> {
  return handle(model, () => ({
    process: (msg: Message) => {
      if (msg.type === 'sample' && msg.name === site) msg.mask = tensor(false)
    },
  }))
}

/**
 * A dataset's generating program as `data`'s `Truth.program` (the hand-off section 4.11 of the design describes): a
 * program over rows `{ x?, y?, n }` gives per-row log densities log p(x, y), and log p(x) without targets (a discrete
 * target is summed out, a continuous one integrates to one); other discrete latent sites are summed out exactly, row
 * by row. Classification programs get the class posterior p(y | x) by Bayes' rule, programs whose target depends on the
 * features directly get E[y | x] from the target's distribution, and every program draws rows.
 *
 * @param model - The program.
 * @param options - The feature and target sites and the number of classes.
 * @returns The generating program.
 * @throws OmaiError with code `value` (from its methods) when a site is missing or E[y | x] is not direct.
 * @concept dataset-truth
 * @example
 * ```ts
 * import { generatingProgram, plate, type RowsData, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item, mul, tensor, toArray } from 'omai-engine/tensor'
 * const line = (d: RowsData) => plate('rows', d.n, () => {
 *   const x = sample('x', normal(0, 1), { obs: d.x })
 *   sample('y', normal(mul(x, 2), 0.5), { obs: d.y })
 * })
 * const program = generatingProgram(line)
 * const lp = -0.5 - 0.5 * Math.log(2 * Math.PI) - Math.log(0.5) - 0.5 * Math.log(2 * Math.PI)
 * Math.abs(item(program.logProb(tensor([1]), tensor([2]))) - lp) < 1e-12 // => true
 * toArray(program.regressionFunction?.(tensor([1, 3])) ?? tensor([])) // => [2, 6]
 * program.sample(key(0), 5).x.shape // => [5]
 * ```
 */
export function generatingProgram(model: Model<RowsData>, options: GeneratingProgramOptions = {}): GeneratingProgram {
  const features = options.features ?? 'x'
  const target = options.target === null ? undefined : (options.target ?? 'y')
  const otherLatents = (t: ProgramTrace): SampleSite[] =>
    [...t.sites.values()].filter(
      (s): s is SampleSite => s.type === 'sample' && !s.isObserved && s.name !== features && s.name !== target,
    )
  const logProb = (x: Value, y?: Value): Value => {
    const n = x.shape[0] ?? 0
    const data: RowsData = { x, n, ...(y === undefined ? {} : { y }) }
    const t = traceOf(seed(model, makeKey(0)), data)
    const latent = otherLatents(t)
    const targetSite = target === undefined ? undefined : t.sites.get(target)
    const discreteTarget = targetSite?.type === 'sample' && isDiscrete(targetSite.fn.support)
    if (latent.length === 0 && (y !== undefined || target === undefined || !discreteTarget)) {
      // Every random site is observed, or a continuous target is absent and integrates to one.
      let total: Value = tensor(new Array<number>(n).fill(0))
      for (const s of t.sites.values()) if (s.type === 'sample' && s.isObserved) total = add(total, perRow(s, n))
      return total
    }
    const rowModel = y === undefined && target !== undefined && !discreteTarget ? withoutDensity(model, target) : model
    for (const s of latent)
      if (!isDiscrete(s.fn.support))
        throw new OmaiError(
          'value',
          `generatingProgram: the latent site ${s.name} is not discrete, so log p(x, y) is not exact`,
        )
    const rows: Value[] = []
    for (let i = 0; i < n; i++) {
      const d: RowsData = {
        x: sliceAxis(x, 0, i, i + 1),
        n: 1,
        ...(y === undefined ? {} : { y: sliceAxis(y, 0, i, i + 1) }),
      }
      rows.push(reshape(logJoint(enumerate(rowModel), d).logJoint, [1]))
    }
    return concat(rows)
  }
  const classes = options.classes
  return {
    kind: 'program',
    model: model as (data: never) => unknown,
    features,
    ...(target === undefined ? {} : { target }),
    ...(classes === undefined ? {} : { classes }),
    logProb,
    ...(classes === undefined || target === undefined
      ? {}
      : { posterior: (x: Value) => posteriorFromJoint((xx, yy) => logProb(xx, yy), classes)(x) }),
    ...(target === undefined
      ? {}
      : {
          regressionFunction: (x: Value): Value => {
            const t = traceOf(seed(model, makeKey(0)), { x, n: x.shape[0] ?? 0 })
            const s = t.sites.get(target)
            if (s?.type !== 'sample') throw new OmaiError('value', `generatingProgram: no target site ${target}`)
            if (otherLatents(t).length > 0)
              throw new OmaiError(
                'value',
                'generatingProgram: the target depends on latent sites, so E[y | x] is not direct',
              )
            const m = primalOf(s.fn.mean())
            const n = x.shape[0] ?? 0
            // A per-row target with singleton axes (a [n, 1] column) reads as [n].
            return m.shape.length > 1 && m.shape.slice(1).every((v) => v === 1) ? primalOf(reshape(m, [n])) : m
          },
        }),
    sample: (k: Key, n: number) => {
      const t = traceOf(seed(model, foldIn(k, 0)), { n })
      const fx = t.sites.get(features)
      if (fx?.type !== 'sample') throw new OmaiError('value', `generatingProgram: no feature site ${features}`)
      const x = primalOf(fx.value) as Tensor
      const fy = target === undefined ? undefined : t.sites.get(target)
      return fy?.type === 'sample' ? { x, y: primalOf(fy.value) } : { x }
    },
  }
}
