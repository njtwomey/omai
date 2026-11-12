/**
 * The posterior of a set of runs of message passing weighted by their evidence, as the library's `Posterior`: each run keeps
 * what it believes of every site (its mean and variance, the covariance of a vector site, and the family the belief was in:
 * `SiteRecord`), and the posterior is the mixture of the runs' beliefs, site by site, with the runs' weights. Shared by the
 * methods that run EP many times (inside a grid over hyperparameters, from several starts).
 */

import type { Json } from '../../base/index.ts'
import { OmaiError } from '../../base/index.ts'
import { mixture, mixtureSameFamily } from '../../dists/compose/index.ts'
import {
  beta as betaDist,
  categorical,
  type Distribution,
  gamma as gammaDist,
  multivariateNormal,
  normal,
} from '../../dists/index.ts'
import { uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import {
  argsort,
  concat,
  cumsum,
  div,
  fromStorage,
  maximum,
  minimum,
  mul,
  reshape,
  searchsorted,
  sqrt,
  stack as stackValues,
  sub,
  type Tensor,
  take,
  tensor,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import type { RunStatus } from '../../trace/index.ts'
import { definePosterior, type Posterior } from '../posterior.ts'

/** What one run keeps of a site's belief: enough to rebuild the marginal and to mix it. */
export interface SiteRecord {
  /** The family the belief was in (`normal`, `multivariateNormal`, `gamma`, `beta`), or `moments` for any other (read as a moment-matched normal). */
  readonly kind: 'normal' | 'multivariateNormal' | 'gamma' | 'beta' | 'categorical' | 'moments'
  readonly shape: readonly number[]
  readonly mean: readonly number[]
  readonly variance: readonly number[]
  /** The covariance of a vector site, row-major. */
  readonly covariance?: readonly number[]
  /** The probabilities of a categorical site (`shape.product` rows of one category axis, row-major). */
  readonly probs?: readonly number[]
}

/** What a node keeps of a distribution of a site. */
export function recordOf(d: Distribution): SiteRecord {
  const mean = Array.from(toFlat(d.mean()))
  const variance = Array.from(toFlat(d.variance()))
  const shape = [...d.batchShape, ...d.eventShape]
  const cov = (d as unknown as { covariance?: () => Tensor }).covariance
  if (d.name === 'multivariateNormal' && cov !== undefined)
    return { kind: 'multivariateNormal', shape, mean, variance, covariance: Array.from(toFlat(cov.call(d))) }
  if (d.name === 'categorical') {
    const probs = d.params['probs']
    if (probs !== undefined)
      return { kind: 'categorical', shape, mean, variance, probs: Array.from(toFlat(probs as Tensor)) }
  }
  const kind = d.name === 'normal' || d.name === 'gamma' || d.name === 'beta' ? d.name : 'moments'
  return { kind, shape, mean, variance }
}

/** The distribution a record stands for. */
export function distributionOf(r: SiteRecord): Distribution {
  const t = (xs: readonly number[]): Tensor => fromStorage(Float64Array.from(xs), [...r.shape], 'float64')
  if (r.kind === 'multivariateNormal' && r.covariance !== undefined) {
    const n = r.mean.length
    const cov = fromStorage(Float64Array.from(r.covariance), [n, n], 'float64')
    return multivariateNormal(t(r.mean), { covariance: cov })
  }
  if (r.kind === 'categorical' && r.probs !== undefined) {
    const k =
      r.probs.length /
      Math.max(
        1,
        r.shape.reduce((a, b) => a * b, 1),
      )
    return categorical({ probs: fromStorage(Float64Array.from(r.probs), [...r.shape, k], 'float64') })
  }
  const sd = r.variance.map((v) => Math.sqrt(Math.max(v, 1e-300)))
  if (r.kind === 'gamma')
    return gammaDist(
      t(r.mean.map((m, i) => (m * m) / (r.variance[i] ?? 1))),
      t(r.mean.map((m, i) => m / (r.variance[i] ?? 1))),
    )
  if (r.kind === 'beta') {
    const common = r.mean.map((m, i) => (m * (1 - m)) / (r.variance[i] ?? 1) - 1)
    return betaDist(t(r.mean.map((m, i) => m * (common[i] ?? 1))), t(r.mean.map((m, i) => (1 - m) * (common[i] ?? 1))))
  }
  return normal(t(r.mean), t(sd))
}

/** A distribution with the nodes as the last batch axis: the components of a mixture over one batched family (a site of the same shape at every node). */
function stackBatch(records: readonly SiteRecord[], shape: readonly number[]): Distribution {
  const first = records[0]
  if (first === undefined) throw new OmaiError('value', 'ep-grid: no nodes')
  const k = records.length
  const stack = (f: (r: SiteRecord) => Value): Tensor => {
    const columns = records.map((r) => f(r))
    return reshape(stackValues(columns, -1), [...shape, k]) as Tensor
  }
  const m = (r: SiteRecord): Tensor => tensor(r.mean)
  const v = (r: SiteRecord): Value => maximum(tensor(r.variance), 1e-300)
  if (first.kind === 'gamma')
    return gammaDist(
      stack((r) => div(mul(m(r), m(r)), v(r))),
      stack((r) => div(m(r), v(r))),
    )
  if (first.kind === 'beta') {
    const common = (r: SiteRecord): Value => sub(div(mul(m(r), sub(1, m(r))), v(r)), 1)
    return betaDist(
      stack((r) => mul(m(r), common(r))),
      stack((r) => mul(sub(1, m(r)), common(r))),
    )
  }
  return normal(
    stack(m),
    stack((r) => sqrt(v(r))),
  )
}

/** `n` indices drawn by the weights, by inversion of the cumulative sum at uniform points. */
export function chooseNodes(key: Key, weights: readonly number[], n: number): number[] {
  const cum = cumsum(tensor(weights))
  const total = toFlat(cum)[weights.length - 1] ?? 0
  const at = searchsorted(cum, mul(drawUniform(key, [n]), total))
  return toFlat(minimum(at, weights.length - 1))
}

/** What {@link siteMixturePosterior} reads. */
export interface SiteMixture {
  readonly engine: string
  readonly status: RunStatus
  /** The estimate of log p(data) and what it is (`ep-grid`, `ep-multistart`); none when the runs are not weighed by it. */
  readonly logEvidence?: { readonly value: number; readonly method: string }
  readonly diagnostics: Readonly<Record<string, Json>>
  /** The weight of each run (they are normalised here). */
  readonly weights: readonly number[]
  /** What each run believes of each site. */
  readonly runs: readonly Readonly<Record<string, SiteRecord>>[]
  /** Values that vary with the run and are reported as sites of their own, as draws weighted by the runs' weights (the hyperparameters on a grid), by name. */
  readonly values?: Readonly<Record<string, readonly number[]>>
}

/**
 * The posterior that mixes the runs' beliefs of every site with their weights: closed-form moments (the mixture's), draws that
 * pick a run by its weight and draw each site from its belief there, and the per-run values as weighted draws.
 *
 * @param spec - The runs and their weights.
 * @returns The posterior (form `variational`).
 */
export function siteMixturePosterior(spec: SiteMixture): Posterior {
  const total = spec.weights.reduce((a, b) => a + b, 0)
  if (spec.runs.length === 0 || !(total > 0)) throw new OmaiError('value', 'siteMixturePosterior: no run has a weight')
  const w = spec.weights.map((v) => v / total)
  const first = spec.runs[0] as Readonly<Record<string, SiteRecord>>
  const siteNames = Object.keys(first)
  const valueNames = Object.keys(spec.values ?? {})
  const marginals: Record<string, Distribution> = {}
  for (const s of siteNames) {
    const records = spec.runs.map((r) => r[s] as SiteRecord)
    const record = first[s] as SiteRecord
    marginals[s] =
      record.kind === 'categorical'
        ? distributionOf({
            ...record,
            probs: (record.probs ?? []).map((_, i) =>
              records.reduce((a, r, j) => a + (w[j] ?? 0) * (r.probs?.[i] ?? 0), 0),
            ),
          })
        : record.kind === 'multivariateNormal'
          ? mixture({ probs: w }, records.map(distributionOf))
          : mixtureSameFamily({ probs: w }, stackBatch(records, record.shape))
  }
  const draws: Record<string, Tensor> = {}
  for (const v of valueNames) draws[v] = tensor([...(spec.values?.[v] ?? [])])
  const sampler = (key: Key, n: number): Record<string, Tensor> => {
    const picks = chooseNodes(foldIn(key, 0), w, n)
    const byRun = new Map<number, number[]>()
    for (const [at, run] of picks.entries()) byRun.set(run, [...(byRun.get(run) ?? []), at])
    const out: Record<string, Tensor> = {}
    for (const [si, s] of siteNames.entries()) {
      const shape = [...(first[s] as SiteRecord).shape]
      const size = shape.reduce((a, b) => a * b, 1)
      // The draws of each run are stacked run by run, then put back in the order of the picks.
      const parts: Value[] = []
      const positionsInOrder: number[] = []
      for (const [run, positions] of byRun) {
        const d = distributionOf((spec.runs[run] as Readonly<Record<string, SiteRecord>>)[s] as SiteRecord)
        const drawn = d.sample(foldIn(key, 1 + si * 100003 + run), [positions.length])
        parts.push(reshape(drawn, [positions.length, size]))
        positionsInOrder.push(...positions)
      }
      out[s] =
        parts.length === 0
          ? zeros([n, ...shape])
          : (reshape(take(concat(parts, 0), argsort(positionsInOrder), 0), [n, ...shape]) as Tensor)
    }
    for (const v of valueNames) {
      const column = spec.values?.[v] ?? []
      out[v] = tensor(picks.map((i) => column[i] as number))
    }
    return out
  }
  return definePosterior({
    form: 'variational',
    engine: spec.engine,
    sites: [...valueNames, ...siteNames],
    status: spec.status,
    ...(spec.logEvidence === undefined ? {} : { logEvidence: spec.logEvidence }),
    diagnostics: { ...spec.diagnostics, effectiveRuns: 1 / w.reduce((a, v) => a + v * v, 0) },
    marginals,
    draws,
    logWeights: w.map((v) => Math.log(v)),
    sampler,
  })
}
