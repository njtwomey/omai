/**
 * Under-sampling by editing and selection, imbalanced-learn style: the edited nearest neighbours family (a sample is
 * dropped when its neighbours disagree with its class: `editedNearestNeighbours`, repeated and with growing
 * neighbourhoods), the condensing rules that keep a small consistent subset (`condensedNearestNeighbour`,
 * `oneSidedSelection`, `neighbourhoodCleaningRule`), the removal of the samples a model finds hardest
 * (`instanceHardnessThreshold`) and the replacement of a class by its k-means centroids (`clusterCentroids`). They
 * return subsets of the input (centroids aside), with `indices` naming the rows kept, in the order imbalanced-learn
 * gives them (class by class).
 */

import { OmaiError, quantileOf } from '../../base/index.ts'
import { kmeansCodebook } from '../../numerics/spatial/index.ts'
import { choice as drawChoice, foldIn } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { keyFor } from '../_shared.ts'
import type { Estimator } from '../estimator.ts'
import { type CvOption, crossValPredict } from '../model-selection/cross-validation.ts'
import { stratifiedKFold } from '../model-selection/splitters.ts'
import { naiveBayes } from '../naive-bayes.ts'
import { type MatrixView, takeRows } from '../validation.ts'
import {
  assemble,
  type Classes,
  classesOf,
  distancesFrom,
  fromRows,
  modeOf,
  nearest,
  neighbourLists,
  tomekKeep,
} from './_sampling.ts'
import { type Resampler, type SamplingStrategy, samplingTargets } from './resampling.ts'

/**
 * Which classes a cleaning under-sampler edits: `'auto'` (every class but the minority), a named set of classes, or a
 * record whose keys are the labels to edit (the values are not used, as in imbalanced-learn).
 */
export type CleaningStrategy = Exclude<SamplingStrategy, number>

/** Options of the cleaning under-samplers. */
export interface CleaningOptions {
  readonly samplingStrategy?: CleaningStrategy
}

/** The labels a cleaning strategy edits, given the class sizes. */
function cleanedLabels(
  counts: readonly number[],
  classes: readonly number[],
  strategy: CleaningStrategy,
): ReadonlySet<number> {
  if (typeof strategy === 'object') return new Set(classes.filter((label) => strategy[label] !== undefined))
  const targets = samplingTargets(counts, classes, strategy, 'under')
  return new Set(classes.filter((_, i) => targets[i] !== undefined))
}

function minorityOf(counts: readonly number[]): number {
  return counts.indexOf(counts.reduce((a, b) => Math.min(a, b)))
}

/** Row counts by class (aligned with `classes`) among some rows. */
function countsAmong(c: Classes, rows: readonly number[]): number[] {
  const counts = c.classes.map(() => 0)
  for (const r of rows) {
    const k = c.classes.indexOf(c.labels[r] ?? 0)
    counts[k] = (counts[k] ?? 0) + 1
  }
  return counts
}

/**
 * One editing pass (imblearn EditedNearestNeighbours): over the given rows, a row of an edited class stays when all
 * (`'all'`) or most (`'mode'`) of its `k` nearest other rows share its class. Rows come back class by class.
 */
function editPass(
  c: Classes,
  rows: readonly number[],
  clean: ReadonlySet<number>,
  k: number,
  kind: 'all' | 'mode',
): number[] {
  if (rows.length < 2) return [...rows]
  const sub = takeRows(c.m, rows)
  const around = neighbourLists(sub, sub, Math.min(k, rows.length - 1), true)
  const kept: number[] = []
  for (const label of c.classes) {
    rows.forEach((row, p) => {
      if (c.labels[row] !== label) return
      if (!clean.has(label)) {
        kept.push(row)
        return
      }
      const labels = (around[p] ?? []).map((q) => c.labels[rows[q] ?? 0] ?? 0)
      if (kind === 'all' ? labels.every((l) => l === label) : modeOf(labels) === label) kept.push(row)
    })
  }
  return kept
}

/** Options of {@link editedNearestNeighbours}. */
export interface EdgeEditingOptions extends CleaningOptions {
  /** Neighbours examined (default 3). */
  readonly nNeighbors?: number
  /** Keep a sample when all (`'all'`, the default) or most (`'mode'`) of its neighbours share its class. */
  readonly kindSel?: 'all' | 'mode'
}

/**
 * Edited nearest neighbours (Wilson 1972; imblearn EditedNearestNeighbours): drop each sample of an edited class
 * (by default every class but the minority) whose k nearest neighbours do not all (or, with `kindSel: 'mode'`, mostly)
 * share its class, which removes noise and blurred borders.
 *
 * @param options - nNeighbors (3), kindSel and the sampling strategy.
 * @returns The resampler.
 * @concept class-imbalance
 * @index estimator Edited nearest neighbours under-sampling; repeated and all-k variants beside it.
 * @example
 * ```ts
 * import { editedNearestNeighbours } from 'omai-engine/learn/preprocessing'
 * const x = [[0], [0.1], [0.2], [0.3], [0.45], [0.5], [0.6], [1]]
 * const r = editedNearestNeighbours({ nNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 1, 0, 1, 1] })
 * Array.from(r.indices) // => [0, 1, 2, 4, 6, 7]
 * ```
 */
export function editedNearestNeighbours(options: EdgeEditingOptions = {}): Resampler {
  return {
    name: 'edited-nearest-neighbours',
    kind: 'resampler',
    fitResample: (data) => {
      const c = classesOf(data, 'editedNearestNeighbours')
      const counts = c.members.map((m) => m.length)
      const clean = cleanedLabels(counts, c.classes, options.samplingStrategy ?? 'auto')
      const all = Array.from({ length: c.m.n }, (_, i) => i)
      return assemble(c, editPass(c, all, clean, options.nNeighbors ?? 3, options.kindSel ?? 'all'), [], [])
    },
  }
}

/** Whether an editing loop must stop after a pass: a class that is not the original minority has fallen below it, or a class vanished. */
function loopStops(c: Classes, rows: readonly number[], minority: number, allowMinority: boolean): boolean {
  const counts = countsAmong(c, rows)
  const original = c.members.map((m) => m.length)
  const fellBelow = counts.some((n, k) => n > 0 && k !== minority && n < (original[minority] ?? 0))
  return (!allowMinority && fellBelow) || counts.filter((n) => n > 0).length < c.classes.length
}

/** Options of {@link repeatedEditedNearestNeighbours}. */
export interface RepeatedEditingOptions extends EdgeEditingOptions {
  /** The most passes (default 100). */
  readonly maxIter?: number
}

/**
 * Repeated edited nearest neighbours (Tomek 1976; imblearn RepeatedEditedNearestNeighbours): editing passes until no
 * sample is removed, a class other than the minority falls below the minority's original size, a class vanishes, or
 * `maxIter` passes are done.
 *
 * @param options - nNeighbors (3), kindSel, maxIter (100) and the sampling strategy.
 * @returns The resampler.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { repeatedEditedNearestNeighbours } from 'omai-engine/learn/preprocessing'
 * const x = [[0], [0.1], [0.2], [0.3], [0.45], [0.5], [0.6], [1]]
 * const r = repeatedEditedNearestNeighbours({ nNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 1, 0, 1, 1] })
 * Array.from(r.indices).length <= 7 // => true
 * ```
 */
export function repeatedEditedNearestNeighbours(options: RepeatedEditingOptions = {}): Resampler {
  return {
    name: 'repeated-edited-nearest-neighbours',
    kind: 'resampler',
    fitResample: (data) => {
      const c = classesOf(data, 'repeatedEditedNearestNeighbours')
      const counts = c.members.map((m) => m.length)
      const minority = minorityOf(counts)
      let rows = Array.from({ length: c.m.n }, (_, i) => i)
      for (let pass = 0; pass < (options.maxIter ?? 100); pass++) {
        const now = countsAmong(c, rows)
        const clean = cleanedLabels(now, c.classes, options.samplingStrategy ?? 'auto')
        const next = editPass(c, rows, clean, options.nNeighbors ?? 3, options.kindSel ?? 'all')
        const converged = next.length === rows.length
        rows = next
        if (converged || loopStops(c, rows, minority, false)) break
      }
      return assemble(c, rows, [], [])
    },
  }
}

/** Options of {@link allKnn}. */
export interface AllKnnOptions extends EdgeEditingOptions {
  /** Let a class other than the minority fall below the minority's size without stopping (default false). */
  readonly allowMinority?: boolean
}

/**
 * AllKNN (Tomek 1976; imblearn AllKNN): edited nearest neighbours repeated with a growing neighbourhood, 1 neighbour,
 * then 2, up to `nNeighbors`; it stops early when a class other than the minority falls below the minority's size
 * (unless `allowMinority`) or a class vanishes.
 *
 * @param options - nNeighbors (the largest neighbourhood, 3), kindSel, allowMinority and the sampling strategy.
 * @returns The resampler.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { allKnn } from 'omai-engine/learn/preprocessing'
 * const x = [[0], [0.1], [0.2], [0.3], [0.45], [0.5], [0.6], [1]]
 * const r = allKnn({ nNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 1, 0, 1, 1] })
 * Array.from(r.indices).length <= 7 // => true
 * ```
 */
export function allKnn(options: AllKnnOptions = {}): Resampler {
  return {
    name: 'all-knn',
    kind: 'resampler',
    fitResample: (data) => {
      const c = classesOf(data, 'allKnn')
      const counts = c.members.map((m) => m.length)
      const minority = minorityOf(counts)
      let rows = Array.from({ length: c.m.n }, (_, i) => i)
      for (let k = 1; k <= (options.nNeighbors ?? 3); k++) {
        const clean = cleanedLabels(countsAmong(c, rows), c.classes, options.samplingStrategy ?? 'auto')
        rows = editPass(c, rows, clean, k, options.kindSel ?? 'all')
        if (loopStops(c, rows, minority, options.allowMinority ?? false)) break
      }
      return assemble(c, rows, [], [])
    },
  }
}

/** The label a k-NN vote gives a row, from a store of rows (ties to the smaller label). */
function vote(c: Classes, store: readonly number[], row: number, k: number): number {
  const view = takeRows(c.m, store)
  const d = distancesFrom(c.m, row, view)
  const order = Array.from(d.keys()).sort((p, q) => (d[p] ?? 0) - (d[q] ?? 0) || p - q)
  return modeOf(order.slice(0, k).map((s) => c.labels[store[s] ?? 0] ?? 0))
}

/** Options of {@link condensedNearestNeighbour} and {@link oneSidedSelection}. */
export interface CondensingOptions extends CleaningOptions {
  /** Neighbours of the k-NN classifier that decides which samples are kept (default 1). */
  readonly nNeighbors?: number
  /** How many random samples of an edited class start the store (default 1). */
  readonly nSeeds?: number
  /**
   * The starting samples of each edited class by label, as positions among the class's rows in data order (default:
   * `nSeeds` drawn at random without replacement).
   */
  readonly seeds?: Readonly<Record<number, readonly number[]>>
}

function seedPositions(
  options: CondensingOptions,
  label: number,
  size: number,
  draw: () => ArrayLike<number>,
  who: string,
): number[] {
  const given = options.seeds?.[label]
  const positions = given === undefined ? Array.from(draw()) : [...given]
  for (const p of positions) {
    if (!(Number.isInteger(p) && p >= 0 && p < size))
      throw new OmaiError('value', `${who}: seed ${p} outside the class`)
  }
  return positions
}

/**
 * Condensed nearest neighbour (Hart 1968; imblearn CondensedNearestNeighbour): for each edited class (by default every
 * class but the minority) start a store with the minority class and a few random samples of the class, then add each
 * sample of the class the k-NN classifier on the store gets wrong, so that the kept subset classifies the rest. The
 * test for "already stored" follows imbalanced-learn, which compares positions within the class with row indices.
 *
 * @param options - nNeighbors (1), nSeeds (1), seeds and the sampling strategy.
 * @returns The resampler.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { condensedNearestNeighbour } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * const x = [[0], [0.1], [0.2], [0.3], [0.4], [2], [2.1]]
 * const r = condensedNearestNeighbour().fitResample({ x, y: [0, 0, 0, 0, 0, 1, 1] }, { key: key(0) })
 * Array.from(r.indices).length < 7 // => true
 * ```
 */
export function condensedNearestNeighbour(options: CondensingOptions = {}): Resampler {
  const k = options.nNeighbors ?? 1
  return {
    name: 'condensed-nearest-neighbour',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'condensedNearestNeighbour')
      const counts = c.members.map((m) => m.length)
      const minority = c.members[minorityOf(counts)] ?? []
      const clean = cleanedLabels(counts, c.classes, options.samplingStrategy ?? 'auto')
      const key = keyFor(fit, 'condensedNearestNeighbour')
      const rows: number[] = []
      c.members.forEach((members, cls) => {
        const label = c.classes[cls] ?? 0
        if (!clean.has(label)) {
          rows.push(...members)
          return
        }
        const seeds = seedPositions(
          options,
          label,
          members.length,
          () => toFlat(drawChoice(foldIn(key, cls), members.length, options.nSeeds ?? 1, { replace: false })),
          'condensedNearestNeighbour',
        )
        const kept = seeds.map((p) => members[p] ?? 0)
        const store = [...minority, ...kept]
        let good = new Set(kept)
        members.forEach((row, position) => {
          if (good.has(position)) return
          if (vote(c, store, row, k) === label) return
          kept.push(row)
          store.push(row)
          const right = members.flatMap((r, p) => (vote(c, store, r, k) === label ? [p] : []))
          good = new Set([...kept, ...right])
        })
        rows.push(...kept)
      })
      return assemble(c, rows, [], [])
    },
  }
}

/**
 * One-sided selection (Kubat and Matwin 1997; imblearn OneSidedSelection): for each edited class a store of the
 * minority and a random seed sample, plus every sample of the class that the k-NN classifier on the store misclassifies
 * (one pass), then Tomek links are removed from the edited classes.
 *
 * @param options - nNeighbors (1), nSeeds (1), seeds and the sampling strategy.
 * @returns The resampler.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { oneSidedSelection } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * const x = [[0], [0.1], [0.2], [0.3], [0.4], [2], [2.1]]
 * const r = oneSidedSelection().fitResample({ x, y: [0, 0, 0, 0, 0, 1, 1] }, { key: key(0) })
 * Array.from(r.indices).length < 7 // => true
 * ```
 */
export function oneSidedSelection(options: CondensingOptions = {}): Resampler {
  const k = options.nNeighbors ?? 1
  return {
    name: 'one-sided-selection',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'oneSidedSelection')
      const counts = c.members.map((m) => m.length)
      const minority = c.members[minorityOf(counts)] ?? []
      const clean = cleanedLabels(counts, c.classes, options.samplingStrategy ?? 'auto')
      const key = keyFor(fit, 'oneSidedSelection')
      const rows: number[] = []
      c.members.forEach((members, cls) => {
        const label = c.classes[cls] ?? 0
        if (!clean.has(label)) {
          rows.push(...members)
          return
        }
        const seeds = seedPositions(
          options,
          label,
          members.length,
          () => toFlat(drawChoice(foldIn(key, cls), members.length, options.nSeeds ?? 1, { replace: false })),
          'oneSidedSelection',
        )
        const chosen = new Set(seeds)
        const store = [...minority, ...seeds.map((p) => members[p] ?? 0)]
        const wrong = members.filter((row, p) => !chosen.has(p) && vote(c, store, row, k) !== label)
        rows.push(...seeds.map((p) => members[p] ?? 0), ...wrong)
      })
      return assemble(
        c,
        tomekKeep(c.m, rows, c.labels, (label) => clean.has(label)),
        [],
        [],
      )
    },
  }
}

/** Options of {@link neighbourhoodCleaningRule}. */
export interface NeighbourhoodCleaningOptions extends CleaningOptions {
  /** Neighbours examined (default 3). */
  readonly nNeighbors?: number
  /** Edited classes larger than this fraction of the minority class also lose the neighbours of misclassified minority samples (default 0.5). */
  readonly thresholdCleaning?: number
}

/**
 * Neighbourhood cleaning rule (Laurikkala 2001; imblearn NeighbourhoodCleaningRule): remove the samples that edited
 * nearest neighbours (majority vote) removes, and in the edited classes large enough (`thresholdCleaning` times the
 * minority size), the neighbours of every minority sample that its own k nearest samples misclassify.
 *
 * @param options - nNeighbors (3), thresholdCleaning (0.5) and the sampling strategy.
 * @returns The resampler.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { neighbourhoodCleaningRule } from 'omai-engine/learn/preprocessing'
 * const x = [[0], [0.1], [0.2], [0.3], [0.45], [0.5], [0.6], [1]]
 * const r = neighbourhoodCleaningRule({ nNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 1, 0, 1, 1] })
 * Array.from(r.indices).length <= 7 // => true
 * ```
 */
export function neighbourhoodCleaningRule(options: NeighbourhoodCleaningOptions = {}): Resampler {
  const k = options.nNeighbors ?? 3
  return {
    name: 'neighbourhood-cleaning-rule',
    kind: 'resampler',
    fitResample: (data) => {
      const c = classesOf(data, 'neighbourhoodCleaningRule')
      const counts = c.members.map((m) => m.length)
      const minorityIndex = minorityOf(counts)
      const minority = c.members[minorityIndex] ?? []
      const clean = cleanedLabels(counts, c.classes, options.samplingStrategy ?? 'auto')
      const all = Array.from({ length: c.m.n }, (_, i) => i)
      const edited = new Set(editPass(c, all, clean, k, 'mode'))
      const remove = new Set(all.filter((i) => !edited.has(i)))
      const large = c.classes.filter(
        (label, i) =>
          clean.has(label) && (counts[i] ?? 0) > (counts[minorityIndex] ?? 0) * (options.thresholdCleaning ?? 0.5),
      )
      for (const i of minority) {
        // The k-NN vote counts the sample itself; the neighbours to remove do not.
        if (vote(c, all, i, k) === c.labels[i]) continue
        for (const nb of nearest(c.m, i, c.m, k, i)) if (large.includes(c.labels[nb] ?? 0)) remove.add(nb)
      }
      return assemble(
        c,
        all.filter((i) => !remove.has(i)),
        [],
        [],
      )
    },
  }
}

/** Options of {@link instanceHardnessThreshold}. */
export interface InstanceHardnessOptions {
  /** Which classes to cut and to how many (default `'auto'`: every class but the minority, down to its size). */
  readonly samplingStrategy?: SamplingStrategy
  /** A classifier with `predictProba` (default Gaussian naive Bayes; imbalanced-learn uses a random forest). */
  readonly estimator?: Estimator
  /** The cross-validation that gives each sample's out-of-fold probability (default 5 stratified shuffled folds). */
  readonly cv?: CvOption
}

/**
 * Instance hardness threshold (Smith et al. 2014; imblearn InstanceHardnessThreshold): the out-of-fold probability a
 * classifier gives each sample's own class measures how easy it is; each edited class (by default every class but the
 * minority) keeps the most easily classified samples, enough to bring it down to the target size.
 *
 * @param options - estimator, cv and the sampling strategy.
 * @returns The resampler.
 * @throws OmaiError with code `value` when the estimator has no probabilities.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { instanceHardnessThreshold } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 24 }, (_, i) => [i < 18 ? i / 6 : 1 + (i - 18) / 6])
 * const y = x.map((_, i) => (i < 18 ? 0 : 1))
 * const r = instanceHardnessThreshold({ cv: 3 }).fitResample({ x, y }, { key: key(0) })
 * r.y.shape // => [12]
 * ```
 */
export function instanceHardnessThreshold(options: InstanceHardnessOptions = {}): Resampler {
  return {
    name: 'instance-hardness-threshold',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'instanceHardnessThreshold')
      const counts = c.members.map((m) => m.length)
      const targets = samplingTargets(counts, c.classes, options.samplingStrategy ?? 'auto', 'under')
      const key = keyFor(fit, 'instanceHardnessThreshold')
      const estimator = options.estimator ?? naiveBayes()
      const proba = toFlat(
        crossValPredict(
          estimator,
          { x: data.x, y: data.y ?? [] },
          { cv: options.cv ?? stratifiedKFold(5, { shuffle: true }), method: 'predictProba', key },
        ),
      )
      const k = c.classes.length
      const own = (row: number): number => proba[row * k + c.classes.indexOf(c.labels[row] ?? 0)] ?? 0
      const rows = c.members.flatMap((members, cls) => {
        const target = targets[cls]
        if (target === undefined) return members
        const threshold = quantileOf(members.map(own), 1 - target / members.length)
        return members.filter((row) => own(row) >= threshold)
      })
      return assemble(c, rows, [], [])
    },
  }
}

/** Options of {@link clusterCentroids}. */
export interface ClusterCentroidsOptions {
  /** Which classes to replace and by how many centroids (default `'auto'`: every class but the minority, down to its size). */
  readonly samplingStrategy?: SamplingStrategy
  /** `'soft'` replaces a class by the centroids; `'hard'` keeps the real sample nearest each centroid (default `'soft'`). */
  readonly voting?: 'soft' | 'hard'
}

/**
 * Cluster centroids under-sampling (imblearn ClusterCentroids): each edited class (by default every class but the
 * minority) is replaced by the k-means centroids of its samples, as many as the target size (soft voting: the
 * centroids are new points and appear last with index -1), or by the real sample nearest each centroid (hard voting).
 *
 * @param options - voting and the sampling strategy.
 * @returns The resampler.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { clusterCentroids } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * const x = [[0], [0.1], [5], [5.2], [5.1], [0.2], [9]]
 * const r = clusterCentroids().fitResample({ x, y: [0, 0, 0, 0, 0, 0, 1] }, { key: key(0) })
 * r.y.shape // => [2]
 * ```
 */
export function clusterCentroids(options: ClusterCentroidsOptions = {}): Resampler {
  return {
    name: 'cluster-centroids',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'clusterCentroids')
      const counts = c.members.map((m) => m.length)
      const targets = samplingTargets(counts, c.classes, options.samplingStrategy ?? 'auto', 'under')
      const key = keyFor(fit, 'clusterCentroids')
      const rows: number[] = []
      const synthetic: number[][] = []
      const labels: number[] = []
      c.members.forEach((members, cls) => {
        const label = c.classes[cls] ?? 0
        const target = targets[cls]
        if (target === undefined) {
          rows.push(...members)
          return
        }
        const xc: MatrixView = takeRows(c.m, members)
        const centroids = kmeansCodebook(fromRows(xc), target, { key: foldIn(key, cls) })
        const flat = toFlat(centroids.centroids)
        for (let t = 0; t < target; t++) {
          const centre = flat.slice(t * xc.d, (t + 1) * xc.d)
          if (options.voting === 'hard') {
            const d = distancesFrom({ n: 1, d: xc.d, data: Float64Array.from(centre) }, 0, xc)
            rows.push(members[d.indexOf(Math.min(...d))] ?? 0)
          } else {
            synthetic.push(centre)
            labels.push(label)
          }
        }
      })
      return assemble(c, rows, synthetic, labels)
    },
  }
}
