/**
 * Weak supervision with known clean labels: labelling functions that vote or abstain (data programming, as in Snorkel),
 * crowdsourced labels from workers with their own confusion matrices (the Dawid-Skene setting), positive-unlabelled
 * data under SCAR or biased labelling, bags with only their label proportions observed, bags with hidden witness
 * instances for multiple-instance learning, and clusters whose classes follow the cluster assumption of
 * semi-supervised learning, or break it.
 */

import { softmaxOf } from '../base/index.ts'
import {
  type ClassificationData,
  type DatasetGenerator,
  type InteractionData,
  interactions,
  xyClasses,
} from '../data/index.ts'
import { categorical, dirichlet, foldIn, type Key, normal, permutation, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { toFlat, type Value } from '../tensor/index.ts'
import { labels, matrix, stamp } from './_util.ts'

/** Binary or multiclass features: class `c` centred at `separation × e_c` in `d` dimensions, unit noise. */
function gaussianClasses(
  k: Key,
  n: number,
  classes: number,
  d: number,
  separation: number,
): { x: Float64Array; y: number[] } {
  const y = Array.from(categorical(foldIn(k, 0), new Array<number>(classes).fill(1), [n]).data)
  const z = normal(foldIn(k, 1), [n, d]).data
  const x = Float64Array.from(z, (v, i) => v + (i % d === (y[Math.floor(i / d)] ?? 0) % d ? separation : 0))
  return { x, y }
}

/**
 * Programmatic weak supervision (data programming): `m` labelling functions vote on every row or abstain (`-1`), each
 * with its own coverage and accuracy, and wrong votes uniform over the other classes; `duplicates` copies of the first
 * function add correlated votes. The features are the votes; the target is the hidden true class. The truth has the
 * coverages and accuracies and the exact posterior of the true class given a row of votes under conditionally
 * independent functions (Dawid-Skene), the target of label models.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { labellingFunctions } from 'mdml-engine/datasets'
 * generate(labellingFunctions, { knobs: { functions: 4 } }).data.featureNames // => ['lf0', 'lf1', 'lf2', 'lf3']
 * ```
 */
export const labellingFunctions: DatasetGenerator<ClassificationData> = {
  name: 'labellingFunctions',
  kind: 'classification',
  description: 'Votes of labelling functions with known coverages and accuracies, and the hidden labels.',
  knobs: dict({
    n: int(1, 10_000_000, { label: 'samples', description: 'The number of rows.', default: 1000 }),
    classes: int(2, 20, { label: 'classes', description: 'The number of classes.', default: 2 }),
    functions: int(1, 100, {
      label: 'labelling functions',
      description: 'The number of labelling functions.',
      default: 6,
    }),
    minAccuracy: real(0, 1, {
      label: 'lowest accuracy',
      description: 'Accuracies are uniform between the lowest and highest.',
      default: 0.6,
    }),
    maxAccuracy: real(0, 1, {
      label: 'highest accuracy',
      description: 'Accuracies are uniform between the lowest and highest.',
      default: 0.9,
    }),
    minCoverage: real(0, 1, {
      label: 'lowest coverage',
      description: 'Coverages (chances of voting) are uniform between the lowest and highest.',
      default: 0.2,
    }),
    maxCoverage: real(0, 1, {
      label: 'highest coverage',
      description: 'Coverages (chances of voting) are uniform between the lowest and highest.',
      default: 0.8,
    }),
    duplicates: int(0, 20, {
      description: 'Copies of the first function that agree with it, breaking conditional independence.',
      default: 0,
      label: 'correlated copies of the first function',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const kc = knobs['classes'] as number
    const m = knobs['functions'] as number
    const dup = knobs['duplicates'] as number
    const ua = uniform(foldIn(k, 0), [m]).data
    const uc = uniform(foldIn(k, 1), [m]).data
    const lerp = (lo: number, hi: number, t: number) => lo + (hi - lo) * t
    const accuracy = Array.from(ua, (t) => lerp(knobs['minAccuracy'] as number, knobs['maxAccuracy'] as number, t))
    const coverage = Array.from(uc, (t) => lerp(knobs['minCoverage'] as number, knobs['maxCoverage'] as number, t))
    const y = Array.from(categorical(foldIn(k, 2), new Array<number>(kc).fill(1), [n]).data)
    const width = m + dup
    const votes = new Float64Array(n * width)
    for (let j = 0; j < m; j++) {
      const u = uniform(foldIn(foldIn(k, 3), j), [n, 3]).data
      for (let i = 0; i < n; i++) {
        let v = -1
        if ((u[3 * i] ?? 1) < (coverage[j] ?? 0)) {
          const truth = y[i] ?? 0
          if ((u[3 * i + 1] ?? 1) < (accuracy[j] ?? 0)) v = truth
          else {
            const other = Math.floor((u[3 * i + 2] ?? 0) * (kc - 1))
            v = other >= truth ? other + 1 : other
          }
        }
        votes[i * width + j] = v
      }
    }
    for (let c = 0; c < dup; c++) for (let i = 0; i < n; i++) votes[i * width + m + c] = votes[i * width] ?? -1
    const posterior = (row: readonly number[]): number[] => {
      const lp = Array.from({ length: kc }, (_, c) => {
        let s = Math.log(1 / kc)
        for (let j = 0; j < m; j++) {
          const v = row[j] ?? -1
          if (v < 0) continue
          s += Math.log(v === c ? (accuracy[j] ?? 0) : (1 - (accuracy[j] ?? 0)) / (kc - 1))
        }
        return s
      })
      return softmaxOf(lp)
    }
    return {
      data: xyClasses(matrix(votes, n, width), labels(y), {
        ...stamp('labellingFunctions', labellingFunctions.description, knobs),
        featureNames: Array.from({ length: width }, (_, j) => `lf${j}`),
      }),
      truth: {
        extra: { accuracy, coverage, posterior, abstain: -1, duplicateOf: Array.from({ length: dup }, () => 0) },
      },
    }
  },
}

/**
 * Crowdsourced labels (the Dawid-Skene setting): items with hidden classes, workers with their own confusion
 * matrices (rows drawn from Dirichlets concentrated on the diagonal by `skill`), and each item labelled by `perItem`
 * random workers. Interactions are (worker, item, given label). The truth has every item's class, every worker's
 * confusion matrix and the exact posterior of each item's class given its labels.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { crowdLabels } from 'mdml-engine/datasets'
 * generate(crowdLabels, { knobs: { items: 10, perItem: 3 } }).data.values.shape // => [30]
 * ```
 */
export const crowdLabels: DatasetGenerator<InteractionData> = {
  name: 'crowdLabels',
  kind: 'interactions',
  description: 'Labels given by workers with known confusion matrices to items with hidden classes.',
  knobs: dict({
    items: int(1, 1_000_000, { label: 'items', description: 'The number of items to label.', default: 500 }),
    workers: int(1, 10_000, { label: 'workers', description: 'The number of workers.', default: 20 }),
    perItem: int(1, 100, { description: 'How many workers label each item.', default: 5, label: 'workers per item' }),
    classes: int(2, 20, { label: 'classes', description: 'The number of classes.', default: 3 }),
    skill: real(0, 100, {
      description:
        'The Dirichlet concentration on the diagonal of each worker’s confusion matrix: higher is more accurate.',
      default: 4,
      label: 'diagonal concentration of confusion matrices',
    }),
    spammers: real(0, 1, {
      description: 'The share of workers who answer uniformly at random whatever the item.',
      default: 0,
      label: 'share of spammers',
    }),
    adversaries: real(0, 1, {
      description: 'The share of workers who answer the class after the true one (cyclically) with their skill.',
      default: 0,
      label: 'share of adversaries',
    }),
  }),
  generate: (k, knobs) => {
    const ni = knobs['items'] as number
    const nw = knobs['workers'] as number
    const per = Math.min(knobs['perItem'] as number, nw)
    const kc = knobs['classes'] as number
    const skill = knobs['skill'] as number
    // The first workers are spammers, then adversaries, the rest honest: the kind is part of the truth.
    const nSpam = Math.round((knobs['spammers'] as number) * nw)
    const nAdv = Math.min(nw - nSpam, Math.round((knobs['adversaries'] as number) * nw))
    const workerKinds = Array.from({ length: nw }, (_, w) =>
      w < nSpam ? 'spammer' : w < nSpam + nAdv ? 'adversary' : 'honest',
    )
    const confusion = Array.from({ length: nw }, (_, w) =>
      Array.from({ length: kc }, (_, c) => {
        const kind = workerKinds[w]
        if (kind === 'spammer') return new Array<number>(kc).fill(1 / kc)
        const peak = kind === 'adversary' ? (c + 1) % kc : c
        return Array.from(
          dirichlet(
            foldIn(foldIn(k, 0), w * kc + c),
            Array.from({ length: kc }, (_, j) => (j === peak ? 1 + skill : 1)),
            1,
          ).data,
        )
      }),
    )
    const truthLabels = Array.from(categorical(foldIn(k, 1), new Array<number>(kc).fill(1), [ni]).data)
    const users: number[] = []
    const items: number[] = []
    const values: number[] = []
    for (let i = 0; i < ni; i++) {
      const who = permutation(foldIn(foldIn(k, 2), i), nw).data.slice(0, per)
      const u = uniform(foldIn(foldIn(k, 3), i), [per]).data
      for (const [j, w] of who.entries()) {
        const row = confusion[w]?.[truthLabels[i] ?? 0] ?? []
        let acc = 0
        let given = row.length - 1
        for (const [c, p] of row.entries()) {
          acc += p
          if ((u[j] ?? 1) < acc) {
            given = c
            break
          }
        }
        users.push(w)
        items.push(i)
        values.push(given)
      }
    }
    const posterior = (answers: readonly { readonly worker: number; readonly label: number }[]): number[] => {
      const lp = Array.from({ length: kc }, (_, c) =>
        answers.reduce((s, a) => s + Math.log(confusion[a.worker]?.[c]?.[a.label] ?? 1e-300), Math.log(1 / kc)),
      )
      return softmaxOf(lp)
    }
    return {
      data: interactions(
        { users, items, values, nUsers: nw, nItems: ni },
        stamp('crowdLabels', crowdLabels.description, knobs),
      ),
      truth: {
        clusters: labels(truthLabels),
        latents: { trueLabels: truthLabels },
        extra: { confusion, posterior, workerKinds },
      },
    }
  },
}

/**
 * Positive-unlabelled data: two Gaussian classes, of which a positive is labelled with probability `labelFrequency`
 * completely at random (SCAR) or with a probability that rises with its first feature (SAR); unlabelled rows include
 * every negative. The target is the labelled indicator `s`; the truth has the true classes, the label frequency `c` and
 * the labelling propensity, so `p(y = 1 | x) = p(s = 1 | x) / c` can be checked under SCAR.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { positiveUnlabelled } from 'mdml-engine/datasets'
 * generate(positiveUnlabelled).data.classes // => ['unlabelled', 'labelled']
 * ```
 */
export const positiveUnlabelled: DatasetGenerator<ClassificationData> = {
  name: 'positiveUnlabelled',
  kind: 'classification',
  description: 'Positive and unlabelled data: some positives labelled, everything else unlabelled.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of rows.', default: 2000 }),
    features: int(1, 100, { label: 'features', description: 'The number of feature columns.', default: 2 }),
    prior: real(0.01, 0.99, {
      description: 'The fraction of rows that are positive.',
      default: 0.4,
      label: 'fraction of positives',
    }),
    labelFrequency: real(0.01, 1, {
      description: 'The probability that a positive is labelled.',
      default: 0.5,
      label: 'c = P(labelled | positive)',
    }),
    mechanism: choice(['SCAR', 'SAR'], {
      label: 'labelling mechanism',
      description: 'Labelled completely at random (SCAR) or depending on the features (SAR).',
      default: 'SCAR',
    }),
    separation: real(0, 10, {
      label: 'class separation',
      description: 'The distance between the class means, in standard deviations.',
      default: 2,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['features'] as number
    const prior = knobs['prior'] as number
    const c = knobs['labelFrequency'] as number
    const sep = knobs['separation'] as number
    const u = uniform(foldIn(k, 0), [n, 2]).data
    const y = Array.from({ length: n }, (_, i) => ((u[2 * i] ?? 1) < prior ? 1 : 0))
    const z = normal(foldIn(k, 1), [n, d]).data
    const x = Float64Array.from(z, (v, i) => v + ((y[Math.floor(i / d)] ?? 0) === 1 ? sep / Math.sqrt(d) : 0))
    const sar = knobs['mechanism'] === 'SAR'
    // SAR: a logistic in the first feature calibrated so the mean labelling probability over positives is about c.
    const propensity = (row: ArrayLike<number>) =>
      sar ? Math.min(1, (2 * c) / (1 + Math.exp(-((row[0] ?? 0) - sep / Math.sqrt(d))))) : c
    const s = y.map((yi, i) => (yi === 1 && (u[2 * i + 1] ?? 1) < propensity(x.subarray(i * d, (i + 1) * d)) ? 1 : 0))
    return {
      data: xyClasses(matrix(x, n, d), labels(s), {
        ...stamp('positiveUnlabelled', positiveUnlabelled.description, knobs),
        classes: ['unlabelled', 'labelled'],
      }),
      truth: {
        latents: { trueLabels: labels(y) },
        extra: { labelFrequency: c, classPrior: prior, propensity, mechanism: sar ? 'SAR' : 'SCAR' },
      },
    }
  },
}

/**
 * Learning from label proportions: Gaussian-class rows grouped into bags (at random, or by sorting on the first
 * feature so bags are more homogeneous), where only each bag's class proportions are observed. The labels in the
 * dataset are hidden (`-1`); the truth has the true labels, every row's bag and every bag's proportions.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { labelProportionBags } from 'mdml-engine/datasets'
 * (generate(labelProportionBags, { knobs: { n: 100, bagSize: 10 } }).truth.extra?.['proportions'] as number[][]).length // => 10
 * ```
 */
export const labelProportionBags: DatasetGenerator<ClassificationData> = {
  name: 'labelProportionBags',
  kind: 'classification',
  description: 'Rows in bags with only the class proportions of each bag observed.',
  knobs: dict({
    n: int(1, 10_000_000, { label: 'samples', description: 'The number of rows.', default: 1000 }),
    classes: int(2, 20, { label: 'classes', description: 'The number of classes.', default: 2 }),
    features: int(1, 100, { label: 'features', description: 'The number of feature columns.', default: 2 }),
    bagSize: int(1, 10_000, { label: 'bag size', description: 'Rows per bag.', default: 20 }),
    grouping: choice(['random', 'sorted'], {
      label: 'grouping',
      description: 'Bags of random rows, or of rows sorted by a feature (bags with skewed proportions).',
      default: 'random',
    }),
    separation: real(0, 10, {
      label: 'class separation',
      description: 'The distance between the class means, in standard deviations.',
      default: 2,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const kc = knobs['classes'] as number
    const d = knobs['features'] as number
    const size = knobs['bagSize'] as number
    const { x, y } = gaussianClasses(foldIn(k, 0), n, kc, d, knobs['separation'] as number)
    const order =
      knobs['grouping'] === 'sorted'
        ? Array.from({ length: n }, (_, i) => i).sort((a, b) => (x[a * d] ?? 0) - (x[b * d] ?? 0))
        : Array.from(permutation(foldIn(k, 1), n).data)
    const bag = new Array<number>(n).fill(0)
    for (const [p, i] of order.entries()) bag[i] = Math.floor(p / size)
    const nBags = Math.ceil(n / size)
    const proportions = Array.from({ length: nBags }, () => new Array<number>(kc).fill(0))
    const sizes = new Array<number>(nBags).fill(0)
    for (let i = 0; i < n; i++) {
      const b = bag[i] ?? 0
      const row = proportions[b] as number[]
      row[y[i] ?? 0] = (row[y[i] ?? 0] ?? 0) + 1
      sizes[b] = (sizes[b] ?? 0) + 1
    }
    for (const [b, row] of proportions.entries()) for (let c = 0; c < kc; c++) row[c] = (row[c] ?? 0) / (sizes[b] ?? 1)
    return {
      data: xyClasses(matrix(x, n, d), labels(new Array<number>(n).fill(-1)), {
        ...stamp('labelProportionBags', labelProportionBags.description, knobs),
        classes: Array.from({ length: kc }, (_, c) => String(c)),
      }),
      truth: { latents: { trueLabels: labels(y) }, extra: { bags: bag, proportions, bagSizes: sizes } },
    }
  },
}

/**
 * Multiple-instance bags: rows grouped into bags of instances, where a bag is positive when it holds at least one
 * witness (a positive instance) and negative when it holds none. The dataset's target is the bag's label, repeated for
 * each of its instances; the truth has every instance's own label, the bag of every row and the bag labels.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { multipleInstanceBags } from 'mdml-engine/datasets'
 * generate(multipleInstanceBags, { knobs: { bags: 6, bagSize: 4 } }).data.x.shape // => [24, 2]
 * ```
 */
export const multipleInstanceBags: DatasetGenerator<ClassificationData> = {
  name: 'multipleInstanceBags',
  kind: 'classification',
  description: 'Bags of instances, positive when they hold a witness; only the bag label is observed.',
  knobs: dict({
    bags: int(2, 1_000_000, { label: 'bags', description: 'The number of bags.', default: 100 }),
    bagSize: int(1, 1000, { label: 'bag size', description: 'Instances per bag.', default: 5 }),
    features: int(1, 100, { label: 'features', description: 'The number of feature columns.', default: 2 }),
    positiveShare: real(0, 1, {
      label: 'positive bags',
      description: 'The share of bags that are positive.',
      default: 0.5,
    }),
    witnessRate: real(0, 1, {
      label: 'witness rate',
      description: 'In a positive bag, the chance that an instance is a witness (one is always forced).',
      default: 0.3,
    }),
    separation: real(0, 10, {
      label: 'separation',
      description: 'The distance between witnesses and other instances, in standard deviations.',
      default: 3,
    }),
  }),
  generate: (k, knobs) => {
    const nb = knobs['bags'] as number
    const size = knobs['bagSize'] as number
    const d = knobs['features'] as number
    const n = nb * size
    const rate = knobs['witnessRate'] as number
    const sep = knobs['separation'] as number
    const ub = uniform(foldIn(k, 0), [nb]).data
    const ui = uniform(foldIn(k, 1), [n]).data
    const forced = uniform(foldIn(k, 2), [nb]).data
    const bagLabel = Array.from({ length: nb }, (_, b) => ((ub[b] ?? 1) < (knobs['positiveShare'] as number) ? 1 : 0))
    const instance = new Array<number>(n).fill(0)
    for (let b = 0; b < nb; b++) {
      if (bagLabel[b] !== 1) continue
      const witness = Math.floor((forced[b] ?? 0) * size)
      for (let i = 0; i < size; i++) instance[b * size + i] = i === witness || (ui[b * size + i] ?? 1) < rate ? 1 : 0
    }
    const z = normal(foldIn(k, 3), [n, d]).data
    const x = Float64Array.from(z, (v, i) => v + ((instance[Math.floor(i / d)] ?? 0) === 1 ? sep / Math.sqrt(d) : 0))
    const bag = Array.from({ length: n }, (_, i) => Math.floor(i / size))
    return {
      data: xyClasses(matrix(x, n, d), labels(bag.map((b) => bagLabel[b] ?? 0)), {
        ...stamp('multipleInstanceBags', multipleInstanceBags.description, knobs),
        classes: ['negative', 'positive'],
      }),
      truth: {
        latents: { instanceLabels: labels(instance) },
        extra: { bags: bag, bagLabels: bagLabel, bagSize: size },
      },
    }
  },
}

/**
 * Clusters and classes for semi-supervised learning: equal isotropic Gaussian clusters on a ring, each assigned a class,
 * and each point's label follows its cluster's class with probability `purity` and is uniform over the classes
 * otherwise. At purity 1 the cluster assumption holds exactly (a classifier can propagate labels along clusters); lower
 * purity breaks it by a known amount. The truth has the clusters, the cluster-to-class map and the exact posterior.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { clusterAssumption } from 'mdml-engine/datasets'
 * generate(clusterAssumption, { knobs: { n: 20 } }).data.x.shape // => [20, 2]
 * ```
 */
export const clusterAssumption: DatasetGenerator<ClassificationData> = {
  name: 'clusterAssumption',
  kind: 'classification',
  description: 'Gaussian clusters on a ring with classes that follow the clusters, up to a planted breakage.',
  knobs: dict({
    n: int(1, 10_000_000, { label: 'samples', description: 'The number of rows.', default: 600 }),
    clusters: int(2, 50, { label: 'clusters', description: 'The number of clusters.', default: 6 }),
    classes: int(2, 20, {
      label: 'classes',
      description: 'The number of classes; cluster j has class j mod classes.',
      default: 2,
    }),
    spread: real(0.05, 5, {
      label: 'cluster spread',
      description: 'The standard deviation of each cluster, with the ring of radius 3.',
      default: 0.5,
    }),
    purity: real(0, 1, {
      label: 'purity',
      description: 'The chance that a point carries its cluster’s class; otherwise its class is uniform.',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const nc = knobs['clusters'] as number
    const kc = knobs['classes'] as number
    const sd = knobs['spread'] as number
    const purity = knobs['purity'] as number
    const centres = Array.from({ length: nc }, (_, j) => [
      3 * Math.cos((2 * Math.PI * j) / nc),
      3 * Math.sin((2 * Math.PI * j) / nc),
    ])
    const cluster = Array.from(categorical(foldIn(k, 0), new Array<number>(nc).fill(1), [n]).data)
    const z = normal(foldIn(k, 1), [n, 2]).data
    const x = Float64Array.from(
      { length: n * 2 },
      (_, i) => (centres[cluster[Math.floor(i / 2)] ?? 0]?.[i % 2] ?? 0) + sd * (z[i] ?? 0),
    )
    const u = uniform(foldIn(k, 2), [n, 2]).data
    const y = cluster.map((c, i) =>
      (u[2 * i] ?? 1) < purity ? c % kc : Math.min(kc - 1, Math.floor((u[2 * i + 1] ?? 0) * kc)),
    )
    const posterior = (row: readonly number[]): number[] => {
      const r = softmaxOf(
        centres.map((c) => -(((row[0] ?? 0) - (c[0] ?? 0)) ** 2 + ((row[1] ?? 0) - (c[1] ?? 0)) ** 2) / (2 * sd * sd)),
      )
      const p = new Array<number>(kc).fill((1 - purity) / kc)
      r.forEach((w, j) => {
        p[j % kc] = (p[j % kc] ?? 0) + w * purity
      })
      return p
    }
    return {
      data: xyClasses(matrix(x, n, 2), labels(y), {
        ...stamp('clusterAssumption', clusterAssumption.description, knobs),
        classes: Array.from({ length: kc }, (_, c) => String(c)),
      }),
      truth: {
        clusters: labels(cluster),
        posterior: (v: Value): Value => {
          const flat = toFlat(v)
          const rows = Math.floor(flat.length / 2)
          const out = new Float64Array(rows * kc)
          for (let i = 0; i < rows; i++) out.set(posterior([flat[2 * i] ?? 0, flat[2 * i + 1] ?? 0]), i * kc)
          return matrix(out, rows, kc)
        },
        extra: { clusterClass: Array.from({ length: nc }, (_, j) => j % kc), centres, purity, posterior },
      },
    }
  },
}
