import {
  type ClassificationData,
  type DatasetGenerator,
  type MultilabelData,
  type Truth,
  take,
  xyClasses,
  xyMultilabel,
} from '../../lib/data/index.ts'
import { inv, slogdet } from '../../lib/numerics/linalg/index.ts'
import { foldIn, type Key, normal, permutation, poisson, randint, uniform } from '../../lib/random/index.ts'
import { bool, dict, int, real } from '../../lib/space/index.ts'
import {
  add,
  concat,
  div,
  fromStorage,
  item,
  matmul,
  mul,
  sliceAxis,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  take as takeColumns,
  tensor,
  type Value,
} from '../../lib/tensor/index.ts'
import { mixtureTruth, monteCarloBayesError, withLazyBayesError } from './_mixtures.ts'
import { labels, matrix, stamp } from './_util.ts'

/** Distinct random vertices of the unit hypercube `{0, 1}^d`. */
function hypercubeVertices(k: Key, count: number, d: number): number[][] {
  const seen = new Set<string>()
  const out: number[][] = []
  for (let attempt = 0; out.length < count; attempt++) {
    const bits = Array.from(randint(foldIn(k, attempt), [d], 0, 2).data)
    const key = bits.join('')
    if (seen.has(key) && attempt < 10_000 * count) continue
    seen.add(key)
    out.push(bits)
  }
  return out
}

/** Per-cluster row counts as in sklearn: `floor(n w_class / clustersPerClass)`, remainder dealt round-robin. */
function clusterCounts(n: number, weights: readonly number[], perClass: number): number[] {
  const nClusters = weights.length * perClass
  const counts = Array.from({ length: nClusters }, (_, c) =>
    Math.floor((n * (weights[c % weights.length] ?? 0)) / perClass),
  )
  let left = n - counts.reduce((a, b) => a + b, 0)
  for (let i = 0; left > 0; i++, left--) counts[i % nClusters] = (counts[i % nClusters] ?? 0) + 1
  return counts
}

/**
 * A random classification problem (sklearn.datasets.make_classification): Gaussian clusters at vertices of a
 * hypercube of side `2 classSep` in the informative subspace, each with a random linear covariance, then redundant
 * features (random linear combinations), repeated features, and noise features, with class imbalance and label
 * flipping. The truth is exact: the class posterior of the mixture in the informative coordinates (with flips
 * mixed in), its log density, a Monte Carlo Bayes error, and where each kind of feature ended up after the column
 * shuffle (`truth.extra.informative`, `redundant`, `repeated`, `noise`).
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { makeClassification } from 'whitebox-engine/methods/datasets'
 * generate(makeClassification, { knobs: { nFeatures: 6, nInformative: 3, nRedundant: 1 } }).truth.relevantFeatures?.length // => 4
 * ```
 */
export const makeClassification: DatasetGenerator<ClassificationData> = {
  name: 'makeClassification',
  kind: 'classification',
  description:
    'Gaussian clusters on hypercube vertices with informative, redundant, repeated and noise features (sklearn make_classification).',
  knobs: dict({
    n: int(2, 100_000, { default: 100, label: 'samples' }),
    nFeatures: int(1, 1000, { default: 20, label: 'features' }),
    nInformative: int(1, 100, { default: 2, label: 'informative features' }),
    nRedundant: int(0, 100, { default: 2, label: 'redundant features' }),
    nRepeated: int(0, 100, { default: 0, label: 'repeated features' }),
    nClasses: int(2, 50, { default: 2, label: 'classes' }),
    nClustersPerClass: int(1, 20, { default: 2, label: 'clusters per class' }),
    classSep: real(0, 20, { default: 1, label: 'class separation' }),
    imbalance: real(1, 1000, { default: 1, log: true, label: 'largest to smallest class weight' }),
    flipY: real(0, 1, { default: 0.01, label: 'label flip probability' }),
    hypercube: bool({ default: true, label: 'centres on hypercube vertices' }),
    shift: real(-100, 100, { default: 0 }),
    scale: real(0.001, 100, { default: 1 }),
    shuffle: bool({ default: true }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const nInf = knobs['nInformative'] as number
    const nRed = knobs['nRedundant'] as number
    const nRep = knobs['nRepeated'] as number
    const d = Math.max(knobs['nFeatures'] as number, nInf + nRed + nRep)
    const nUseless = d - nInf - nRed - nRep
    const nClasses = knobs['nClasses'] as number
    const perClass = knobs['nClustersPerClass'] as number
    const sep = knobs['classSep'] as number
    const flipY = knobs['flipY'] as number
    const shift = knobs['shift'] as number
    const scale = knobs['scale'] as number
    const ratio = knobs['imbalance'] as number
    const raw = Array.from({ length: nClasses }, (_, c) => ratio ** (-c / Math.max(1, nClasses - 1)))
    const total = raw.reduce((a, b) => a + b, 0)
    const weights = raw.map((w) => w / total)
    const nClusters = nClasses * perClass
    const counts = clusterCounts(n, weights, perClass)
    let centroids = hypercubeVertices(foldIn(k, 0), nClusters, nInf).map((v) => v.map((b) => 2 * sep * b - sep))
    if (knobs['hypercube'] === false) {
      const a = uniform(foldIn(k, 1), [nClusters]).data
      const b = uniform(foldIn(k, 2), [nInf]).data
      centroids = centroids.map((v, c) => v.map((x, j) => x * (a[c] ?? 1) * (b[j] ?? 1)))
    }
    // Cluster transforms A_c (rows z A_c), as in sklearn: entries uniform on [-1, 1].
    const transforms = Array.from({ length: nClusters }, (_, c) =>
      matrix(uniform(foldIn(k, 10 + c), [nInf, nInf], { low: -1, high: 1 }).data, nInf, nInf),
    )
    const zInf = normal(foldIn(k, 3), [n, nInf])
    const xInf = new Float64Array(n * nInf)
    const y: number[] = []
    let start = 0
    for (const [c, m] of counts.entries()) {
      const block = matmul(sliceAxis(zInf, 0, start, start + m), transforms[c] as Tensor) as Tensor
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < nInf; j++)
          xInf[(start + i) * nInf + j] = (block.data[i * nInf + j] ?? 0) + (centroids[c]?.[j] ?? 0)
        y.push(c % nClasses)
      }
      start += m
    }
    const inf = matrix(xInf, n, nInf)
    const parts: Value[] = [inf]
    if (nRed > 0)
      parts.push(matmul(inf, matrix(uniform(foldIn(k, 4), [nInf, nRed], { low: -1, high: 1 }).data, nInf, nRed)))
    let xs = concat(parts, 1) as Tensor
    const repeatedFrom = Array.from(uniform(foldIn(k, 5), [nRep]).data, (u) => Math.floor((nInf + nRed - 1) * u + 0.5))
    if (nRep > 0) xs = concat([xs, takeColumns(xs, tensor(repeatedFrom, { dtype: 'int32' }), 1)], 1) as Tensor
    if (nUseless > 0) xs = concat([xs, normal(foldIn(k, 6), [n, nUseless])], 1) as Tensor
    // Label noise: a flipped label is redrawn uniformly (it may land on the same class).
    const flip = uniform(foldIn(k, 7), [n]).data
    const redraw = randint(foldIn(k, 8), [n], 0, nClasses).data
    const clean = [...y]
    const noisy = y.map((c, i) => ((flip[i] ?? 1) < flipY ? (redraw[i] ?? c) : c))
    const xScaled = mul(add(xs, shift), scale) as Tensor
    // Column shuffle, tracked so the truth knows where every feature went.
    const colOrder =
      knobs['shuffle'] === false
        ? Array.from({ length: d }, (_, j) => j)
        : Array.from(permutation(foldIn(k, 9), d).data)
    const position = new Array<number>(d)
    for (const [newJ, oldJ] of colOrder.entries()) position[oldJ] = newJ
    const x = takeColumns(xScaled, tensor(colOrder, { dtype: 'int32' }), 1) as Tensor
    const place = (from: number, count: number) => Array.from({ length: count }, (_, j) => position[from + j] ?? 0)
    const informative = place(0, nInf)
    // Exact posterior: x_inf = z A_c + mu_c, so log N = log N(z; 0, I) - log|det A_c| with z = (x_inf - mu_c) A_c^-1.
    const invs = transforms.map((a) => inv(a))
    const logDets = transforms.map((a) => item(slogdet(a).logabsdet))
    const toInformative = (v: Value): Value =>
      sub(div(takeColumns(v, tensor(informative, { dtype: 'int32' }), 1), scale), shift)
    const logDensities = (v: Value): Value => {
      const xi = toInformative(v)
      const cols = centroids.map((mu, c) => {
        const z = matmul(sub(xi, tensor(mu)), invs[c] as Value)
        return add(mul(sum(square(z), 1), -0.5), -0.5 * nInf * Math.log(2 * Math.PI) - (logDets[c] ?? 0))
      })
      return stack(cols, 1)
    }
    const model = {
      logDensities,
      classOf: centroids.map((_, c) => c % nClasses),
      logWeights: counts.map((m) => Math.log(Math.max(m, 1e-300) / n)),
      nClasses,
    }
    const exact = mixtureTruth(model)
    const cleanPosterior = exact.posterior
    const posterior = (v: Value): Value => add(mul(cleanPosterior(v), 1 - flipY), flipY / nClasses)
    const draw = (key: Key, m: number): Value => {
      const cluster = Array.from(uniform(foldIn(key, 0), [m]).data, (u) => {
        let acc = 0
        for (const [c, cnt] of counts.entries()) {
          acc += cnt / n
          if (u < acc) return c
        }
        return nClusters - 1
      })
      const z = normal(foldIn(key, 1), [m, nInf]).data
      const out = new Float64Array(m * d)
      for (let i = 0; i < m; i++) {
        const c = cluster[i] ?? 0
        const a = transforms[c] as Tensor
        for (let j = 0; j < nInf; j++) {
          let s = centroids[c]?.[j] ?? 0
          for (let l = 0; l < nInf; l++) s += (z[i * nInf + l] ?? 0) * (a.data[l * nInf + j] ?? 0)
          out[i * d + (informative[j] ?? 0)] = (s + shift) * scale
        }
      }
      return matrix(out, m, d)
    }
    const truth: Truth = {
      logProb: exact.logProb,
      posterior,
      clusters: labels(Array.from({ length: n }, (_, i) => clusterOf(counts, i))),
      relevantFeatures: place(0, nInf + nRed + nRep).sort((a, b) => a - b),
      latents: { cleanLabels: labels(clean) },
      extra: {
        informative,
        redundant: place(nInf, nRed),
        repeated: place(nInf + nRed, nRep),
        noise: place(nInf + nRed + nRep, nUseless),
        centroids,
        priors: weights,
      },
    }
    const data = xyClasses(x, labels(noisy), stamp('makeClassification', makeClassification.description, knobs))
    const bayes = () => monteCarloBayesError(posterior, draw, foldIn(k, 20), centroids.length)
    if (knobs['shuffle'] === false) return { data, truth: withLazyBayesError(truth, bayes) }
    const order = permutation(foldIn(k, 30), n)
    const reorder = (t: Tensor<'int32'>) => labels(Array.from(order.data, (i) => t.data[i] ?? 0))
    return {
      data: take(data, order),
      truth: withLazyBayesError(
        {
          ...truth,
          clusters: reorder(truth.clusters as Tensor<'int32'>),
          latents: { cleanLabels: reorder(labels(clean)) },
        },
        bayes,
      ),
    }
  },
}

function clusterOf(counts: readonly number[], i: number): number {
  let acc = 0
  for (const [c, m] of counts.entries()) {
    acc += m
    if (i < acc) return c
  }
  return counts.length - 1
}

/**
 * A random multilabel problem (sklearn.datasets.make_multilabel_classification): each row draws a number of labels
 * from a Poisson, the labels from class probabilities, a document length from a Poisson, and word counts from the
 * mixture of its labels' word distributions. The truth holds the class and word probabilities.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { makeMultilabelClassification } from 'whitebox-engine/methods/datasets'
 * generate(makeMultilabelClassification, { knobs: { n: 20, nClasses: 4 } }).data.y.shape // => [20, 4]
 * ```
 */
export const makeMultilabelClassification: DatasetGenerator<MultilabelData> = {
  name: 'makeMultilabelClassification',
  kind: 'multilabel',
  description:
    'Bag-of-words rows with labels drawn from a Poisson-multinomial model (sklearn make_multilabel_classification).',
  knobs: dict({
    n: int(1, 100_000, { default: 100, label: 'samples' }),
    nFeatures: int(1, 10_000, { default: 20, label: 'features (vocabulary)' }),
    nClasses: int(1, 100, { default: 5, label: 'classes' }),
    nLabels: real(0, 100, { default: 2, label: 'mean labels per row' }),
    length: real(1, 10_000, { default: 50, label: 'mean document length' }),
    allowUnlabeled: bool({ default: true }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['nFeatures'] as number
    const nc = knobs['nClasses'] as number
    const meanLabels = knobs['nLabels'] as number
    const length = knobs['length'] as number
    const allowUnlabeled = knobs['allowUnlabeled'] !== false
    const pc = Array.from(uniform(foldIn(k, 0), [nc]).data)
    const pcSum = pc.reduce((a, b) => a + b, 0)
    const classProbs = pc.map((v) => v / pcSum)
    const pw = uniform(foldIn(k, 1), [d, nc]).data
    const wordProbs = Array.from({ length: nc }, (_, c) => {
      let s = 0
      for (let w = 0; w < d; w++) s += pw[w * nc + c] ?? 0
      return Array.from({ length: d }, (_, w) => (pw[w * nc + c] ?? 0) / s)
    })
    const pick = (u: number, probs: readonly number[]) => {
      let acc = 0
      for (const [i, p] of probs.entries()) {
        acc += p
        if (u < acc) return i
      }
      return probs.length - 1
    }
    const x = new Float64Array(n * d)
    const y = new Int32Array(n * nc)
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, 100 + i)
      let size = nc + 1
      for (let t = 0; size > nc || (!allowUnlabeled && size === 0); t++) {
        size = poisson(foldIn(ki, t), meanLabels, [1]).data[0] ?? 0
        if (t > 1000) size = Math.min(nc, Math.max(allowUnlabeled ? 0 : 1, size))
      }
      const chosen = new Set<number>()
      for (let t = 0; chosen.size < size; t++)
        chosen.add(pick(uniform(foldIn(ki, 2000 + t), [1]).data[0] ?? 0, classProbs))
      let words = 0
      for (let t = 0; words === 0; t++) words = poisson(foldIn(ki, 5000 + t), length, [1]).data[0] ?? 0
      const mixture =
        chosen.size === 0
          ? new Array<number>(d).fill(1 / d)
          : Array.from({ length: d }, (_, w) => {
              let s = 0
              for (const c of chosen) s += wordProbs[c]?.[w] ?? 0
              return s / chosen.size
            })
      const u = uniform(foldIn(ki, 9000), [words]).data
      for (const v of u) {
        const w = pick(v, mixture)
        x[i * d + w] = (x[i * d + w] ?? 0) + 1
      }
      for (const c of chosen) y[i * nc + c] = 1
    }
    const truth: Truth = { latents: { classProbabilities: classProbs, wordProbabilities: wordProbs } }
    return {
      data: xyMultilabel(fromStorage(x, [n, d], 'float64'), fromStorage(y, [n, nc], 'int32'), {
        ...stamp('makeMultilabelClassification', makeMultilabelClassification.description, knobs),
      }),
      truth,
    }
  },
}
