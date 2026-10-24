import { argminOf, heap, WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'

const squared = (a: ArrayLike<number>, ia: number, b: ArrayLike<number>, ib: number, d: number): number => {
  let s = 0
  for (let k = 0; k < d; k++) s += ((a[ia + k] ?? 0) - (b[ib + k] ?? 0)) ** 2
  return s
}

/** A trained codebook. */
export interface Codebook {
  /** Centroids `[k, d]`. */
  readonly centroids: Tensor
  /** The nearest centroid of each training point. */
  readonly assignments: Int32Array
  /** Sum of squared distances to the assigned centroids. */
  readonly inertia: number
  /** Inertia after each Lloyd iteration. */
  readonly history: readonly number[]
  readonly iterations: number
}

/**
 * A vector-quantisation codebook by k-means: k-means++ seeding (Arthur and Vassilvitskii) or given initial centroids,
 * then Lloyd iterations until the relative change in inertia is below tol. The trainer shared by product
 * quantisation, IVF and the clustering methods.
 *
 * @param data - Points `[n, d]`.
 * @param k - Number of centroids.
 * @param options - Key (for k-means++), initial centroids, iteration cap (default 100) and tolerance (default 1e-6).
 * @returns Centroids, assignments and the inertia history.
 * @concept vector-quantisation
 * @example
 * ```ts
 * import { kmeansCodebook } from 'whitebox-engine/numerics/spatial'
 * import { key } from 'whitebox-engine/random'
 * kmeansCodebook([[0], [0.1], [10], [10.1]], 2, { key: key(0) }).inertia < 0.011 // => true
 * ```
 */
export function kmeansCodebook(
  data: TensorLike,
  k: number,
  options: {
    readonly key?: Key
    readonly init?: TensorLike
    readonly maxIterations?: number
    readonly tol?: number
  } = {},
): Codebook {
  const t = concreteOrThrow(promoteFloat(data)[0] as Tensor, 'kmeansCodebook')
  const x = toFlat(t)
  const [n = 0, d = 0] = t.shape
  if (k < 1 || k > n) throw new WhiteboxError('value', `kmeansCodebook: k must be in [1, ${n}]`)
  const c = new Float64Array(k * d)
  if (options.init !== undefined) {
    const init = toFlat(concreteOrThrow(promoteFloat(options.init)[0] as Tensor, 'kmeansCodebook init'))
    if (init.length !== k * d) throw new WhiteboxError('shape', 'kmeansCodebook: init must be [k, d]')
    c.set(init)
  } else {
    if (options.key === undefined)
      throw new WhiteboxError('value', 'kmeansCodebook: a key is needed for k-means++ seeding')
    const u = uniform(options.key, [k]).data
    const first = Math.min(n - 1, Math.floor((u[0] ?? 0) * n))
    c.set(x.slice(first * d, first * d + d), 0)
    const nearest = Float64Array.from({ length: n }, (_, i) => squared(x, i * d, c, 0, d))
    for (let j = 1; j < k; j++) {
      // Sample proportionally to the squared distance to the nearest chosen centroid.
      let total = 0
      for (const v of nearest) total += v
      let target = (u[j] ?? 0) * total
      let pick = n - 1
      for (let i = 0; i < n; i++) {
        target -= nearest[i] ?? 0
        if (target < 0) {
          pick = i
          break
        }
      }
      c.set(x.slice(pick * d, pick * d + d), j * d)
      for (let i = 0; i < n; i++) nearest[i] = Math.min(nearest[i] ?? 0, squared(x, i * d, c, j * d, d))
    }
  }
  const assignments = new Int32Array(n)
  const history: number[] = []
  let iterations = 0
  for (let it = 0; it < (options.maxIterations ?? 100); it++) {
    iterations++
    let inertia = 0
    for (let i = 0; i < n; i++) {
      let best = 0
      let bestD = Number.POSITIVE_INFINITY
      for (let j = 0; j < k; j++) {
        const dd = squared(x, i * d, c, j * d, d)
        if (dd < bestD) {
          bestD = dd
          best = j
        }
      }
      assignments[i] = best
      inertia += bestD
    }
    history.push(inertia)
    const sums = new Float64Array(k * d)
    const counts = new Int32Array(k)
    for (let i = 0; i < n; i++) {
      const j = assignments[i] ?? 0
      counts[j] = (counts[j] ?? 0) + 1
      for (let q = 0; q < d; q++) sums[j * d + q] = (sums[j * d + q] ?? 0) + (x[i * d + q] ?? 0)
    }
    for (let j = 0; j < k; j++) {
      // Empty clusters keep their centroid.
      if ((counts[j] ?? 0) > 0) for (let q = 0; q < d; q++) c[j * d + q] = (sums[j * d + q] ?? 0) / (counts[j] ?? 1)
    }
    const prev = history[history.length - 2]
    if (prev !== undefined && prev - inertia <= (options.tol ?? 1e-6) * Math.max(prev, 1e-300)) break
  }
  // Final assignment against the final centroids.
  let inertia = 0
  for (let i = 0; i < n; i++) {
    const ds = Array.from({ length: k }, (_, j) => squared(x, i * d, c, j * d, d))
    const j = argminOf(ds)
    assignments[i] = j
    inertia += ds[j] ?? 0
  }
  return { centroids: fromStorage(c, [k, d], 'float64'), assignments, inertia, history, iterations }
}

/** A product quantiser. */
export interface ProductQuantiser {
  readonly subspaces: number
  readonly centroidsPerSubspace: number
  /** One codebook `[k, d / m]` per subspace. */
  readonly codebooks: readonly Tensor[]
  /** Codes `[n, m]` (the nearest centroid in each subspace). */
  readonly encode: (points: TensorLike) => Int32Array
  /** Reconstructions `[n, d]` from codes. */
  readonly decode: (codes: Int32Array) => Tensor
  /** The asymmetric distance table `[m, k]`: squared distances from each subvector of q to each centroid. */
  readonly distanceTable: (q: ArrayLike<number>) => Float64Array
  /** Squared distances from q to encoded points by table lookups (asymmetric distance computation). */
  readonly asymmetricDistances: (q: ArrayLike<number>, codes: Int32Array) => Float64Array
  /** Squared distances between two encoded points from centroid-to-centroid tables (symmetric distance computation). */
  readonly symmetricDistance: (a: Int32Array, ia: number, b: Int32Array, ib: number) => number
}

/**
 * Product quantisation (Jegou, Douze and Schmid 2011): split vectors into m subvectors and quantise each with its own
 * k-centroid codebook, so a vector is stored in m small integers and distances to a query are sums of m table lookups.
 *
 * @param data - Training points `[n, d]` (d divisible by m).
 * @param options - Subspaces m (default 4), centroids per subspace k (default 256, capped at n), key, Lloyd iterations.
 * @returns The quantiser.
 * @concept product-quantisation
 * @example
 * ```ts
 * import { productQuantiser } from 'whitebox-engine/numerics/spatial'
 * import { key } from 'whitebox-engine/random'
 * const pq = productQuantiser([[0, 0], [0, 1], [1, 0], [1, 1]], { m: 2, k: 2, key: key(0) })
 * Array.from(pq.encode([[1, 0]])).length // => 2
 * ```
 */
export function productQuantiser(
  data: TensorLike,
  options: { readonly m?: number; readonly k?: number; readonly key: Key; readonly maxIterations?: number },
): ProductQuantiser {
  const t = concreteOrThrow(promoteFloat(data)[0] as Tensor, 'productQuantiser')
  const x = toFlat(t)
  const [n = 0, d = 0] = t.shape
  const m = options.m ?? 4
  if (d % m !== 0) throw new WhiteboxError('value', `productQuantiser: dimension ${d} is not divisible by ${m}`)
  const ds = d / m
  const k = Math.min(options.k ?? 256, n)
  const codebooks = Array.from({ length: m }, (_, s) => {
    const sub = new Float64Array(n * ds)
    for (let i = 0; i < n; i++) for (let q = 0; q < ds; q++) sub[i * ds + q] = x[i * d + s * ds + q] ?? 0
    return kmeansCodebook(fromStorage(sub, [n, ds], 'float64'), k, {
      key: foldIn(options.key, s),
      ...(options.maxIterations === undefined ? {} : { maxIterations: options.maxIterations }),
    }).centroids
  })
  const books = codebooks.map((c) => Float64Array.from(c.data))
  const encode = (points: TensorLike): Int32Array => {
    const p = concreteOrThrow(promoteFloat(points)[0] as Tensor, 'encode')
    const pf = toFlat(p)
    const rows = p.shape[0] ?? 0
    const codes = new Int32Array(rows * m)
    for (let i = 0; i < rows; i++) {
      for (let s = 0; s < m; s++) {
        let best = 0
        let bestD = Number.POSITIVE_INFINITY
        for (let j = 0; j < k; j++) {
          const dd = squared(pf, i * d + s * ds, books[s] as Float64Array, j * ds, ds)
          if (dd < bestD) {
            bestD = dd
            best = j
          }
        }
        codes[i * m + s] = best
      }
    }
    return codes
  }
  const decode = (codes: Int32Array): Tensor => {
    const rows = codes.length / m
    const out = new Float64Array(rows * d)
    for (let i = 0; i < rows; i++) {
      for (let s = 0; s < m; s++) {
        const j = codes[i * m + s] ?? 0
        for (let q = 0; q < ds; q++) out[i * d + s * ds + q] = (books[s] as Float64Array)[j * ds + q] ?? 0
      }
    }
    return fromStorage(out, [rows, d], 'float64')
  }
  const distanceTable = (q: ArrayLike<number>): Float64Array => {
    const table = new Float64Array(m * k)
    for (let s = 0; s < m; s++)
      for (let j = 0; j < k; j++) table[s * k + j] = squared(q, s * ds, books[s] as Float64Array, j * ds, ds)
    return table
  }
  const centroidTables = books.map((b) => {
    const table = new Float64Array(k * k)
    for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) table[a * k + c] = squared(b, a * ds, b, c * ds, ds)
    return table
  })
  return {
    subspaces: m,
    centroidsPerSubspace: k,
    codebooks,
    encode,
    decode,
    distanceTable,
    asymmetricDistances: (q, codes) => {
      const table = distanceTable(q)
      const rows = codes.length / m
      const out = new Float64Array(rows)
      for (let i = 0; i < rows; i++) {
        let s2 = 0
        for (let s = 0; s < m; s++) s2 += table[s * k + (codes[i * m + s] ?? 0)] ?? 0
        out[i] = s2
      }
      return out
    },
    symmetricDistance: (a, ia, b, ib) => {
      let s2 = 0
      for (let s = 0; s < m; s++)
        s2 += (centroidTables[s] as Float64Array)[(a[ia * m + s] ?? 0) * k + (b[ib * m + s] ?? 0)] ?? 0
      return s2
    },
  }
}

/** An inverted-file index. */
export interface IvfIndex {
  readonly lists: number
  /** Centroids of the coarse quantiser `[nlist, d]`. */
  readonly coarse: Tensor
  /** Point indices per list. */
  readonly invertedLists: readonly (readonly number[])[]
  /** The k nearest among the points of the nprobe nearest lists (exact, or by PQ on residuals when trained with pq). */
  readonly search: (
    q: ArrayLike<number>,
    k: number,
    options?: { readonly nprobe?: number },
  ) => {
    readonly indices: number[]
    readonly distances: number[]
    readonly stats: { readonly listsProbed: number; readonly candidates: number }
  }
}

/**
 * An inverted-file index (IVF, as faiss): a coarse k-means quantiser assigns every point to a list; a query visits only
 * the nprobe lists whose centroids are nearest. With pq, residuals to the coarse centroid are product-quantised and
 * candidates are ranked by asymmetric distances (IVF-PQ).
 *
 * @param data - Points `[n, d]`.
 * @param options - Number of lists (default round(sqrt(n))), key, and optional PQ settings for the residuals.
 * @returns The index.
 * @example
 * ```ts
 * import { ivfIndex } from 'whitebox-engine/numerics/spatial'
 * import { key } from 'whitebox-engine/random'
 * ivfIndex([[0, 0], [0, 1], [9, 9], [9, 8]], { lists: 2, key: key(0) }).search([9, 8.6], 1).indices // => [2]
 * ```
 */
export function ivfIndex(
  data: TensorLike,
  options: { readonly lists?: number; readonly key: Key; readonly pq?: { readonly m?: number; readonly k?: number } },
): IvfIndex {
  const t = concreteOrThrow(promoteFloat(data)[0] as Tensor, 'ivfIndex')
  const x = toFlat(t)
  const [n = 0, d = 0] = t.shape
  const nlist = options.lists ?? Math.max(1, Math.round(Math.sqrt(n)))
  const coarse = kmeansCodebook(t, nlist, { key: foldIn(options.key, 0) })
  const cf = Float64Array.from(coarse.centroids.data)
  const lists: number[][] = Array.from({ length: nlist }, () => [])
  for (const [i, j] of coarse.assignments.entries()) lists[j]?.push(i)
  let pq: ProductQuantiser | undefined
  let codes: Int32Array | undefined
  if (options.pq !== undefined) {
    const residuals = new Float64Array(n * d)
    for (let i = 0; i < n; i++) {
      const j = coarse.assignments[i] ?? 0
      for (let q = 0; q < d; q++) residuals[i * d + q] = (x[i * d + q] ?? 0) - (cf[j * d + q] ?? 0)
    }
    pq = productQuantiser(fromStorage(residuals, [n, d], 'float64'), {
      key: foldIn(options.key, 1),
      ...(options.pq.m === undefined ? {} : { m: options.pq.m }),
      ...(options.pq.k === undefined ? {} : { k: options.pq.k }),
    })
    codes = pq.encode(fromStorage(residuals, [n, d], 'float64'))
  }
  return {
    lists: nlist,
    coarse: coarse.centroids,
    invertedLists: lists,
    search: (q, k, searchOptions = {}) => {
      const nprobe = Math.min(nlist, searchOptions.nprobe ?? 1)
      const order = Array.from({ length: nlist }, (_, j) => [squared(q, 0, cf, j * d, d), j] as const).sort(
        (a, b) => a[0] - b[0],
      )
      const best = heap((a: readonly [number, number], b: readonly [number, number]) => b[0] - a[0])
      let candidates = 0
      for (const [, j] of order.slice(0, nprobe)) {
        const members = lists[j] ?? []
        candidates += members.length
        let distances: ArrayLike<number>
        if (pq !== undefined && codes !== undefined) {
          const residual = Array.from({ length: d }, (_, q2) => (q[q2] ?? 0) - (cf[j * d + q2] ?? 0))
          const listCodes = new Int32Array(members.length * pq.subspaces)
          for (const [r, i] of members.entries()) {
            listCodes.set(codes.subarray(i * pq.subspaces, (i + 1) * pq.subspaces), r * pq.subspaces)
          }
          distances = pq.asymmetricDistances(residual, listCodes)
        } else distances = members.map((i) => squared(q, 0, x, i * d, d))
        members.forEach((i, r) => {
          best.push([distances[r] ?? 0, i])
          if (best.size > k) best.pop()
        })
      }
      const top = best.toArray().sort((a, b) => a[0] - b[0] || a[1] - b[1])
      return {
        indices: top.map((p) => p[1]),
        distances: top.map((p) => Math.sqrt(p[0])),
        stats: { listsProbed: nprobe, candidates },
      }
    },
  }
}
