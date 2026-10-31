/**
 * A deliberately heavy toy method for the worker recipe: full-batch gradient descent on a logistic regression with tens
 * of thousands of planted rows, so a run of a few hundred steps takes seconds of real computation. It logs the loss, the
 * accuracy, the gradient norm and the weights, and has one dear derived path: the loss over a slice of the weights around
 * the current iterate, on a grid, which the page asks the worker for at any step.
 */

import { foldIn, type Key, normal } from 'mdml-engine/random'
import { dict, int, real } from 'mdml-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from 'mdml-engine/trace'

/** The iterate of the descent. */
export interface Descent {
  readonly k: number
  readonly w: Float64Array
}

interface Data {
  readonly x: Float64Array
  readonly y: Float64Array
  readonly rows: number
  readonly dims: number
}

// The planted data, built once per (size, key): the algorithm and the contract's derived path read the same rows.
const cache = new Map<string, Data>()

function planted(rows: number, dims: number, key: Key): Data {
  const id = `${rows}|${dims}|${key.words.join(',')}`
  const hit = cache.get(id)
  if (hit !== undefined) return hit
  const x = normal(foldIn(key, 1), [rows, dims]).data as Float64Array
  const truth = normal(foldIn(key, 2), [dims]).data as Float64Array
  const noise = normal(foldIn(key, 3), [rows]).data as Float64Array
  const y = new Float64Array(rows)
  for (let i = 0; i < rows; i++) {
    let z = 0
    for (let j = 0; j < dims; j++) z += (x[i * dims + j] ?? 0) * (truth[j] ?? 0)
    y[i] = z + 0.5 * (noise[i] ?? 0) > 0 ? 1 : 0
  }
  const data = { x, y, rows, dims }
  if (cache.size >= 4) cache.clear()
  cache.set(id, data)
  return data
}

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

// The mean logistic loss over the first `n` rows at weights `w`.
function lossAt(d: Data, w: Float64Array, n: number): number {
  let total = 0
  for (let i = 0; i < n; i++) {
    let z = 0
    for (let j = 0; j < d.dims; j++) z += (d.x[i * d.dims + j] ?? 0) * (w[j] ?? 0)
    const y = d.y[i] ?? 0
    // log(1 + exp(z)) - y z, stable for large z.
    total += (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - y * z
  }
  return total / n
}

const LANDSCAPE = 21
const LAND_ROWS = 1500

/** The heavy descent, for the worker recipe. */
export const heavyDescent: Method<Descent, Config> = defineMethod<Descent, Config>({
  name: 'heavy-descent',
  config: dict({
    rows: int(1000, 100000, { default: 40000, label: 'rows of data' }),
    rate: real(0.01, 2, { default: 0.5, label: 'learning rate' }),
  }),
  make: (c, key) => {
    const dims = 20
    const d = planted(c['rows'] as number, dims, key)
    const rate = c['rate'] as number
    return {
      name: 'heavy-descent',
      config: { rows: d.rows, rate },
      init: () => ({ k: 0, w: new Float64Array(dims) }),
      step: (s, ctx) => {
        const g = new Float64Array(dims)
        let loss = 0
        let right = 0
        for (let i = 0; i < d.rows; i++) {
          let z = 0
          for (let j = 0; j < dims; j++) z += (d.x[i * dims + j] ?? 0) * (s.w[j] ?? 0)
          const p = sigmoid(z)
          const y = d.y[i] ?? 0
          loss -= y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12)
          if (p > 0.5 === (y === 1)) right++
          const r = p - y
          for (let j = 0; j < dims; j++) g[j] = (g[j] ?? 0) + r * (d.x[i * dims + j] ?? 0)
        }
        let norm = 0
        const w = new Float64Array(dims)
        for (let j = 0; j < dims; j++) {
          const gj = (g[j] ?? 0) / d.rows
          norm += gj * gj
          w[j] = (s.w[j] ?? 0) - rate * gj
        }
        ctx.log('loss', loss / d.rows)
        ctx.log('accuracy', right / d.rows)
        ctx.log('gradient-norm', Math.sqrt(norm))
        ctx.log('weights', Array.from(w.subarray(0, 4)))
        return { k: s.k + 1, w }
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as Descent,
    }
  },
  contract: (c, key) => {
    const d = planted(c['rows'] as number, 20, key)
    return {
      loss: scalar({ direction: 'lower', description: 'the mean logistic loss over all rows' }),
      accuracy: scalar({ direction: 'higher', description: 'the share of rows classified right' }),
      'gradient-norm': scalar({ direction: 'lower', description: 'the norm of the full gradient' }),
      weights: tensor({ role: 'param', shape: [4], description: 'the first four weights' }),
      landscape: derived<Descent>({
        description: 'the loss over the first two weights, the rest held at the iterate (dear: a grid of losses)',
        cost: 'dear',
        compute: (s) => {
          const axis = Array.from({ length: LANDSCAPE }, (_, i) => -3 + (6 * i) / (LANDSCAPE - 1))
          const w = Float64Array.from(s.w)
          const values = axis.map((w1) =>
            axis.map((w0) => {
              w[0] = w0
              w[1] = w1
              return lossAt(d, w, LAND_ROWS)
            }),
          )
          return { kind: 'decisionSurface', features: ['w0', 'w1'], x: axis, y: axis, values, field: { name: 'loss' } }
        },
      }),
    }
  },
  defaults: {
    loss: Log(),
    accuracy: Log(),
    'gradient-norm': Log(),
    weights: Log({ detail: 'full' }),
  },
})
