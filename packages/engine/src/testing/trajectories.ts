// Optimiser trajectories on the oracle's test objective (oracle/omai_oracle/optim_helpers.py). Not published.
import { valueAndGrad } from '../lib/autodiff/index.ts'
import type { GradientTransformation } from '../lib/optim/index.ts'
import { applyUpdates } from '../lib/optim/index.ts'
import {
  add,
  mul,
  pow,
  reshape,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  tensor,
  type Value,
} from '../lib/tensor/index.ts'
import { decode, type GoldenCase } from './golden.ts'

/** 0.5 sum a (p - c)^2 + 0.05 sum p^4, elementwise. */
export const testObjective =
  (a: Value, c: Value) =>
  (p: Value): Value =>
    add(mul(sum(mul(a, square(sub(p, c)))), 0.5), mul(sum(pow(p, 4)), 0.05))

/** The oracle's sparsity mask: entry i is dropped at step t when (i + t) % 3 == 0 (flat index). */
export function everyThird(shape: readonly number[], t: number): Value {
  const n = shape.reduce((a, b) => a * b, 1)
  const flat = Array.from({ length: n }, (_, i) => ((i + t) % 3 !== 0 ? 1 : 0))
  return reshape(tensor(flat), [...shape])
}

/** The oracle's row mask: row r is dropped at step t when (r + t) % 3 == 0, entry (r, j) when (r + j + t) % 4 == 0. */
export function rowsAndEntries(shape: readonly number[], t: number): Value {
  const [m = 0, n = 0] = shape
  const rows = Array.from({ length: m }, (_, r) =>
    Array.from({ length: n }, (_, j) => ((r + t) % 3 !== 0 && (r + j + t) % 4 !== 0 ? 1 : 0)),
  )
  return tensor(rows)
}

/** Run an optimiser on a fixture case's objective and return the stacked parameters after every step. */
export function trajectory(
  c: GoldenCase,
  opt: GradientTransformation,
  options: { readonly mask?: 'element' | 'rows' } = {},
): { readonly path: Value; readonly state: unknown } {
  const [p0, a, cc] = c.inputs.map((x) => decode(x) as Tensor) as [Tensor, Tensor, Tensor]
  const vg = valueAndGrad(testObjective(a, cc))
  let params: Value = p0
  let state = opt.init(params)
  const path: Value[] = []
  for (let t = 0; t < (c.params['steps'] as number); t++) {
    const g = vg(params).grad
    const grad =
      options.mask === 'element'
        ? mul(g, everyThird(p0.shape, t))
        : options.mask === 'rows'
          ? mul(g, rowsAndEntries(p0.shape, t))
          : g
    const { updates, state: next } = opt.update(grad, state, params)
    params = applyUpdates(params, updates) as Value
    state = next
    path.push(params)
  }
  return { path: stack(path), state }
}
