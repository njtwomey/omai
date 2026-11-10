// The brute-force reference the message-passing tests share: numerical integration by composite Gauss-Legendre over
// boxes (the exact message of a factor, by integrating it against the incoming messages), and the tilted moments of a
// port computed that way. Node only; not published.
import { gaussLegendre } from '../lib/numerics/quadrature/index.ts'
import { toFlat } from '../lib/tensor/index.ts'

const rules = new Map<number, { nodes: number[]; weights: number[] }>()
function rule(order: number): { nodes: number[]; weights: number[] } {
  let r = rules.get(order)
  if (r === undefined) {
    const g = gaussLegendre(order)
    r = { nodes: toFlat(g.nodes), weights: toFlat(g.weights) }
    rules.set(order, r)
  }
  return r
}

/** The nodes and weights of a composite Gauss-Legendre rule on [a, b] with `panels` panels of `order` points. */
export function nodesOn(a: number, b: number, panels = 12, order = 16): { x: number[]; w: number[] } {
  const { nodes, weights } = rule(order)
  const x: number[] = []
  const w: number[] = []
  const h = (b - a) / panels
  for (let p = 0; p < panels; p++) {
    const lo = a + p * h
    for (let i = 0; i < order; i++) {
      x.push(lo + ((nodes[i] ?? 0) + 1) * 0.5 * h)
      w.push((weights[i] ?? 0) * 0.5 * h)
    }
  }
  return { x, w }
}

/** The integral of f over [a, b]. */
export function integrate(f: (x: number) => number, a: number, b: number, panels = 12, order = 16): number {
  const { x, w } = nodesOn(a, b, panels, order)
  let s = 0
  for (let i = 0; i < x.length; i++) s += (w[i] ?? 0) * f(x[i] ?? 0)
  return s
}

/** The integral of f over a box, one interval per dimension (a product rule: keep the dimension at most 3). */
export function integrateBox(
  f: (x: readonly number[]) => number,
  box: readonly (readonly [number, number])[],
  panels = 8,
  order = 12,
): number {
  const grids = box.map(([a, b]) => nodesOn(a, b, panels, order))
  const point: number[] = new Array<number>(box.length).fill(0)
  const walk = (d: number, weight: number): number => {
    if (d === box.length) return weight * f(point)
    const g = grids[d] as { x: number[]; w: number[] }
    let s = 0
    for (let i = 0; i < g.x.length; i++) {
      point[d] = g.x[i] ?? 0
      s += walk(d + 1, weight * (g.w[i] ?? 0))
    }
    return s
  }
  return walk(0, 1)
}

/** An interval holding a Gaussian's mass: mean +- 12 sd. */
export const around = (mean: number, variance: number, widths = 12): [number, number] => [
  mean - widths * Math.sqrt(variance),
  mean + widths * Math.sqrt(variance),
]

/** The integral of f over (0, hi), by the substitution x = e^u (resolves the behaviour at the origin). */
export function integratePositive(f: (x: number) => number, hi: number, panels = 24, order = 16): number {
  return integrate((u) => f(Math.exp(u)) * Math.exp(u), -40, Math.log(hi), panels, order)
}
