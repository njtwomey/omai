// The brute-force reference the message-passing tests share: numerical integration by composite Gauss-Legendre over
// boxes (the exact message of a factor, by integrating it against the incoming messages), and the tilted moments of a
// port computed that way. Node only; not published.
import { lgammaOf, trigammaOf } from '../lib/message-passing/_scalar.ts'
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

/** One port of a factor for the brute-force tilted moments. */
export interface BrutePort {
  readonly kind: 'gaussian' | 'bernoulli'
  /** The incoming message (natural parameters of the family); absent for an observed port. */
  readonly eta?: readonly number[]
  /** The observed value. */
  readonly value?: number
  /** A deterministic port: its value is this function of the other (free) ports' values. */
  readonly derive?: (x: readonly number[]) => number
  /** The interval of a Gaussian port (default its message's mean +- 10 sd). */
  readonly range?: readonly [number, number]
}

/** log of the unnormalised message exp(eta . T(x)) of a scalar Gaussian or Bernoulli port. */
export function logMessage(kind: BrutePort['kind'], eta: readonly number[], x: number): number {
  return kind === 'gaussian' ? (eta[0] ?? 0) * x - 0.5 * (eta[1] ?? 0) * x * x : (eta[0] ?? 0) * x
}

/**
 * The log mass of a factor under the incoming messages of its ports, and the expected sufficient statistics at port
 * `to` of the tilted distribution (the factor times every port's message), by integrating over the Gaussian ports and
 * summing over the Bernoulli ones. A port with `derive` is determined by the others (a deterministic factor); an
 * observed port is a constant. For a Gaussian target the moments are (E x, E x^2), for a Bernoulli target P(true).
 */
export function bruteTilted(
  ports: readonly BrutePort[],
  logf: (x: readonly number[]) => number,
  to: number,
  options: { readonly panels?: number; readonly order?: number } = {},
): { readonly logZ: number; readonly moments: number[] } {
  const free = ports.map((_, i) => i).filter((i) => ports[i]?.derive === undefined && ports[i]?.value === undefined)
  const intervals = free.map((i) => {
    const p = ports[i] as BrutePort
    if (p.kind === 'bernoulli') return [0, 1] as const
    if (p.range !== undefined) return p.range
    const j = p.eta?.[1] ?? 1
    const m = (p.eta?.[0] ?? 0) / j
    return around(m, 1 / j, 10)
  })
  const grids = free.map((i, d) => {
    const p = ports[i] as BrutePort
    if (p.kind === 'bernoulli') return { x: [0, 1], w: [1, 1] }
    const [a, b] = intervals[d] as readonly [number, number]
    return nodesOn(a, b, options.panels ?? 10, options.order ?? 12)
  })
  const x = new Array<number>(ports.length).fill(0)
  for (const [i, p] of ports.entries()) if (p.value !== undefined) x[i] = p.value
  let m0 = 0
  let m1 = 0
  let m2 = 0
  const target = ports[to] as BrutePort
  const visit = (d: number, logw: number, w: number): void => {
    if (d === free.length) {
      for (const [i, p] of ports.entries()) if (p.derive !== undefined) x[i] = p.derive(x)
      let lw = logw + logf(x)
      for (const [i, p] of ports.entries())
        if (p.eta !== undefined && (p.derive !== undefined || true)) lw += logMessage(p.kind, p.eta, x[i] ?? 0)
      const v = w * Math.exp(lw)
      const t = x[to] ?? 0
      m0 += v
      m1 += v * t
      m2 += v * t * t
      return
    }
    const g = grids[d] as { x: number[]; w: number[] }
    const i = free[d] as number
    for (let k = 0; k < g.x.length; k++) {
      x[i] = g.x[k] ?? 0
      visit(d + 1, logw, w * (g.w[k] ?? 0))
    }
  }
  visit(0, 0, 1)
  return { logZ: Math.log(m0), moments: target.kind === 'gaussian' ? [m1 / m0, m2 / m0] : [m1 / m0] }
}

/** One independent belief of a brute-force variational check: a Gaussian, gamma or beta message and the coordinate the log density reads. */
export interface BruteBelief {
  readonly kind: 'gaussian' | 'gamma' | 'beta' | 'discrete'
  readonly eta: readonly number[]
  /** For a gamma belief: the log density reads the log of the variable (default the variable itself). */
  readonly log?: boolean
  /** For a Gaussian belief: where the integrand has a kink, so the panels are cut there. */
  readonly kinks?: readonly number[]
  /** The panels of the composite rule (default 16 for a Gaussian, 24 for a gamma or a beta): fewer for a product over many beliefs. */
  readonly panels?: number
}

const logOnePlusExp = (x: number): number => (x > 30 ? x : Math.log1p(Math.exp(x)))

/** The nodes of one belief by composite Gauss-Legendre (a rule of its own, independent of the library's Gauss-Hermite and grids), with the density folded into the weights. */
function bruteNodes(b: BruteBelief): { x: number[]; w: number[]; stat: number[][] } {
  if (b.kind === 'discrete') {
    const top = Math.max(...b.eta)
    const total = b.eta.reduce((a, v) => a + Math.exp(v - top), 0)
    return { x: b.eta.map((_, i) => i), w: b.eta.map((v) => Math.exp(v - top) / total), stat: [] }
  }
  if (b.kind === 'gaussian') {
    const j = b.eta[1] ?? 1
    const mean = (b.eta[0] ?? 0) / j
    const [lo, hi] = around(mean, 1 / j, 12)
    const cuts = [lo, ...(b.kinks ?? []).filter((k) => k > lo && k < hi).sort((p, q) => p - q), hi]
    const x: number[] = []
    const w: number[] = []
    for (let k = 0; k + 1 < cuts.length; k++) {
      const part = nodesOn(cuts[k] as number, cuts[k + 1] as number, b.panels ?? 16, 16)
      x.push(...part.x)
      w.push(...part.w)
    }
    return {
      x,
      w: x.map((v, i) => (w[i] ?? 0) * Math.exp(-0.5 * j * (v - mean) ** 2) * Math.sqrt(j / (2 * Math.PI))),
      stat: [],
    }
  }
  if (b.kind === 'gamma') {
    const shape = (b.eta[0] ?? 0) + 1
    const rate = -(b.eta[1] ?? 0)
    const logZ = lgammaOf(shape) - shape * Math.log(rate)
    const mode = Math.log(shape / rate)
    const sd = Math.sqrt(trigammaOf(shape))
    const { x: t, w } = nodesOn(
      mode - Math.min(Math.max(14 * sd, 30 / shape), 60),
      mode + Math.min(14 * sd, 10),
      b.panels ?? 24,
      12,
    )
    return {
      x: b.log === true ? t : t.map(Math.exp),
      w: t.map((u, i) => (w[i] ?? 0) * Math.exp(shape * u - rate * Math.exp(u) - logZ)),
      stat: [],
    }
  }
  const p = (b.eta[0] ?? 0) + 1
  const q = (b.eta[1] ?? 0) + 1
  const logB = lgammaOf(p) + lgammaOf(q) - lgammaOf(p + q)
  const mode = Math.log(p / q)
  const sd = Math.sqrt(trigammaOf(p) + trigammaOf(q))
  const { x: u, w } = nodesOn(
    mode - Math.min(Math.max(14 * sd, 30 / p), 60),
    mode + Math.min(Math.max(14 * sd, 30 / q), 60),
    b.panels ?? 24,
    12,
  )
  return {
    x: u.map((v) => 1 / (1 + Math.exp(-v))),
    w: u.map((v, i) => (w[i] ?? 0) * Math.exp(-p * logOnePlusExp(-v) - q * logOnePlusExp(v) - logB)),
    stat: [],
  }
}

/**
 * The expectation of a function of the coordinates under independent Gaussian, gamma, beta and discrete beliefs, by a composite
 * Gauss-Legendre product rule over each belief's range (not the library's quadrature): the brute-force reference of a variational
 * expected log.
 *
 * @param beliefs - The independent beliefs.
 * @param g - The function of one coordinate per belief.
 * @returns `E[g]`.
 */
export function bruteExpectation(beliefs: readonly BruteBelief[], g: (x: readonly number[]) => number): number {
  const grids = beliefs.map(bruteNodes)
  const x = new Array<number>(beliefs.length).fill(0)
  const walk = (d: number, weight: number): number => {
    if (d === grids.length) return weight * g(x)
    const grid = grids[d] as { x: number[]; w: number[] }
    let s = 0
    for (let i = 0; i < grid.x.length; i++) {
      const wi = grid.w[i] as number
      if (wi < 1e-13) continue
      x[d] = grid.x[i] as number
      s += walk(d + 1, weight * wi)
    }
    return s
  }
  return walk(0, 1)
}
