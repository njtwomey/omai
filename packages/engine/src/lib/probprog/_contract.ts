// Exact summation over enumerated discrete sites (Pyro's TraceEnum machinery, funsor's `partial_sum_product`): site log
// densities become named log factors over enumerated variables and plates, and plated sum-product contraction
// eliminates the variables (greedy variable elimination inside each plate context) and multiplies over plates. Every
// operation is a tensor operation, so the result differentiates with respect to continuous sites and parameters.
import { WhiteboxError } from '../base/index.ts'
import {
  add,
  broadcastTo,
  cast,
  logsumexp,
  mul,
  reshape,
  sum,
  tensor,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import type { ProgramTrace, SampleSite } from './handlers.ts'
import type { Message, Messenger, PlateFrame } from './primitives.ts'

/** A log factor with named axes: `value.shape[i]` is the size of `dims[i]`; `plates` are the site's plates. */
export interface NamedFactor {
  readonly value: Value
  readonly dims: readonly string[]
  readonly plates: readonly string[]
  /** The site the factor came from (for diagnostics and marginals). */
  readonly site?: string
}

function sizeOf(f: NamedFactor, d: string): number {
  return f.value.shape[f.dims.indexOf(d)] ?? 1
}

/** Reorder and broadcast-pad a factor's axes to `dims` (missing names become size-1 axes). */
export function alignFactor(f: NamedFactor, dims: readonly string[]): Value {
  const present = dims.filter((d) => f.dims.includes(d))
  const perm = present.map((d) => f.dims.indexOf(d))
  const moved = perm.every((p, i) => p === i) ? f.value : transpose(f.value, perm)
  const shape = dims.map((d) => (f.dims.includes(d) ? sizeOf(f, d) : 1))
  return reshape(moved, shape)
}

function union(a: readonly string[], b: readonly string[]): string[] {
  return [...a, ...b.filter((d) => !a.includes(d))]
}

/** The product of two log factors (their sum). */
export function multiplyFactors(a: NamedFactor, b: NamedFactor): NamedFactor {
  const dims = union(a.dims, b.dims)
  const value = add(alignFactor(a, dims), alignFactor(b, dims))
  const shape = dims.map((d) => Math.max(a.dims.includes(d) ? sizeOf(a, d) : 1, b.dims.includes(d) ? sizeOf(b, d) : 1))
  return { value: broadcastTo(value, shape), dims, plates: union(a.plates, b.plates) }
}

/** Sum a log factor's variables out (log-sum-exp over their axes). */
export function sumOut(f: NamedFactor, vars: readonly string[]): NamedFactor {
  const axes = vars.map((v) => f.dims.indexOf(v)).filter((a) => a >= 0)
  if (axes.length === 0) return f
  return { value: logsumexp(f.value, axes), dims: f.dims.filter((d) => !vars.includes(d)), plates: f.plates }
}

/** Multiply a log factor over plates (sum over their axes, times each plate's scale). */
export function productOver(
  f: NamedFactor,
  plates: readonly string[],
  scales: ReadonlyMap<string, number>,
): NamedFactor {
  const axes = plates.map((p) => f.dims.indexOf(p)).filter((a) => a >= 0)
  let value = axes.length === 0 ? f.value : sum(f.value, axes)
  const s = plates.reduce((acc, p) => acc * (scales.get(p) ?? 1), 1)
  if (s !== 1) value = mul(value, s)
  return { value, dims: f.dims.filter((d) => !plates.includes(d)), plates: f.plates.filter((p) => !plates.includes(p)) }
}

/** Eliminate variables from a group of factors by greedy variable elimination (smallest intermediate first). */
function eliminate(factors: readonly NamedFactor[], vars: readonly string[]): NamedFactor {
  let fs = [...factors]
  const todo = new Set(vars)
  while (todo.size > 0) {
    let best = ''
    let bestCost = Number.POSITIVE_INFINITY
    for (const v of todo) {
      const touching = fs.filter((f) => f.dims.includes(v))
      const dims = touching.reduce<string[]>((acc, f) => union(acc, f.dims), [])
      const cost = dims.reduce((acc, d) => {
        const owner = touching.find((f) => f.dims.includes(d))
        return acc * (owner === undefined ? 1 : sizeOf(owner, d))
      }, 1)
      if (cost < bestCost || (cost === bestCost && v < best)) {
        best = v
        bestCost = cost
      }
    }
    const touching = fs.filter((f) => f.dims.includes(best))
    const rest = fs.filter((f) => !f.dims.includes(best))
    todo.delete(best)
    if (touching.length === 0) continue
    const joined = touching.slice(1).reduce((acc, f) => multiplyFactors(acc, f), touching[0] as NamedFactor)
    fs = [...rest, sumOut(joined, [best])]
  }
  if (fs.length === 0) return { value: tensor(0), dims: [], plates: [] }
  return fs.slice(1).reduce((acc, f) => multiplyFactors(acc, f), fs[0] as NamedFactor)
}

const keyOf = (plates: readonly string[]): string => [...plates].sort().join('\u0000')

/**
 * Plated sum-product contraction in log space (funsor's `partial_sum_product` with log-sum-exp and addition):
 * contract the deepest plate context first, eliminating variables local to it, then multiply over the plates that no
 * remaining variable needs, and pass the result outwards.
 *
 * @param factors - Log factors.
 * @param sumVars - The enumerated variables to sum out.
 * @param scales - The scale of each plate (size / subsample size), for product over subsampled plates.
 * @returns The contracted log total (a scalar value).
 * @throws WhiteboxError with code `not-implemented` when a variable is shared across plates in a way plated
 *   contraction cannot handle.
 */
export function contractFactors(
  factors: readonly NamedFactor[],
  sumVars: ReadonlySet<string>,
  scales: ReadonlyMap<string, number>,
): Value {
  const varPlates = new Map<string, Set<string>>()
  const groups = new Map<string, { plates: string[]; factors: NamedFactor[] }>()
  const push = (f: NamedFactor): void => {
    const k = keyOf(f.plates)
    const g = groups.get(k)
    if (g === undefined) groups.set(k, { plates: [...f.plates], factors: [f] })
    else g.factors.push(f)
  }
  for (const f of factors) {
    push(f)
    for (const v of f.dims) {
      if (!sumVars.has(v)) continue
      const prev = varPlates.get(v)
      if (prev === undefined) varPlates.set(v, new Set(f.plates))
      else for (const p of [...prev]) if (!f.plates.includes(p)) prev.delete(p)
    }
  }
  let total: Value = tensor(0)
  let guard = 0
  while (groups.size > 0) {
    guard += 1
    if (guard > 10000) throw new WhiteboxError('value', 'contract: did not terminate')
    let leafKey = ''
    let leaf: { plates: string[]; factors: NamedFactor[] } | undefined
    for (const [k, g] of groups) {
      if (
        leaf === undefined ||
        g.plates.length > leaf.plates.length ||
        (g.plates.length === leaf.plates.length && k < leafKey)
      ) {
        leaf = g
        leafKey = k
      }
    }
    if (leaf === undefined) break
    groups.delete(leafKey)
    const leafPlates = leaf.plates
    const leafVars = new Set([...varPlates.entries()].filter(([, ps]) => keyOf([...ps]) === leafKey).map(([v]) => v))
    // Connected components of the leaf's factors through the leaf's variables.
    const remaining = [...leaf.factors]
    while (remaining.length > 0) {
      const component: NamedFactor[] = [remaining.shift() as NamedFactor]
      const vars = new Set(component[0]?.dims.filter((d) => leafVars.has(d)) ?? [])
      let grew = true
      while (grew) {
        grew = false
        for (let i = remaining.length - 1; i >= 0; i--) {
          const f = remaining[i] as NamedFactor
          if (f.dims.some((d) => vars.has(d))) {
            component.push(f)
            for (const d of f.dims) if (leafVars.has(d)) vars.add(d)
            remaining.splice(i, 1)
            grew = true
          }
        }
      }
      const f = eliminate(component, [...vars])
      const left = f.dims.filter((d) => sumVars.has(d))
      if (left.length === 0) {
        total = add(total, productOver({ ...f, plates: leafPlates }, leafPlates, scales).value)
        continue
      }
      const outer = new Set<string>()
      for (const v of left) for (const p of varPlates.get(v) ?? []) outer.add(p)
      if (keyOf([...outer]) === leafKey)
        throw new WhiteboxError(
          'not-implemented',
          `contract: enumerated variables ${left.join(', ')} cannot be summed out inside plates ${leafPlates.join(', ')}`,
        )
      const reduced = productOver(
        { ...f, plates: leafPlates },
        leafPlates.filter((p) => !outer.has(p)),
        scales,
      )
      push({ ...reduced, plates: [...outer] })
    }
  }
  return total
}

/**
 * The enumeration handler (NumPyro's `enum`): every latent sample site marked `infer.enumerate = 'parallel'` without
 * a value takes all its states at once, along a fresh batch dimension left of the plates. Inside a markov loop,
 * dimensions of sites more than `history` steps back are reused.
 *
 * @param firstAvailableDim - The first dimension left of every plate (negative).
 * @returns The handler.
 */
export function enumerationHandler(firstAvailableDim: number): Messenger {
  const live: { dim: number; loop: string | undefined; step: number }[] = []
  let next = firstAvailableDim
  return {
    process: (msg: Message) => {
      if (msg.type !== 'sample' || msg.isObserved || msg.value !== undefined) return
      if (msg.infer.enumerate !== 'parallel' || msg.fn === undefined) return
      const support = msg.fn.support
      let low: number
      let high: number
      if (support.kind === 'boolean') {
        low = 0
        high = 1
      } else if (support.kind === 'integerInterval') {
        low = support.low
        high = support.high
      } else
        throw new WhiteboxError(
          'value',
          `enumerate: site ${msg.name} has the ${support.kind} support, which is not finite`,
        )
      if (msg.fn.eventShape.length > 0 || msg.sampleShape.length > 0)
        throw new WhiteboxError('value', `enumerate: site ${msg.name} must have a scalar event and no sample shape`)
      const frame = msg.markov[0]
      let dim: number | undefined
      if (frame !== undefined) {
        const reusable = live.filter((e) => e.loop === frame.name && e.step < frame.step - frame.history)
        reusable.sort((a, b) => b.dim - a.dim)
        const chosen = reusable[0]
        if (chosen !== undefined) {
          dim = chosen.dim
          live.splice(live.indexOf(chosen), 1)
        }
      }
      if (dim === undefined) {
        dim = next
        next -= 1
      }
      live.push({ dim, loop: frame?.name, step: frame?.step ?? 0 })
      const k = high - low + 1
      const shape = [k, ...new Array<number>(-dim - 1).fill(1)]
      msg.value = reshape(tensor(Array.from({ length: k }, (_, i) => low + i)), shape)
      msg.enumDim = dim
    },
  }
}

/** The deepest plate nesting of a trace (the number of batch dimensions plates use). */
export function maxPlateNesting(t: ProgramTrace): number {
  let depth = 0
  for (const site of t.sites.values()) for (const f of site.plates) depth = Math.max(depth, -f.dim)
  return depth
}

/** Whether a trace has enumerated sites. */
export function hasEnumerated(t: ProgramTrace): boolean {
  for (const site of t.sites.values()) if (site.type === 'sample' && site.enumDim !== undefined) return true
  return false
}

/** The product of a site's plate scales. */
export function plateScale(plates: readonly PlateFrame[]): number {
  return plates.reduce((acc, f) => acc * (f.subsampleSize === 0 ? 1 : f.size / f.subsampleSize), 1)
}

/** A site's log density with its mask applied (masked elements contribute 0). */
export function maskedLogProb(site: SampleSite): Value {
  const lp = site.fn.logProb(site.value)
  if (site.mask === undefined) return lp
  return where(cast(site.mask, 'bool'), lp, 0)
}

/**
 * The named log factors of a trace: one per sample site (latent or observed), with axes named by enumerated sites and
 * plates; the handler scale (beyond plate subsampling) multiplies the factor.
 *
 * @param t - A trace made under {@link enumerationHandler}.
 * @param plateDepth - The plate nesting the enumeration dimensions were placed beyond.
 * @returns The factors, the enumerated variables and the plate scales.
 */
export function traceFactors(
  t: ProgramTrace,
  plateDepth: number,
): { readonly factors: NamedFactor[]; readonly sumVars: Set<string>; readonly scales: Map<string, number> } {
  const factors: NamedFactor[] = []
  const sumVars = new Set<string>()
  const scales = new Map<string, number>()
  const owner = new Map<number, string>()
  for (const site of t.sites.values()) {
    if (site.type !== 'sample') continue
    if (site.enumDim !== undefined) {
      owner.set(site.enumDim, site.name)
      sumVars.add(site.name)
    }
    for (const f of site.plates) scales.set(f.name, f.subsampleSize === 0 ? 1 : f.size / f.subsampleSize)
    let lp = maskedLogProb(site)
    const handlerScale = site.scale / plateScale(site.plates)
    if (handlerScale !== 1) lp = mul(lp, handlerScale)
    const rank = lp.shape.length
    const dims: string[] = []
    const keep: number[] = []
    // Batch axes that are neither plates nor enumerated sites are independent elements of this site alone: their
    // product (the sum of their log densities) is taken here.
    const loose: number[] = []
    for (let a = 0; a < rank; a++) {
      const size = lp.shape[a] ?? 1
      if (size === 1) continue
      const d = a - rank
      const name = d >= -plateDepth ? site.plates.find((f) => f.dim === d)?.name : owner.get(d)
      if (name === undefined) {
        if (d < -plateDepth)
          throw new WhiteboxError(
            'shape',
            `enumerate: site ${site.name} has a batch dimension ${d} of size ${size} left of the plates that belongs to no enumerated site`,
            { site: site.name },
          )
        loose.push(a)
        continue
      }
      dims.push(name)
      keep.push(size)
    }
    if (loose.length > 0) lp = sum(lp, loose)
    factors.push({ value: reshape(lp, keep), dims, plates: site.plates.map((f) => f.name), site: site.name })
  }
  return { factors, sumVars, scales }
}
