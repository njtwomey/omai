// The conflict-driven clause learning core shared by `cdcl` (sat.ts) and the core, enumeration and MaxSAT routines
// (maxsat.ts): two watched literals, first-UIP learning with local minimisation, VSIDS activities with a lazy heap,
// phase saving and assumptions with failed-assumption analysis (MiniSat's analyzeFinal). Literals are coded as
// `2v` (positive) and `2v + 1` (negative) for variables `v = 1..n`.
import { type Heap, heap } from '../base/index.ts'

export const code = (lit: number): number => (lit > 0 ? 2 * lit : -2 * lit + 1)
export const dimacs = (c: number): number => (c & 1 ? -(c >> 1) : c >> 1)

/** Counters of a CDCL run. */
export interface CdclStats {
  decisions: number
  propagations: number
  conflicts: number
  learned: number
  restarts: number
}

export class CdclCore {
  readonly n: number
  readonly clauses: number[][] = []
  readonly learnt: boolean[] = []
  private readonly watches: number[][]
  readonly assign: Int8Array
  readonly level: Int32Array
  readonly reason: Int32Array
  readonly trail: number[] = []
  readonly trailLim: number[] = []
  private qhead = 0
  private readonly activity: Float64Array
  private varInc = 1
  private readonly decay: number
  private readonly order: Heap<[number, number]>
  private readonly polarity: Uint8Array
  private readonly seen: Uint8Array
  ok = true
  readonly stats: CdclStats = { decisions: 0, propagations: 0, conflicts: 0, learned: 0, restarts: 0 }

  constructor(n: number, clauses: readonly (readonly number[])[], decay = 0.95) {
    this.n = n
    this.decay = decay
    this.watches = Array.from({ length: 2 * n + 2 }, () => [])
    this.assign = new Int8Array(n + 1)
    this.level = new Int32Array(n + 1)
    this.reason = new Int32Array(n + 1).fill(-1)
    this.activity = new Float64Array(n + 1)
    this.polarity = new Uint8Array(n + 1)
    this.seen = new Uint8Array(n + 1)
    this.order = heap<[number, number]>((a, b) => b[0] - a[0] || a[1] - b[1])
    for (let v = 1; v <= n; v++) this.order.push([0, v])
    for (const c of clauses) if (!this.addClause(c)) break
  }

  get decisionLevel(): number {
    return this.trailLim.length
  }

  /** The value of a coded literal: 1 true, -1 false, 0 unassigned. */
  value(c: number): number {
    const a = this.assign[c >> 1] ?? 0
    return c & 1 ? -a : a
  }

  /** Add an input clause at level 0 (false once the formula is known unsatisfiable). */
  addClause(lits: readonly number[]): boolean {
    if (!this.ok) return false
    const codes = [...new Set(lits.map(code))]
    if (codes.some((c) => codes.includes(c ^ 1))) return true
    const live = codes.filter((c) => this.value(c) !== -1 || this.level[c >> 1] !== 0)
    if (live.some((c) => this.value(c) === 1 && this.level[c >> 1] === 0)) return true
    if (live.length === 0) {
      this.ok = false
      return false
    }
    if (live.length === 1) {
      this.enqueue(live[0] as number, -1)
      if (this.propagate() >= 0) this.ok = false
      return this.ok
    }
    this.attach(live, false)
    return true
  }

  /** Store and watch a clause of two or more literals; returns its index. */
  attach(lits: number[], learnt: boolean): number {
    const i = this.clauses.length
    this.clauses.push(lits)
    this.learnt.push(learnt)
    this.watches[(lits[0] as number) ^ 1]?.push(i)
    this.watches[(lits[1] as number) ^ 1]?.push(i)
    return i
  }

  enqueue(c: number, reason: number): void {
    const v = c >> 1
    this.assign[v] = c & 1 ? -1 : 1
    this.level[v] = this.decisionLevel
    this.reason[v] = reason
    this.trail.push(c)
  }

  /** Unit propagation; returns the index of a conflicting clause or -1. */
  propagate(): number {
    while (this.qhead < this.trail.length) {
      const p = this.trail[this.qhead++] as number
      this.stats.propagations++
      const falseLit = p ^ 1
      const ws = this.watches[p] as number[]
      let keep = 0
      for (let k = 0; k < ws.length; k++) {
        const ci = ws[k] as number
        const c = this.clauses[ci] as number[]
        if (c[0] === falseLit) {
          c[0] = c[1] as number
          c[1] = falseLit
        }
        if (this.value(c[0] as number) === 1) {
          ws[keep++] = ci
          continue
        }
        let moved = false
        for (let j = 2; j < c.length; j++) {
          if (this.value(c[j] as number) !== -1) {
            c[1] = c[j] as number
            c[j] = falseLit
            this.watches[(c[1] as number) ^ 1]?.push(ci)
            moved = true
            break
          }
        }
        if (moved) continue
        ws[keep++] = ci
        if (this.value(c[0] as number) === -1) {
          for (k++; k < ws.length; k++) ws[keep++] = ws[k] as number
          ws.length = keep
          this.qhead = this.trail.length
          return ci
        }
        this.enqueue(c[0] as number, ci)
      }
      ws.length = keep
    }
    return -1
  }

  private bump(v: number): void {
    const a = (this.activity[v] ?? 0) + this.varInc
    this.activity[v] = a
    if (a > 1e100) {
      for (let u = 1; u <= this.n; u++) this.activity[u] = (this.activity[u] ?? 0) * 1e-100
      this.varInc *= 1e-100
      for (let u = 1; u <= this.n; u++) this.order.push([this.activity[u] ?? 0, u])
    } else this.order.push([a, v])
  }

  /** First-UIP conflict analysis with local minimisation: the learnt clause (asserting literal first), its backjump level. */
  analyze(confl: number): { learnt: number[]; backjump: number; lbd: number } {
    const out: number[] = [-1]
    let pathC = 0
    let p = -1
    let index = this.trail.length - 1
    let ci = confl
    const marked: number[] = []
    do {
      const c = this.clauses[ci] as number[]
      for (let j = p === -1 ? 0 : 1; j < c.length; j++) {
        const q = c[j] as number
        const v = q >> 1
        if (this.seen[v] === 0 && (this.level[v] ?? 0) > 0) {
          this.seen[v] = 1
          marked.push(v)
          this.bump(v)
          if ((this.level[v] ?? 0) >= this.decisionLevel) pathC++
          else out.push(q)
        }
      }
      while (this.seen[(this.trail[index] as number) >> 1] === 0) index--
      p = this.trail[index--] as number
      ci = this.reason[p >> 1] ?? -1
      this.seen[p >> 1] = 0
      pathC--
    } while (pathC > 0)
    out[0] = p ^ 1
    // Local minimisation: drop a literal whose reason's other literals are all in the clause (or at level 0).
    const minimised = out.filter((q, i) => {
      if (i === 0) return true
      const r = this.reason[q >> 1] ?? -1
      if (r < 0) return true
      const rc = this.clauses[r] as number[]
      return !rc.every((x) => x >> 1 === q >> 1 || this.seen[x >> 1] === 1 || this.level[x >> 1] === 0)
    })
    for (const v of marked) this.seen[v] = 0
    let backjump = 0
    let at = 1
    for (let i = 1; i < minimised.length; i++) {
      const lv = this.level[(minimised[i] as number) >> 1] ?? 0
      if (lv > backjump) {
        backjump = lv
        at = i
      }
    }
    if (minimised.length > 1) {
      const t = minimised[1] as number
      minimised[1] = minimised[at] as number
      minimised[at] = t
    }
    this.varInc /= this.decay
    const levels = new Set(minimised.map((q) => this.level[q >> 1]))
    return { learnt: minimised, backjump, lbd: levels.size }
  }

  /** Learn a clause after backjumping, and assert its first literal. */
  learn(lits: number[]): number {
    this.stats.learned++
    if (lits.length === 1) {
      this.enqueue(lits[0] as number, -1)
      return -1
    }
    const i = this.attach(lits, true)
    this.enqueue(lits[0] as number, i)
    return i
  }

  cancelUntil(lv: number): void {
    if (this.decisionLevel <= lv) return
    const start = this.trailLim[lv] as number
    for (let i = this.trail.length - 1; i >= start; i--) {
      const c = this.trail[i] as number
      const v = c >> 1
      this.polarity[v] = c & 1 ? 0 : 1
      this.assign[v] = 0
      this.reason[v] = -1
      this.order.push([this.activity[v] ?? 0, v])
    }
    this.trail.length = start
    this.trailLim.length = lv
    this.qhead = start
  }

  newDecisionLevel(): void {
    this.trailLim.push(this.trail.length)
  }

  /** The unassigned variable of highest activity, with its saved phase (default false), or -1. */
  pickBranch(): number {
    for (let top = this.order.pop(); top !== undefined; top = this.order.pop()) {
      const [act, v] = top
      if (this.assign[v] === 0 && act === this.activity[v]) return this.polarity[v] === 1 ? 2 * v : 2 * v + 1
    }
    for (let v = 1; v <= this.n; v++) if (this.assign[v] === 0) return this.polarity[v] === 1 ? 2 * v : 2 * v + 1
    return -1
  }

  /** The assumptions responsible for `p` being false (MiniSat analyzeFinal), as coded literals. */
  analyzeFinal(p: number): number[] {
    const out = [p]
    if (this.decisionLevel === 0) return out
    this.seen[p >> 1] = 1
    for (let i = this.trail.length - 1; i >= (this.trailLim[0] ?? 0); i--) {
      const x = this.trail[i] as number
      const v = x >> 1
      if (this.seen[v] === 0) continue
      const r = this.reason[v] ?? -1
      if (r < 0) {
        if ((this.level[v] ?? 0) > 0) out.push(x ^ 1)
      } else {
        for (const q of (this.clauses[r] as number[]).slice(1)) if ((this.level[q >> 1] ?? 0) > 0) this.seen[q >> 1] = 1
      }
      this.seen[v] = 0
    }
    this.seen[p >> 1] = 0
    return out
  }

  /** The current full assignment as signed DIMACS literals (unassigned variables reported false). */
  model(): number[] {
    return Array.from({ length: this.n }, (_, i) => ((this.assign[i + 1] ?? 0) === 1 ? i + 1 : -(i + 1)))
  }
}

/** The Luby restart sequence 1, 1, 2, 1, 1, 2, 4, ... (`i` from 0). */
export function luby(i: number): number {
  let size = 1
  let seq = 0
  while (size < i + 1) {
    seq++
    size = 2 * size + 1
  }
  let x = i
  while (size - 1 !== x) {
    size = (size - 1) >> 1
    seq--
    x %= size
  }
  return 2 ** seq
}
