import { type Json, WhiteboxError } from '../base/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { CdclCore, code } from './_cdcl.ts'
import { type Cnf, cdcl, type Model, type SatRunOptions } from './sat.ts'

// ------------------------------------------------------------------------------------------------ enumeration

/**
 * Every model of a CNF (over all its variables), found by CDCL with blocking clauses, in the order found.
 *
 * @param f - The formula.
 * @param options - At most `max` models (default 10000).
 * @returns The models.
 * @concept satisfiability
 * @example
 * ```ts
 * import { allModels, cnf } from 'whitebox-engine/logic'
 * allModels(cnf([[1, 2]])).length // => 3
 * ```
 */
export function allModels(f: Cnf, options: { readonly max?: number } = {}): Model[] {
  const max = options.max ?? 10_000
  const core = new CdclCore(f.numVars, f.clauses)
  const out: Model[] = []
  while (core.ok && out.length < max) {
    const r = solveCore(core, [])
    if (r === null) break
    out.push(r)
    core.cancelUntil(0)
    if (f.numVars === 0 || !core.addClause(r.map((l) => -l))) break
  }
  return out
}

/** Run a CDCL core to completion under assumptions: a model, or null when unsatisfiable. */
function solveCore(core: CdclCore, assumptions: readonly number[]): Model | null {
  core.cancelUntil(0)
  if (!core.ok) return null
  const codes = assumptions.map(code)
  for (;;) {
    const confl = core.propagate()
    if (confl >= 0) {
      core.stats.conflicts++
      if (core.decisionLevel === 0) {
        core.ok = false
        return null
      }
      const { learnt, backjump } = core.analyze(confl)
      core.cancelUntil(backjump)
      core.learn(learnt)
      continue
    }
    if (core.decisionLevel < codes.length) {
      const a = codes[core.decisionLevel] as number
      const v = core.value(a)
      if (v === -1) return null
      core.newDecisionLevel()
      if (v === 0) core.enqueue(a, -1)
      continue
    }
    const next = core.pickBranch()
    if (next < 0) return core.model()
    core.newDecisionLevel()
    core.stats.decisions++
    core.enqueue(next, -1)
  }
}

/**
 * The number of models of a CNF over its variables, by DPLL-style counting: unit propagation, then splitting on a
 * variable, with `2^k` models for `k` free variables once every clause is satisfied.
 *
 * @param f - The formula.
 * @returns The model count (exact up to 2^53).
 * @concept satisfiability
 * @example
 * ```ts
 * import { cnf, countModels } from 'whitebox-engine/logic'
 * [countModels(cnf([[1, 2]])), countModels(cnf([[1], [-1]])), countModels(cnf([], 3))] // => [3, 0, 8]
 * ```
 */
export function countModels(f: Cnf): number {
  const n = f.numVars
  const value = new Int8Array(n + 1)
  const lit = (l: number): number => (l > 0 ? (value[l] ?? 0) : -(value[-l] ?? 0))
  const count = (): number => {
    const trail: number[] = []
    const undo = (): void => {
      for (const v of trail) value[v] = 0
    }
    // Propagate units.
    for (let changed = true; changed; ) {
      changed = false
      for (const c of f.clauses) {
        let open = 0
        let last = 0
        let sat = false
        for (const l of c) {
          const x = lit(l)
          if (x === 1) {
            sat = true
            break
          }
          if (x === 0) {
            open++
            last = l
          }
        }
        if (sat) continue
        if (open === 0) {
          undo()
          return 0
        }
        if (open === 1) {
          value[Math.abs(last)] = last > 0 ? 1 : -1
          trail.push(Math.abs(last))
          changed = true
        }
      }
    }
    let branch = 0
    for (const c of f.clauses) {
      if (c.some((l) => lit(l) === 1)) continue
      const open = c.find((l) => lit(l) === 0)
      if (open !== undefined) {
        branch = Math.abs(open)
        break
      }
    }
    let result: number
    if (branch === 0) {
      let free = 0
      for (let v = 1; v <= n; v++) if (value[v] === 0) free++
      result = 2 ** free
    } else {
      value[branch] = 1
      const a = count()
      value[branch] = -1
      const b = count()
      value[branch] = 0
      result = a + b
    }
    undo()
    return result
  }
  if (f.clauses.some((c) => c.length === 0)) return 0
  return count()
}

// ------------------------------------------------------------------------------------------------ cores

/**
 * A minimal unsatisfiable subset (MUS) of an unsatisfiable CNF: clause indices such that those clauses are
 * unsatisfiable and removing any one makes them satisfiable. Each clause gets a selector variable; CDCL under the
 * selectors as assumptions yields a core, which deletion-based minimisation then shrinks to a MUS.
 *
 * @param f - The formula.
 * @returns The indices (0-based, ascending) of a MUS, or null when the formula is satisfiable.
 * @concept satisfiability
 * @example
 * ```ts
 * import { cnf, unsatCore } from 'whitebox-engine/logic'
 * unsatCore(cnf([[1, 2], [-1], [3], [-2], [-3, 1]])) // => [0, 1, 3]
 * unsatCore(cnf([[1, 2]])) // => null
 * ```
 */
export function unsatCore(f: Cnf): number[] | null {
  const n = f.numVars
  const m = f.clauses.length
  const empty = f.clauses.findIndex((c) => c.length === 0)
  if (empty >= 0) return [empty]
  const selector = (i: number): number => n + 1 + i
  const relaxed: Cnf = { numVars: n + m, clauses: f.clauses.map((c, i) => [...c, -selector(i)]) }
  // The clauses (a subset of `set`) whose selectors fail together, or null when `set` is satisfiable.
  const failed = (set: readonly number[]): number[] | null => {
    const r = cdcl(relaxed, { assumptions: set.map(selector) })
    if (r.status !== 'unsat') return null
    const inSet = new Set(set)
    return (r.failedAssumptions ?? [])
      .map((s) => s - n - 1)
      .filter((i) => inSet.has(i))
      .sort((a, b) => a - b)
  }
  let current = failed(Array.from({ length: m }, (_, i) => i))
  if (current === null) return null
  // Deletion-based minimisation: a clause stays only if removing it makes the rest satisfiable.
  for (let k = 0; k < current.length; ) {
    const without = current.filter((_, j) => j !== k)
    const smaller = failed(without)
    if (smaller === null) k++
    else current = smaller
  }
  return current
}

// ------------------------------------------------------------------------------------------------ MaxSAT

/** A weighted clause of a MaxSAT problem. */
export interface SoftClause {
  readonly clause: readonly number[]
  readonly weight: number
}

/** A weighted partial MaxSAT problem: hard clauses that must hold and soft clauses whose falsified weight is minimised. */
export interface MaxSatProblem {
  readonly numVars: number
  readonly hard: readonly (readonly number[])[]
  readonly soft: readonly SoftClause[]
}

/** The outcome of {@link maxsat}. */
export interface MaxSatResult {
  /** `optimal`, `unsat` (the hard clauses alone are unsatisfiable) or `unknown` (budget). */
  readonly status: 'optimal' | 'unsat' | 'unknown'
  readonly model: Model | null
  /** The total weight of falsified soft clauses (Infinity when unsatisfiable). */
  readonly cost: number
  /** Nodes of the branch-and-bound tree explored. */
  readonly nodes: number
}

/** The observable state of a MaxSAT branch and bound. */
export interface MaxSatState {
  readonly status: 'running' | 'optimal' | 'unsat'
  readonly best: number
  readonly bestModel: Model | null
  readonly nodes: number
  readonly depth: number
  readonly event: { readonly kind: string; readonly cost: number } | null
}

interface MaxSatMachine extends MaxSatState {
  status: 'running' | 'optimal' | 'unsat'
  best: number
  bestModel: Model | null
  nodes: number
  depth: number
  event: { kind: string; cost: number } | null
  readonly value: Int8Array
  readonly stack: { var: number; tried: number; trail: number }[]
  readonly trail: number[]
}

const MAXSAT_SCHEMA: TraceSchema = channels({
  metrics: {
    best: { description: 'weight of the best assignment found', direction: 'lower' },
    nodes: { description: 'branch-and-bound nodes' },
    depth: { description: 'variables branched on' },
  },
  messages: {
    improve: { description: 'a better assignment', payload: { cost: 'its falsified weight' } },
    prune: { description: 'a branch cut by the bound or a hard conflict', payload: { cost: 'lower bound' } },
  },
})

function maxsatMachine(p: MaxSatProblem): Algorithm<MaxSatMachine> {
  const n = p.numVars
  const lit = (s: MaxSatMachine, l: number): number => (l > 0 ? (s.value[l] ?? 0) : -(s.value[-l] ?? 0))
  // Propagate hard units; false on a hard conflict.
  const propagate = (s: MaxSatMachine): boolean => {
    for (let changed = true; changed; ) {
      changed = false
      for (const c of p.hard) {
        let open = 0
        let last = 0
        let sat = false
        for (const l of c) {
          const x = lit(s, l)
          if (x === 1) {
            sat = true
            break
          }
          if (x === 0) {
            open++
            last = l
          }
        }
        if (sat) continue
        if (open === 0) return false
        if (open === 1) {
          s.value[Math.abs(last)] = last > 0 ? 1 : -1
          s.trail.push(Math.abs(last))
          changed = true
        }
      }
    }
    return true
  }
  // A lower bound: the weight of soft clauses already falsified.
  const bound = (s: MaxSatMachine): number => {
    let cost = 0
    for (const c of p.soft) if (c.clause.every((l) => lit(s, l) === -1)) cost += c.weight
    return cost
  }
  const undoTo = (s: MaxSatMachine, mark: number): void => {
    while (s.trail.length > mark) s.value[s.trail.pop() as number] = 0
  }
  const nextVar = (s: MaxSatMachine): number => {
    for (let v = 1; v <= n; v++) if (s.value[v] === 0) return v
    return 0
  }
  // Explore the node reached by the last assignment: prune, record a leaf, or push a branching frame.
  const visit = (s: MaxSatMachine): void => {
    s.nodes++
    if (!propagate(s)) {
      s.event = { kind: 'prune', cost: Number.POSITIVE_INFINITY }
      return
    }
    const lb = bound(s)
    if (lb >= s.best) {
      s.event = { kind: 'prune', cost: lb }
      return
    }
    const v = nextVar(s)
    if (v === 0) {
      s.best = lb
      s.bestModel = Array.from({ length: n }, (_, i) => ((s.value[i + 1] ?? 0) === 1 ? i + 1 : -(i + 1)))
      s.event = { kind: 'improve', cost: lb }
      return
    }
    s.stack.push({ var: v, tried: 0, trail: s.trail.length })
  }
  return {
    name: 'maxsat-branch-and-bound',
    config: { hard: p.hard.length, soft: p.soft.length },
    schema: MAXSAT_SCHEMA,
    init: () => {
      const s: MaxSatMachine = {
        status: 'running',
        best: Number.POSITIVE_INFINITY,
        bestModel: null,
        nodes: 0,
        depth: 0,
        event: null,
        value: new Int8Array(n + 1),
        stack: [],
        trail: [],
      }
      visit(s)
      if (s.stack.length === 0) s.status = s.bestModel === null ? 'unsat' : 'optimal'
      return s
    },
    step: (s) => {
      s.event = null
      const top = s.stack[s.stack.length - 1]
      if (top === undefined) {
        s.status = s.bestModel === null ? 'unsat' : 'optimal'
        return s
      }
      undoTo(s, top.trail)
      if (top.tried >= 2) {
        s.stack.pop()
        s.depth = s.stack.length
        return s
      }
      // Try true first, then false (the order of a polarity-false solver is irrelevant for optimality).
      const value = top.tried === 0 ? 1 : -1
      top.tried++
      s.value[top.var] = value
      s.trail.push(top.var)
      visit(s)
      s.depth = s.stack.length
      return s
    },
    done: (s) => (s.status === 'running' ? undefined : 'converged'),
    observe: (s) => ({
      metrics: { best: Number.isFinite(s.best) ? s.best : -1, nodes: s.nodes, depth: s.depth },
      messages:
        s.event === null
          ? []
          : [{ kind: s.event.kind, payload: { cost: Number.isFinite(s.event.cost) ? s.event.cost : null } as Json }],
    }),
  }
}

function checkMaxSat(p: MaxSatProblem): void {
  for (const c of [...p.hard, ...p.soft.map((s) => s.clause)]) {
    for (const l of c) {
      if (!Number.isInteger(l) || l === 0 || Math.abs(l) > p.numVars) {
        throw new WhiteboxError('value', `maxsat: invalid literal ${l}`)
      }
    }
  }
  for (const s of p.soft) {
    if (!(s.weight > 0) || !Number.isFinite(s.weight))
      throw new WhiteboxError('value', 'maxsat: weights must be positive')
  }
}

/**
 * Weighted partial MaxSAT by depth-first branch and bound as a traced {@link Algorithm}: hard clauses are propagated,
 * branches whose falsified soft weight already reaches the best found are pruned, and every improvement is recorded.
 * Exact; exponential in the worst case, so meant for small instances.
 *
 * @param p - The problem.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` for an invalid literal or a non-positive weight.
 * @concept satisfiability
 * @example
 * ```ts
 * import { maxsatAlgorithm } from 'whitebox-engine/logic'
 * maxsatAlgorithm({ numVars: 1, hard: [], soft: [{ clause: [1], weight: 1 }] }).name // => 'maxsat-branch-and-bound'
 * ```
 */
export function maxsatAlgorithm(p: MaxSatProblem): Algorithm<MaxSatState> {
  checkMaxSat(p)
  return maxsatMachine(p)
}

/**
 * Solve a weighted partial MaxSAT problem exactly: an assignment satisfying every hard clause that minimises the
 * total weight of falsified soft clauses.
 *
 * @param p - The problem.
 * @param options - A node budget (default 10000000) and tracing.
 * @returns The status, an optimal model and its cost, and the nodes explored.
 * @throws WhiteboxError with code `value` for an invalid literal or a non-positive weight.
 * @concept satisfiability
 * @example
 * ```ts
 * import { maxsat } from 'whitebox-engine/logic'
 * // Choose at most one of x1, x2 (hard), preferring x1 (weight 3) over x2 (weight 2), and x3 unless x1 (weight 1).
 * const r = maxsat({ numVars: 3, hard: [[-1, -2]], soft: [{ clause: [1], weight: 3 }, { clause: [2], weight: 2 }, { clause: [3], weight: 1 }, { clause: [-1, -3], weight: 1 }] })
 * [r.status, r.cost] // => ['optimal', 3]
 * ```
 */
export function maxsat(p: MaxSatProblem, options: SatRunOptions & { readonly maxNodes?: number } = {}): MaxSatResult {
  checkMaxSat(p)
  const { maxNodes = 10_000_000, ...run } = options
  const { state } = runAlgorithm(maxsatMachine(p), { ...run, maxSteps: 3 * maxNodes })
  const status = state.status === 'running' ? 'unknown' : state.status
  return {
    status,
    model: state.bestModel,
    cost: status === 'unsat' ? Number.POSITIVE_INFINITY : state.best,
    nodes: state.nodes,
  }
}
