/**
 * Boolean satisfiability (SAT) solvers on CNF formulas in DIMACS form: reading and writing DIMACS, checking
 * assignments, and three solvers as traced algorithms: DPLL with propagation and branching heuristics, conflict-driven
 * clause learning in MiniSat's architecture (watched literals, first-UIP learning, restarts), and the WalkSAT local
 * search.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { Generator } from '../random/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { CdclCore, type CdclStats, code, dimacs, luby } from './_cdcl.ts'

/**
 * A formula in conjunctive normal form, DIMACS style: variables `1..numVars`, each clause a disjunction of non-zero
 * literals (`v` or `-v`).
 *
 * @concept satisfiability
 */
export interface Cnf {
  readonly numVars: number
  readonly clauses: readonly (readonly number[])[]
}

/** A satisfying assignment as signed literals, one per variable in order (`[1, -2, 3]`: x1 and x3 true, x2 false). */
export type Model = readonly number[]

/** Search effort of a SAT solver. */
export interface SatStats {
  readonly decisions: number
  readonly propagations: number
  readonly conflicts: number
  readonly learned: number
  readonly restarts: number
  readonly backtracks: number
  readonly flips: number
}

/** The outcome of a SAT solver: `unknown` when a budget ran out (or an incomplete method gave up). */
export interface SatResult {
  readonly status: 'sat' | 'unsat' | 'unknown'
  readonly model: Model | null
  readonly stats: SatStats
  /** For CDCL under assumptions: the assumptions that together are unsatisfiable (a subset of those given). */
  readonly failedAssumptions?: readonly number[]
}

/** Options shared by the solvers: tracing and a seed or key (WalkSAT). */
export type SatRunOptions = Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId' | 'seed' | 'key'>>

/**
 * Build a CNF, checking literals (non-zero integers) and taking the variable count from them unless given.
 *
 * @param clauses - The clauses.
 * @param numVars - The number of variables (at least the largest variable used).
 * @returns The formula.
 * @throws MdmlError with code `value` for a zero or non-integer literal, or too small a variable count.
 * @concept satisfiability
 * @example
 * ```ts
 * import { cnf } from 'mdml-engine/logic'
 * cnf([[1, -2], [2, 3]]).numVars // => 3
 * ```
 */
export function cnf(clauses: readonly (readonly number[])[], numVars?: number): Cnf {
  let n = 0
  for (const c of clauses) {
    for (const l of c) {
      if (!Number.isInteger(l) || l === 0) throw new MdmlError('value', `cnf: invalid literal ${l}`)
      n = Math.max(n, Math.abs(l))
    }
  }
  if (numVars !== undefined && numVars < n) throw new MdmlError('value', `cnf: numVars ${numVars} < ${n}`)
  return { numVars: numVars ?? n, clauses: clauses.map((c) => [...c]) }
}

/**
 * Read DIMACS CNF text (`c` comment lines, a `p cnf <vars> <clauses>` header, clauses terminated by `0`).
 *
 * @param text - The DIMACS text.
 * @returns The formula.
 * @throws MdmlError with code `value` for a missing or malformed header or an unterminated clause.
 * @example
 * ```ts
 * import { parseDimacs } from 'mdml-engine/logic'
 * parseDimacs('c example\np cnf 3 2\n1 -2 0\n2 3 0\n').clauses // => [[1, -2], [2, 3]]
 * ```
 */
export function parseDimacs(text: string): Cnf {
  let numVars = -1
  const clauses: number[][] = []
  let current: number[] = []
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (line === '' || line.startsWith('c') || line.startsWith('%')) continue
    if (line.startsWith('p')) {
      const parts = line.split(/\s+/)
      if (parts[1] !== 'cnf' || parts.length < 4) throw new MdmlError('value', `parseDimacs: bad header "${line}"`)
      numVars = Number(parts[2])
      continue
    }
    for (const tok of line.split(/\s+/)) {
      const l = Number(tok)
      if (!Number.isInteger(l)) throw new MdmlError('value', `parseDimacs: bad literal "${tok}"`)
      if (l === 0) {
        clauses.push(current)
        current = []
      } else current.push(l)
    }
  }
  if (numVars < 0) throw new MdmlError('value', 'parseDimacs: missing "p cnf" header')
  if (current.length > 0) throw new MdmlError('value', 'parseDimacs: last clause is not terminated by 0')
  return cnf(clauses, numVars)
}

/**
 * Write a CNF as DIMACS text.
 *
 * @param f - The formula.
 * @param comments - Comment lines to put first.
 * @returns The text.
 * @example
 * ```ts
 * import { cnf, toDimacs } from 'mdml-engine/logic'
 * toDimacs(cnf([[1, -2]])) // => 'p cnf 2 1\n1 -2 0\n'
 * ```
 */
export function toDimacs(f: Cnf, comments: readonly string[] = []): string {
  const head = comments.map((c) => `c ${c}\n`).join('')
  return `${head}p cnf ${f.numVars} ${f.clauses.length}\n${f.clauses.map((c) => `${[...c, 0].join(' ')}\n`).join('')}`
}

/**
 * Whether an assignment satisfies every clause.
 *
 * @param f - The formula.
 * @param model - Signed literals (missing variables count as false).
 * @returns True when every clause has a true literal.
 * @example
 * ```ts
 * import { cnf, satisfies } from 'mdml-engine/logic'
 * [satisfies(cnf([[1, -2], [2]]), [1, 2]), satisfies(cnf([[1, -2], [2]]), [-1, 2])] // => [true, false]
 * ```
 */
export function satisfies(f: Cnf, model: Model): boolean {
  const truth = new Set(model.filter((l) => l > 0))
  return f.clauses.every((c) => c.some((l) => (l > 0 ? truth.has(l) : !truth.has(-l))))
}

function zeroStats(): { -readonly [K in keyof SatStats]: SatStats[K] } {
  return { decisions: 0, propagations: 0, conflicts: 0, learned: 0, restarts: 0, backtracks: 0, flips: 0 }
}

// ------------------------------------------------------------------------------------------------ DPLL

/** Branching rules of {@link dpll}: first unassigned variable, DLIS (most occurrences in open clauses), Jeroslow-Wang. */
export type Branching = 'first' | 'dlis' | 'jeroslow-wang'

/** Options of {@link dpll}. */
export interface DpllOptions extends SatRunOptions {
  readonly branching?: Branching
  /** Assign pure literals (default true). */
  readonly pureLiterals?: boolean
  /** Stop with status `unknown` after this many decisions (default 1000000). */
  readonly maxDecisions?: number
}

/** A step of a solver as recorded for the trace. */
export interface SatEvent {
  readonly kind: 'decide' | 'propagate' | 'pure' | 'conflict' | 'backtrack' | 'learn' | 'restart' | 'flip' | 'solution'
  readonly literals: readonly number[]
  readonly level?: number
}

/** The observable state of a DPLL run. */
export interface DpllState {
  readonly status: 'running' | 'sat' | 'unsat'
  readonly level: number
  readonly assigned: number
  readonly stats: SatStats
  readonly events: readonly SatEvent[]
  readonly model: Model | null
}

interface DpllFrame {
  readonly literal: number
  readonly trail: number
  flipped: boolean
}

interface DpllMachine extends DpllState {
  status: 'running' | 'sat' | 'unsat'
  level: number
  assigned: number
  stats: ReturnType<typeof zeroStats>
  events: SatEvent[]
  model: Model | null
  readonly value: Int8Array
  readonly trail: number[]
  readonly stack: DpllFrame[]
}

const SAT_SCHEMA: TraceSchema = channels({
  metrics: {
    decisions: { description: 'branching decisions' },
    propagations: { description: 'literals assigned by unit propagation' },
    conflicts: { description: 'clauses falsified' },
    learned: { description: 'clauses learned' },
    level: { description: 'decision level' },
    assigned: { description: 'variables assigned' },
  },
  messages: {
    decide: { description: 'a branching decision', payload: { literals: 'the decided literal', level: 'level' } },
    propagate: { description: 'unit propagation', payload: { literals: 'literals implied' } },
    pure: { description: 'pure literal elimination', payload: { literals: 'literals assigned' } },
    conflict: { description: 'a falsified clause', payload: { literals: 'the clause' } },
    backtrack: {
      description: 'undo to a decision and flip it',
      payload: { literals: 'the flipped literal', level: 'level' },
    },
    learn: {
      description: 'a learned clause and the level jumped back to',
      payload: { literals: 'clause', level: 'backjump level' },
    },
    restart: { description: 'a restart', payload: { literals: '[]' } },
    flip: { description: 'a local-search flip', payload: { literals: 'the variable flipped (signed new value)' } },
    solution: { description: 'a satisfying assignment', payload: { literals: 'the model' } },
  },
})

function observeSat(
  s: { readonly stats: SatStats; readonly events: readonly SatEvent[] },
  level: number,
  assigned: number,
) {
  return {
    metrics: {
      decisions: s.stats.decisions,
      propagations: s.stats.propagations,
      conflicts: s.stats.conflicts,
      learned: s.stats.learned,
      level,
      assigned,
    },
    messages: s.events.map((e) => ({
      kind: e.kind,
      payload: { literals: [...e.literals], ...(e.level === undefined ? {} : { level: e.level }) } as Json,
    })),
  }
}

function dpllMachine(f: Cnf, options: DpllOptions): Algorithm<DpllMachine> {
  const branching = options.branching ?? 'first'
  const pure = options.pureLiterals ?? true
  const maxDecisions = options.maxDecisions ?? 1_000_000
  const n = f.numVars
  const clauses = f.clauses
  const litValue = (s: DpllMachine, l: number): number => {
    const v = s.value[Math.abs(l)] ?? 0
    return l > 0 ? v : -v
  }
  const set = (s: DpllMachine, l: number): void => {
    s.value[Math.abs(l)] = l > 0 ? 1 : -1
    s.trail.push(l)
  }
  // Unit propagation to a fixpoint by scanning the clauses: the conflicting clause, or null.
  const propagate = (s: DpllMachine): readonly number[] | null => {
    const implied: number[] = []
    for (let changed = true; changed; ) {
      changed = false
      for (const c of clauses) {
        let unassigned = 0
        let last = 0
        let sat = false
        for (const l of c) {
          const v = litValue(s, l)
          if (v === 1) {
            sat = true
            break
          }
          if (v === 0) {
            unassigned++
            last = l
          }
        }
        if (sat) continue
        if (unassigned === 0) {
          if (implied.length > 0) s.events.push({ kind: 'propagate', literals: implied })
          return c
        }
        if (unassigned === 1) {
          set(s, last)
          implied.push(last)
          s.stats.propagations++
          changed = true
        }
      }
    }
    if (implied.length > 0) s.events.push({ kind: 'propagate', literals: implied })
    if (pure) {
      const seenPos = new Uint8Array(n + 1)
      const seenNeg = new Uint8Array(n + 1)
      for (const c of clauses) {
        if (c.some((l) => litValue(s, l) === 1)) continue
        for (const l of c) {
          if (litValue(s, l) !== 0) continue
          if (l > 0) seenPos[l] = 1
          else seenNeg[-l] = 1
        }
      }
      const pures: number[] = []
      for (let v = 1; v <= n; v++) {
        if (s.value[v] !== 0) continue
        if (seenPos[v] === 1 && seenNeg[v] === 0) pures.push(v)
        else if (seenNeg[v] === 1 && seenPos[v] === 0) pures.push(-v)
      }
      for (const l of pures) set(s, l)
      if (pures.length > 0) s.events.push({ kind: 'pure', literals: pures })
    }
    return null
  }
  const allSatisfied = (s: DpllMachine): boolean => clauses.every((c) => c.some((l) => litValue(s, l) === 1))
  const choose = (s: DpllMachine): number => {
    if (branching === 'first') {
      for (let v = 1; v <= n; v++) if (s.value[v] === 0) return v
      return 0
    }
    const score = new Float64Array(2 * n + 2)
    for (const c of clauses) {
      if (c.some((l) => litValue(s, l) === 1)) continue
      const open = c.filter((l) => litValue(s, l) === 0)
      const w = branching === 'dlis' ? 1 : 2 ** -open.length
      for (const l of open) score[code(l)] = (score[code(l)] ?? 0) + w
    }
    let best = 0
    let bestScore = -1
    for (let v = 1; v <= n; v++) {
      if (s.value[v] !== 0) continue
      for (const l of [v, -v]) {
        const sc = score[code(l)] ?? 0
        if (sc > bestScore) {
          bestScore = sc
          best = l
        }
      }
    }
    return best
  }
  const undoTo = (s: DpllMachine, mark: number): void => {
    while (s.trail.length > mark) s.value[Math.abs(s.trail.pop() as number)] = 0
  }
  const finish = (s: DpllMachine): void => {
    s.status = 'sat'
    s.model = Array.from({ length: n }, (_, i) => ((s.value[i + 1] ?? 0) === 1 ? i + 1 : -(i + 1)))
    s.events.push({ kind: 'solution', literals: s.model })
  }
  // After a conflict: undo to the deepest unflipped decision and take its other branch.
  const backtrack = (s: DpllMachine): void => {
    for (;;) {
      const top = s.stack[s.stack.length - 1]
      if (top === undefined) {
        s.status = 'unsat'
        return
      }
      undoTo(s, top.trail)
      if (top.flipped) {
        s.stack.pop()
        continue
      }
      top.flipped = true
      s.stats.backtracks++
      set(s, -top.literal)
      s.level = s.stack.length
      s.events.push({ kind: 'backtrack', literals: [-top.literal], level: s.level })
      return
    }
  }
  return {
    name: 'dpll',
    config: { branching, pureLiterals: pure },
    schema: SAT_SCHEMA,
    init: () => {
      const s: DpllMachine = {
        status: 'running',
        level: 0,
        assigned: 0,
        stats: zeroStats(),
        events: [],
        model: null,
        value: new Int8Array(n + 1),
        trail: [],
        stack: [],
      }
      if (clauses.some((c) => c.length === 0)) s.status = 'unsat'
      return s
    },
    step: (s) => {
      s.events = []
      const conflict = propagate(s)
      if (conflict !== null) {
        s.stats.conflicts++
        s.events.push({ kind: 'conflict', literals: conflict })
        backtrack(s)
      } else if (allSatisfied(s)) finish(s)
      else {
        const l = choose(s)
        s.stats.decisions++
        s.stack.push({ literal: l, trail: s.trail.length, flipped: false })
        set(s, l)
        s.level = s.stack.length
        s.events.push({ kind: 'decide', literals: [l], level: s.level })
      }
      s.assigned = s.trail.length
      return s
    },
    done: (s) => (s.status !== 'running' ? 'converged' : s.stats.decisions >= maxDecisions ? 'stopped' : undefined),
    observe: (s) => observeSat(s, s.level, s.assigned),
  }
}

/**
 * The Davis-Putnam-Logemann-Loveland procedure as a traced {@link Algorithm}: unit propagation and pure-literal
 * elimination to a fixpoint, then a branching decision; on a conflict, chronological backtracking flips the most recent
 * unflipped decision. Each step propagates and then decides or backtracks.
 *
 * @param f - The formula.
 * @param options - Branching rule, pure literals and a decision budget.
 * @returns The algorithm.
 * @concept satisfiability
 * @example
 * ```ts
 * import { cnf, dpllAlgorithm } from 'mdml-engine/logic'
 * dpllAlgorithm(cnf([[1]])).name // => 'dpll'
 * ```
 */
export function dpllAlgorithm(f: Cnf, options: DpllOptions = {}): Algorithm<DpllState> {
  return dpllMachine(f, options)
}

/**
 * Decide satisfiability with DPLL.
 *
 * @param f - The formula.
 * @param options - Branching rule, pure literals, budget and tracing.
 * @returns The status, a model when satisfiable, and the effort.
 * @concept satisfiability
 * @example
 * ```ts
 * import { cnf, dpll } from 'mdml-engine/logic'
 * dpll(cnf([[1, 2], [-1, 2], [1, -2]])).model // => [1, 2]
 * dpll(cnf([[1, 2], [-1, 2], [1, -2], [-1, -2]])).status // => 'unsat'
 * ```
 */
export function dpll(f: Cnf, options: DpllOptions = {}): SatResult {
  const { branching: _b, pureLiterals: _p, maxDecisions = 1_000_000, ...run } = options
  const { state } = runAlgorithm(dpllMachine(f, options), { ...run, maxSteps: 4 * maxDecisions + 4 * f.numVars + 8 })
  return {
    status: state.status === 'running' ? 'unknown' : state.status,
    model: state.model,
    stats: { ...state.stats },
  }
}

// ------------------------------------------------------------------------------------------------ CDCL

/** Options of {@link cdcl}. */
export interface CdclOptions extends SatRunOptions {
  /** Literals assumed true for this call (MiniSat assumptions). */
  readonly assumptions?: readonly number[]
  /** Restart policy: Luby sequence times `restartBase` conflicts (default), or never. */
  readonly restarts?: 'luby' | 'none'
  readonly restartBase?: number
  /** VSIDS activity decay (default 0.95). */
  readonly decay?: number
  /** Stop with status `unknown` after this many conflicts (default 1000000). */
  readonly maxConflicts?: number
}

/** The observable state of a CDCL run. */
export interface CdclState {
  readonly status: 'running' | 'sat' | 'unsat'
  readonly level: number
  readonly assigned: number
  readonly stats: SatStats
  readonly events: readonly SatEvent[]
  readonly model: Model | null
  /** Clauses learned so far, as DIMACS literals. */
  readonly learnedClauses: readonly (readonly number[])[]
  readonly failedAssumptions: readonly number[] | null
}

interface CdclMachine extends CdclState {
  status: 'running' | 'sat' | 'unsat'
  level: number
  assigned: number
  events: SatEvent[]
  model: Model | null
  learnedClauses: number[][]
  failedAssumptions: number[] | null
  stats: ReturnType<typeof zeroStats>
  readonly core: CdclCore
  sinceRestart: number
  restartIndex: number
}

function syncStats(s: CdclMachine): void {
  const c: CdclStats = s.core.stats
  s.stats.decisions = c.decisions
  s.stats.propagations = c.propagations
  s.stats.conflicts = c.conflicts
  s.stats.learned = c.learned
  s.stats.restarts = c.restarts
}

function cdclMachine(f: Cnf, options: CdclOptions): Algorithm<CdclMachine> {
  const assumptions = (options.assumptions ?? []).map(code)
  const restarts = options.restarts ?? 'luby'
  const base = options.restartBase ?? 100
  const maxConflicts = options.maxConflicts ?? 1_000_000
  for (const a of options.assumptions ?? []) {
    if (!Number.isInteger(a) || a === 0 || Math.abs(a) > f.numVars) {
      throw new MdmlError('value', `cdcl: assumption ${a} is not a literal of the formula`)
    }
  }
  return {
    name: 'cdcl',
    config: {
      restarts,
      restartBase: base,
      decay: options.decay ?? 0.95,
      assumptions: options.assumptions?.length ?? 0,
    },
    schema: SAT_SCHEMA,
    init: () => {
      const core = new CdclCore(f.numVars, f.clauses, options.decay ?? 0.95)
      const s: CdclMachine = {
        status: core.ok ? 'running' : 'unsat',
        level: 0,
        assigned: core.trail.length,
        stats: zeroStats(),
        events: [],
        model: null,
        learnedClauses: [],
        failedAssumptions: core.ok ? null : [],
        core,
        sinceRestart: 0,
        restartIndex: 0,
      }
      return s
    },
    step: (s) => {
      s.events = []
      const core = s.core
      const before = core.trail.length
      const confl = core.propagate()
      const implied = core.trail.slice(before).map(dimacs)
      if (implied.length > 0) s.events.push({ kind: 'propagate', literals: implied })
      if (confl >= 0) {
        core.stats.conflicts++
        s.sinceRestart++
        s.events.push({ kind: 'conflict', literals: (core.clauses[confl] ?? []).map(dimacs) })
        if (core.decisionLevel === 0) {
          s.status = 'unsat'
          s.failedAssumptions = []
        } else {
          const { learnt, backjump } = core.analyze(confl)
          core.cancelUntil(backjump)
          core.learn(learnt)
          const clause = learnt.map(dimacs)
          s.learnedClauses.push(clause)
          s.events.push({ kind: 'learn', literals: clause, level: backjump })
          if (restarts === 'luby' && s.sinceRestart >= base * luby(s.restartIndex)) {
            core.cancelUntil(0)
            core.stats.restarts++
            s.restartIndex++
            s.sinceRestart = 0
            s.events.push({ kind: 'restart', literals: [] })
          }
        }
      } else {
        let decided = false
        while (core.decisionLevel < assumptions.length) {
          const a = assumptions[core.decisionLevel] as number
          const v = core.value(a)
          if (v === 1) core.newDecisionLevel()
          else if (v === -1) {
            s.status = 'unsat'
            s.failedAssumptions = core.analyzeFinal(a ^ 1).map((x) => dimacs(x ^ 1))
            decided = true
            break
          } else {
            core.newDecisionLevel()
            core.stats.decisions++
            core.enqueue(a, -1)
            s.events.push({ kind: 'decide', literals: [dimacs(a)], level: core.decisionLevel })
            decided = true
            break
          }
        }
        if (!decided) {
          const next = core.pickBranch()
          if (next < 0) {
            s.status = 'sat'
            s.model = core.model()
            s.events.push({ kind: 'solution', literals: s.model })
          } else {
            core.newDecisionLevel()
            core.stats.decisions++
            core.enqueue(next, -1)
            s.events.push({ kind: 'decide', literals: [dimacs(next)], level: core.decisionLevel })
          }
        }
      }
      s.level = core.decisionLevel
      s.assigned = core.trail.length
      syncStats(s)
      return s
    },
    done: (s) =>
      s.status !== 'running' ? 'converged' : s.core.stats.conflicts >= maxConflicts ? 'stopped' : undefined,
    observe: (s) => observeSat(s, s.level, s.assigned),
  }
}

/**
 * Conflict-driven clause learning as a traced {@link Algorithm} (MiniSat's architecture): two watched literals for unit
 * propagation, first-UIP conflict analysis with clause minimisation, non-chronological backjumping, VSIDS variable
 * activities with phase saving, Luby restarts, and assumptions with failed-assumption analysis. Each step propagates
 * and then either analyses a conflict (learning a clause and backjumping) or makes a decision, so the trace shows every
 * decision, implication, conflict and learned clause.
 *
 * @param f - The formula.
 * @param options - Assumptions, restarts, decay and a conflict budget.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for an assumption that is not a literal of the formula.
 * @concept satisfiability
 * @reference N. Eén and N. Sörensson, An extensible SAT-solver, SAT 2003.
 * @example
 * ```ts
 * import { cdclAlgorithm, cnf } from 'mdml-engine/logic'
 * cdclAlgorithm(cnf([[1, 2]])).name // => 'cdcl'
 * ```
 */
export function cdclAlgorithm(f: Cnf, options: CdclOptions = {}): Algorithm<CdclState> {
  return cdclMachine(f, options)
}

/**
 * Decide satisfiability with CDCL.
 *
 * @param f - The formula.
 * @param options - Assumptions, restarts, decay, a conflict budget and tracing.
 * @returns The status, a model when satisfiable, the failed assumptions when unsatisfiable under assumptions, and the
 *   effort.
 * @throws MdmlError with code `value` for an assumption that is not a literal of the formula.
 * @concept satisfiability
 * @example
 * ```ts
 * import { cdcl, cnf } from 'mdml-engine/logic'
 * // The pigeonhole principle: 3 pigeons do not fit in 2 holes (x_ph: pigeon p in hole h).
 * const x = (p, h) => 2 * p + h + 1
 * const php = cnf([[x(0, 0), x(0, 1)], [x(1, 0), x(1, 1)], [x(2, 0), x(2, 1)],
 *   [-x(0, 0), -x(1, 0)], [-x(0, 0), -x(2, 0)], [-x(1, 0), -x(2, 0)],
 *   [-x(0, 1), -x(1, 1)], [-x(0, 1), -x(2, 1)], [-x(1, 1), -x(2, 1)]])
 * cdcl(php).status // => 'unsat'
 * cdcl(cnf([[1, 2], [-1, 2]]), { assumptions: [-2] }).failedAssumptions // => [-2]
 * ```
 */
export function cdcl(f: Cnf, options: CdclOptions = {}): SatResult {
  const { assumptions: _a, restarts: _r, restartBase: _b, decay: _d, maxConflicts = 1_000_000, ...run } = options
  const { state } = runAlgorithm(cdclMachine(f, options), {
    ...run,
    maxSteps: 2 * maxConflicts + 2 * (f.numVars + 1) * (maxConflicts + 1) + 8,
  })
  const status = state.status === 'running' ? 'unknown' : state.status
  return {
    status,
    model: state.model,
    stats: { ...state.stats },
    ...(status === 'unsat' && (options.assumptions?.length ?? 0) > 0
      ? { failedAssumptions: state.failedAssumptions ?? [] }
      : {}),
  }
}

// ------------------------------------------------------------------------------------------------ WalkSAT

/** Options of {@link walksat}. */
export interface WalksatOptions extends SatRunOptions {
  /** Probability of a random walk move rather than a greedy one (default 0.5). */
  readonly noise?: number
  /** Flips before giving up (default 100000). */
  readonly maxFlips?: number
}

/** The observable state of a WalkSAT run. */
export interface WalksatState {
  readonly status: 'running' | 'sat'
  readonly assignment: readonly boolean[]
  readonly unsatisfied: number
  readonly stats: SatStats
  readonly events: readonly SatEvent[]
}

interface WalksatMachine extends WalksatState {
  status: 'running' | 'sat'
  assignment: boolean[]
  unsatisfied: number
  events: SatEvent[]
  stats: ReturnType<typeof zeroStats>
}

function walksatMachine(f: Cnf, options: WalksatOptions): Algorithm<WalksatMachine> {
  const noise = options.noise ?? 0.5
  const maxFlips = options.maxFlips ?? 100_000
  if (!(noise >= 0 && noise <= 1)) throw new MdmlError('value', 'walksat: noise must be in [0, 1]')
  const n = f.numVars
  const trueLit = (a: readonly boolean[], l: number): boolean => (l > 0 ? a[l] === true : a[-l] === false)
  const unsatClauses = (a: readonly boolean[]): (readonly number[])[] =>
    f.clauses.filter((c) => !c.some((l) => trueLit(a, l)))
  // The break count of flipping v: clauses satisfied now only by v's literal.
  const breaks = (a: boolean[], v: number): number => {
    let count = 0
    for (const c of f.clauses) {
      let trueCount = 0
      let onlyV = false
      for (const l of c) {
        if (trueLit(a, l)) {
          trueCount++
          onlyV = Math.abs(l) === v
        }
      }
      if (trueCount === 1 && onlyV) count++
    }
    return count
  }
  return {
    name: 'walksat',
    config: { noise, maxFlips },
    schema: SAT_SCHEMA,
    init: (key) => {
      const u = new Generator(key).uniform([n + 1]).data
      const assignment = Array.from({ length: n + 1 }, (_, i) => i > 0 && (u[i] ?? 0) < 0.5)
      const unsat = unsatClauses(assignment)
      return {
        status: unsat.length === 0 && !f.clauses.some((c) => c.length === 0) ? 'sat' : 'running',
        assignment,
        unsatisfied: unsat.length,
        stats: zeroStats(),
        events: [],
      }
    },
    step: (s, ctx) => {
      s.events = []
      const unsat = unsatClauses(s.assignment)
      const [r0, r1, r2] = new Generator(ctx.key).uniform([3]).data
      const clause = unsat[Math.min(unsat.length - 1, Math.floor((r0 ?? 0) * unsat.length))] ?? []
      const vars = clause.map((l) => Math.abs(l))
      if (vars.length === 0) return s
      const scored = vars.map((v) => ({ v, b: breaks(s.assignment, v) }))
      let v: number
      const free = scored.find((x) => x.b === 0)
      if (free !== undefined) v = free.v
      else if ((r1 ?? 0) < noise) v = vars[Math.min(vars.length - 1, Math.floor((r2 ?? 0) * vars.length))] as number
      else v = scored.reduce((best, x) => (x.b < best.b ? x : best)).v
      s.assignment[v] = !s.assignment[v]
      s.stats.flips++
      s.events.push({ kind: 'flip', literals: [s.assignment[v] ? v : -v] })
      s.unsatisfied = unsatClauses(s.assignment).length
      if (s.unsatisfied === 0) {
        s.status = 'sat'
        s.events.push({ kind: 'solution', literals: s.assignment.slice(1).map((b, i) => (b ? i + 1 : -(i + 1))) })
      }
      return s
    },
    done: (s) => (s.status === 'sat' ? 'converged' : s.stats.flips >= maxFlips ? 'stopped' : undefined),
    observe: (s) => ({
      metrics: { flips: s.stats.flips, unsatisfied: s.unsatisfied } as Record<string, number>,
      messages: s.events.map((e) => ({ kind: e.kind, payload: { literals: [...e.literals] } as Json })),
    }),
  }
}

/**
 * WalkSAT (Selman, Kautz and Cohen) as a traced {@link Algorithm}: start from a random assignment and repeatedly pick
 * an unsatisfied clause, flipping a variable that breaks no clause if there is one, else a random variable of the
 * clause with probability `noise`, else the variable that breaks fewest clauses. Incomplete: it cannot prove
 * unsatisfiability.
 *
 * @param f - The formula.
 * @param options - Noise, a flip budget, and a seed or key.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a noise outside [0, 1].
 * @concept satisfiability
 * @example
 * ```ts
 * import { cnf, walksatAlgorithm } from 'mdml-engine/logic'
 * walksatAlgorithm(cnf([[1]])).name // => 'walksat'
 * ```
 */
export function walksatAlgorithm(f: Cnf, options: WalksatOptions = {}): Algorithm<WalksatState> {
  return walksatMachine(f, options)
}

/**
 * Search for a model with WalkSAT.
 *
 * @param f - The formula.
 * @param options - Noise, a flip budget, a seed or key, and tracing.
 * @returns `sat` with a model, or `unknown` when the flips run out.
 * @throws MdmlError with code `value` for a noise outside [0, 1].
 * @concept satisfiability
 * @example
 * ```ts
 * import { cnf, satisfies, walksat } from 'mdml-engine/logic'
 * const f = cnf([[1, 2, 3], [-1, -2], [-2, -3], [-1, -3], [2, 3]])
 * const r = walksat(f, { seed: 1 })
 * [r.status, satisfies(f, r.model)] // => ['sat', true]
 * walksat(cnf([[1], [-1]]), { maxFlips: 50 }).status // => 'unknown'
 * ```
 */
export function walksat(f: Cnf, options: WalksatOptions = {}): SatResult {
  const { noise: _n, maxFlips = 100_000, ...run } = options
  const { state } = runAlgorithm(walksatMachine(f, options), { ...run, maxSteps: maxFlips + 1 })
  const sat = state.status === 'sat'
  return {
    status: sat ? 'sat' : 'unknown',
    model: sat ? state.assignment.slice(1).map((b, i) => (b ? i + 1 : -(i + 1))) : null,
    stats: { ...state.stats },
  }
}

/**
 * Decide satisfiability with the default complete solver (CDCL).
 *
 * @param f - The formula.
 * @param options - Options as for {@link cdcl}.
 * @returns The status, a model, and the effort.
 * @index function Satisfiability by DPLL or CDCL.
 * @example
 * ```ts
 * import { cnf, solveSat } from 'mdml-engine/logic'
 * solveSat(cnf([[1, -2], [2]])).model // => [1, 2]
 * ```
 */
export function solveSat(f: Cnf, options: CdclOptions = {}): SatResult {
  return cdcl(f, options)
}
