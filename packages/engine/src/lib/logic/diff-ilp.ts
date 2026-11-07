/**
 * Differentiable inductive logic programming (Evans and Grefenstette 2018, dILP) at small scale. Each intensional
 * predicate has clause templates, and the candidates of a template are all the clauses of up to two body atoms over
 * the language that are safe and not trivial. A softmax over a template's candidates (the learned weights) says which
 * clause the template holds; the valuation of all ground atoms is forward-chained a fixed number of steps, each
 * candidate's consequences computed with the product as conjunction and the maximum over the groundings of its
 * existential variables as disjunction, mixed by the softmax and amalgamated with the previous valuation by the
 * maximum. The cross-entropy of the labelled examples under the final valuation is a smooth function of the weights,
 * so any solver of `smoothSolvers` minimises it, and the learned program is read off by taking each template's most
 * probable clause. A crisp forward chainer checks the extracted program against the examples.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import {
  type GradientPath,
  gradientContract,
  loggedSolver,
  type SmoothProblem,
  type SolverPath,
  smoothSolverConfig,
  solverContract,
} from '../optim/index.ts'
import { foldIn as foldInKey, type Key, key as makeKey, normal } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  clip,
  concat,
  div,
  fromStorage,
  log,
  matmul,
  maximum,
  max as maxReduce,
  mul,
  neg,
  reshape,
  sliceAxis,
  softmax,
  sub,
  sum,
  take,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  runAlgorithm,
  scalar,
  tensor as tensorSource,
} from '../trace/index.ts'

/** An atom over variables: a predicate and its arguments. */
export interface DilpAtom {
  readonly predicate: string
  readonly args: readonly string[]
}

/** A clause `head :- body`. */
export interface DilpClause {
  readonly head: DilpAtom
  readonly body: readonly DilpAtom[]
}

/** A predicate of the language. */
export interface DilpPredicateSpec {
  readonly name: string
  readonly arity: number
}

/** An extensional predicate: given by its facts. */
export interface DilpExtensional extends DilpPredicateSpec {
  readonly facts: readonly (readonly string[])[]
}

/** What the clauses a template may hold look like. */
export interface DilpTemplate {
  /** Existential variables beyond the head's (default 0). */
  readonly existential?: number
  /** The most body atoms (default 2). */
  readonly maxBody?: number
  /** Whether intensional predicates may appear in the body (default false). */
  readonly intensional?: boolean
}

/** An intensional predicate to be learned: one or two clause templates. */
export interface DilpIntensional extends DilpPredicateSpec {
  readonly templates: readonly DilpTemplate[]
}

/** A labelled ground atom of an intensional predicate. */
export interface DilpExample {
  readonly predicate: string
  readonly args: readonly string[]
  readonly label: number
}

/** An induction task. */
export interface DilpTask {
  readonly constants: readonly string[]
  readonly extensional: readonly DilpExtensional[]
  readonly intensional: readonly DilpIntensional[]
  readonly examples: readonly DilpExample[]
  /** Forward-chaining steps (default 4). */
  readonly steps?: number
}

/** The learned program, with the probability of each chosen clause. */
export interface DilpProgram {
  readonly clauses: readonly (DilpClause & {
    readonly predicate: string
    readonly slot: number
    readonly probability: number
  })[]
}

const HEAD_VARIABLES = ['X', 'Y', 'W']
const EXISTENTIAL_VARIABLES = ['Z', 'U', 'V']

/**
 * A clause as text.
 *
 * @param c - The clause.
 * @returns `head :- body1, body2`.
 * @example
 * ```ts
 * import { dilpClauseText } from 'mdml-engine/logic'
 * dilpClauseText({ head: { predicate: 'p', args: ['X'] }, body: [{ predicate: 'q', args: ['X'] }] }) // => 'p(X) :- q(X)'
 * ```
 */
export function dilpClauseText(c: DilpClause): string {
  const atom = (a: DilpAtom): string => `${a.predicate}(${a.args.join(',')})`
  return `${atom(c.head)} :- ${c.body.map(atom).join(', ')}`
}

interface Slot {
  readonly predicate: string
  readonly slot: number
  readonly existential: number
  readonly clauses: readonly DilpClause[]
  /** Gather indices `[K, H, Gr, B]` into the valuation extended with a 1 at the end. */
  readonly indices: Int32Array
  readonly head: number
  readonly groundings: number
  readonly body: number
}

/** A differentiable ILP problem over a task. */
export interface DiffIlp {
  /** The number of clause weights. */
  readonly size: number
  readonly x0: readonly number[]
  /** Fresh initial weights from a key. */
  init(k: Key): number[]
  /** The candidate clauses of every template, in weight order. */
  readonly candidates: readonly {
    readonly predicate: string
    readonly slot: number
    readonly clauses: readonly DilpClause[]
  }[]
  /** The final valuation of every ground atom (extensional first), as a vector. */
  valuation(theta: readonly number[] | Value): Value
  /** The valuation of the labelled examples. */
  scores(theta: readonly number[] | Value): Value
  /** The cross-entropy of the examples as a smooth problem for the solver slot. */
  problem(options?: { readonly l2?: number }): SmoothProblem
  /** Each template's most probable clause. */
  extract(theta: readonly number[]): DilpProgram
  /** The ground atoms a crisp program derives in the task's steps. */
  derive(program: readonly DilpClause[]): ReadonlySet<string>
  /** The fraction of examples a crisp program classifies correctly. */
  accuracy(program: readonly DilpClause[]): number
}

function checkTask(task: DilpTask): void {
  const names = new Set<string>()
  for (const p of [...task.extensional, ...task.intensional]) {
    if (names.has(p.name)) throw new MdmlError('value', `diffIlp: predicate '${p.name}' is declared twice`)
    names.add(p.name)
    if (!(Number.isInteger(p.arity) && p.arity >= 1 && p.arity <= 3))
      throw new MdmlError('value', `diffIlp: predicate '${p.name}' needs an arity from 1 to 3`)
  }
  for (const i of task.intensional) {
    if (i.templates.length < 1 || i.templates.length > 2)
      throw new MdmlError('value', `diffIlp: '${i.name}' needs one or two templates`)
    for (const t of i.templates)
      if (t.maxBody !== undefined && t.maxBody !== 1 && t.maxBody !== 2)
        throw new MdmlError('value', `diffIlp: a template of '${i.name}' needs maxBody 1 or 2`)
  }
  const known = new Set(task.constants)
  for (const e of task.extensional)
    for (const f of e.facts)
      if (f.length !== e.arity || f.some((c) => !known.has(c)))
        throw new MdmlError('value', `diffIlp: bad fact ${e.name}(${f.join(',')})`)
  for (const ex of task.examples) {
    const p = task.intensional.find((i) => i.name === ex.predicate)
    if (p === undefined || ex.args.length !== p.arity || ex.args.some((c) => !known.has(c)))
      throw new MdmlError('value', `diffIlp: bad example ${ex.predicate}(${ex.args.join(',')})`)
  }
}

/**
 * Set up a differentiable ILP problem: the language's ground atoms, the candidate clauses of every template with their
 * precomputed groundings, and the forward-chaining valuation as tensor operations of the clause weights.
 *
 * @param task - The constants, extensional facts, intensional predicates with templates, and labelled examples.
 * @param k - The key the initial weights are drawn from.
 * @returns The problem.
 * @throws MdmlError with code `value` for a malformed task.
 * @example
 * ```ts
 * import { diffIlp } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const task = {
 *   constants: ['a', 'b'],
 *   extensional: [{ name: 'edge', arity: 2, facts: [['a', 'b']] }],
 *   intensional: [{ name: 'path', arity: 2, templates: [{ maxBody: 1 }] }],
 *   examples: [{ predicate: 'path', args: ['a', 'b'], label: 1 }],
 * }
 * diffIlp(task, key(0)).candidates[0]?.clauses.length // => 2
 * ```
 */
export function diffIlp(task: DilpTask, k: Key): DiffIlp {
  checkTask(task)
  const C = task.constants.length
  const steps = task.steps ?? 4
  const cIndex = new Map(task.constants.map((c, i) => [c, i]))
  const predicates: DilpPredicateSpec[] = [...task.extensional, ...task.intensional]
  const offset = new Map<string, number>()
  let G = 0
  for (const p of predicates) {
    offset.set(p.name, G)
    G += C ** p.arity
  }
  const encode = (p: string, tuple: readonly number[]): number =>
    (offset.get(p) as number) + tuple.reduce((acc, c) => acc * C + c, 0)
  const atomName = (p: string, tuple: readonly number[]): string =>
    `${p}(${tuple.map((c) => task.constants[c]).join(',')})`

  // Candidate clauses of each template.
  const slots: Slot[] = []
  for (const ip of task.intensional) {
    ip.templates.forEach((t, slot) => {
      const v = t.existential ?? 0
      const maxBody = (t.maxBody ?? 2) as 1 | 2
      const vars = [...HEAD_VARIABLES.slice(0, ip.arity), ...EXISTENTIAL_VARIABLES.slice(0, v)]
      const bodyPredicates = predicates.filter(
        (p) => task.extensional.some((e) => e.name === p.name) || (t.intensional ?? false),
      )
      const atoms: DilpAtom[] = []
      for (const p of bodyPredicates) {
        const total = vars.length ** p.arity
        for (let m = 0; m < total; m++) {
          const args: string[] = []
          let r = m
          for (let j = 0; j < p.arity; j++) {
            args.unshift(vars[r % vars.length] as string)
            r = Math.floor(r / vars.length)
          }
          atoms.push({ predicate: p.name, args })
        }
      }
      const head: DilpAtom = { predicate: ip.name, args: HEAD_VARIABLES.slice(0, ip.arity) }
      const same = (a: DilpAtom, b: DilpAtom): boolean => a.predicate === b.predicate && a.args.join() === b.args.join()
      const bodies: DilpAtom[][] = atoms.map((a) => [a])
      if (maxBody === 2)
        for (let i = 0; i < atoms.length; i++)
          for (let j = i + 1; j < atoms.length; j++) bodies.push([atoms[i] as DilpAtom, atoms[j] as DilpAtom])
      const clauses = bodies
        .filter((body) => {
          const used = new Set(body.flatMap((a) => a.args))
          return vars.every((x) => used.has(x)) && !body.some((a) => same(a, head))
        })
        .map((body): DilpClause => ({ head, body }))
      if (clauses.length === 0)
        throw new MdmlError('value', `diffIlp: template ${slot} of '${ip.name}' has no candidate clause`)
      // Groundings.
      const H = C ** ip.arity
      const Gr = C ** v
      const B = maxBody
      const indices = new Int32Array(clauses.length * H * Gr * B).fill(G)
      clauses.forEach((c, ci) => {
        for (let h = 0; h < H; h++) {
          const env = new Map<string, number>()
          let r = h
          for (let j = ip.arity - 1; j >= 0; j--) {
            env.set(HEAD_VARIABLES[j] as string, r % C)
            r = Math.floor(r / C)
          }
          for (let g = 0; g < Gr; g++) {
            let q = g
            for (let j = v - 1; j >= 0; j--) {
              env.set(EXISTENTIAL_VARIABLES[j] as string, q % C)
              q = Math.floor(q / C)
            }
            c.body.forEach((a, bi) => {
              indices[((ci * H + h) * Gr + g) * B + bi] = encode(
                a.predicate,
                a.args.map((x) => env.get(x) as number),
              )
            })
          }
        }
      })
      slots.push({ predicate: ip.name, slot, existential: v, clauses, indices, head: H, groundings: Gr, body: B })
    })
  }
  const weightOffsets: number[] = []
  let size = 0
  for (const s of slots) {
    weightOffsets.push(size)
    size += s.clauses.length
  }
  const init = (kk: Key): number[] => Array.from(normal(kk, [size], { scale: 0.5 }).data)

  // The extensional valuation, fixed.
  const a0 = new Float64Array(G)
  for (const e of task.extensional)
    for (const f of e.facts)
      a0[
        encode(
          e.name,
          f.map((c) => cIndex.get(c) as number),
        )
      ] = 1
  const blockOf = (name: string): [number, number] => {
    const p = predicates.find((x) => x.name === name) as DilpPredicateSpec
    return [offset.get(name) as number, C ** p.arity]
  }
  const extensionalEnd = task.intensional.length === 0 ? G : (offset.get(task.intensional[0]?.name as string) as number)

  const weightsOf = (theta: readonly number[] | Value): Value =>
    Array.isArray(theta) ? fromStorage(Float64Array.from(theta as number[]), [size], 'float64') : (theta as Value)

  const valuation = (theta: readonly number[] | Value): Value => {
    const w = weightsOf(theta)
    if ((w.shape[0] ?? 0) !== size) throw new MdmlError('shape', `diffIlp: expected ${size} weights`)
    const probabilities = slots.map((s, i) =>
      softmax(sliceAxis(w, 0, weightOffsets[i] as number, (weightOffsets[i] as number) + s.clauses.length), 0),
    )
    let a: Value = fromStorage(a0.slice(), [G], 'float64')
    const unit = fromStorage(Float64Array.of(1), [1], 'float64')
    for (let t = 0; t < steps; t++) {
      const extended = concat([a, unit], 0)
      const pieces: Value[] = [sliceAxis(a, 0, 0, extensionalEnd)]
      for (const ip of task.intensional) {
        const [start, len] = blockOf(ip.name)
        let previous = sliceAxis(a, 0, start, start + len)
        let miss: Value | null = null
        slots.forEach((s, i) => {
          if (s.predicate !== ip.name) return
          const gathered = reshape(take(extended, s.indices, 0), [s.clauses.length, s.head, s.groundings, s.body])
          // The product over the body atoms, written out: the gradient of `prod` divides by its inputs, which are often 0.
          let conjunction: Value = reshape(sliceAxis(gathered, 3, 0, 1), [s.clauses.length, s.head, s.groundings])
          for (let b = 1; b < s.body; b++)
            conjunction = mul(
              conjunction,
              reshape(sliceAxis(gathered, 3, b, b + 1), [s.clauses.length, s.head, s.groundings]),
            )
          const consequence = maxReduce(conjunction, 2)
          const mixed = reshape(matmul(reshape(probabilities[i] as Value, [1, s.clauses.length]), consequence), [
            s.head,
          ])
          const complement = sub(1, mixed)
          miss = miss === null ? complement : mul(miss, complement)
        })
        previous = maximum(previous, sub(1, miss as unknown as Value))
        pieces.push(previous)
      }
      a = concat(pieces, 0)
    }
    return a
  }

  const exampleIndex = task.examples.map((e) =>
    encode(
      e.predicate,
      e.args.map((c) => cIndex.get(c) as number),
    ),
  )
  const labels = fromStorage(
    Float64Array.from(task.examples, (e) => e.label),
    [task.examples.length],
    'float64',
  )
  const scores = (theta: readonly number[] | Value): Value => take(valuation(theta), exampleIndex, 0)

  const derive = (program: readonly DilpClause[]): ReadonlySet<string> => {
    const model = new Set<string>()
    for (const e of task.extensional)
      for (const f of e.facts)
        model.add(
          atomName(
            e.name,
            f.map((c) => cIndex.get(c) as number),
          ),
        )
    for (let t = 0; t < steps; t++) {
      const next = new Set(model)
      for (const c of program) {
        const vars = [...new Set([...c.head.args, ...c.body.flatMap((a) => a.args)])]
        const total = C ** vars.length
        for (let m = 0; m < total; m++) {
          const env = new Map<string, number>()
          let r = m
          for (const x of vars) {
            env.set(x, r % C)
            r = Math.floor(r / C)
          }
          const holds = c.body.every((a) =>
            model.has(
              atomName(
                a.predicate,
                a.args.map((x) => env.get(x) as number),
              ),
            ),
          )
          if (holds)
            next.add(
              atomName(
                c.head.predicate,
                c.head.args.map((x) => env.get(x) as number),
              ),
            )
        }
      }
      for (const x of next) model.add(x)
    }
    return model
  }

  return {
    size,
    x0: init(k),
    init,
    candidates: slots.map((s) => ({ predicate: s.predicate, slot: s.slot, clauses: s.clauses })),
    valuation,
    scores,
    problem: (options = {}) => ({
      objective: (theta: Value): Value => {
        const s = clip(scores(theta), 1e-7, 1 - 1e-7)
        const cross = add(mul(labels, log(s)), mul(sub(1, labels), log(sub(1, s))))
        let loss = neg(div(sum(cross), task.examples.length))
        if ((options.l2 ?? 0) > 0) loss = add(loss, mul(0.5 * (options.l2 as number), sum(mul(theta, theta))))
        return loss
      },
      x0: init(k),
    }),
    extract: (theta) =>
      ({
        clauses: slots.map((s, i) => {
          const w = theta.slice(weightOffsets[i] as number, (weightOffsets[i] as number) + s.clauses.length)
          const top = maxOf(w)
          const z = w.reduce((acc, x) => acc + Math.exp(x - top), 0)
          const best = w.indexOf(top)
          return {
            ...(s.clauses[best] as DilpClause),
            predicate: s.predicate,
            slot: s.slot,
            probability: 1 / z,
          }
        }),
      }) as DilpProgram,
    derive,
    accuracy: (program) => {
      const model = derive(program)
      const right = task.examples.filter(
        (e) => model.has(`${e.predicate}(${e.args.join(',')})`) === (e.label === 1),
      ).length
      return right / task.examples.length
    },
  }
}

/** The paths of {@link diffIlpAlgorithm}: the solver's, and the clause weights. */
export type DilpPath = SolverPath | GradientPath | 'dilp/probabilities' | 'dilp/accuracy'

/** The log contract of {@link diffIlpAlgorithm}. */
export const dilpContract: Contract<unknown, DilpPath> = defineContract({
  ...solverContract,
  ...gradientContract,
  'dilp/probabilities': tensorSource({
    role: 'param',
    shape: ['weights'],
    description: 'the clause weights (logits of each template’s softmax)',
  }),
  'dilp/accuracy': scalar({
    description: 'the fraction of examples the extracted program classifies correctly',
    direction: 'higher',
  }),
})

/**
 * Learn a program as a traced algorithm over the solver slot: the cross-entropy of the examples minimised by any solver
 * of `smoothSolvers` (L-BFGS by default), logging the clause weights and the accuracy of the program extracted at each
 * step.
 *
 * @param ilp - The problem.
 * @param options - The solver, the ridge penalty, the key, and a start.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { diffIlp, diffIlpAlgorithm } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const task = {
 *   constants: ['a', 'b'],
 *   extensional: [{ name: 'edge', arity: 2, facts: [['a', 'b']] }],
 *   intensional: [{ name: 'path', arity: 2, templates: [{ maxBody: 1 }] }],
 *   examples: [{ predicate: 'path', args: ['a', 'b'], label: 1 }],
 * }
 * diffIlpAlgorithm(diffIlp(task, key(0))).name // => 'lbfgs'
 * ```
 */
export function diffIlpAlgorithm(
  ilp: DiffIlp,
  options: {
    readonly solver?: SpaceValue
    readonly l2?: number
    readonly key?: Key
    readonly start?: readonly number[]
  } = {},
): Algorithm<unknown, DilpPath> {
  const problem = ilp.problem(options.l2 === undefined ? {} : { l2: options.l2 })
  const inner = loggedSolver<'dilp/probabilities' | 'dilp/accuracy'>(
    options.start === undefined ? problem : { ...problem, x0: [...options.start] },
    options.solver ?? smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-7 }),
    options.key ?? makeKey(0),
    (x, ctx) => {
      ctx.log('dilp/probabilities', x)
      if (ctx.wants('dilp/accuracy')) ctx.log('dilp/accuracy', ilp.accuracy(ilp.extract(Array.from(x)).clauses))
    },
  )
  return { ...inner, schema: contractSchema(dilpContract) }
}

/** A learned program. */
export interface DilpFit {
  readonly theta: number[]
  readonly loss: number
  readonly program: DilpProgram
  readonly converged: boolean
  readonly steps: number
}

/**
 * Learn a logic program from examples. Training starts from the problem's own initial weights, then from fresh ones
 * drawn from the key for each further restart; the lowest loss wins, and the program is its most probable clauses.
 *
 * @param ilp - The problem.
 * @param options - The solver, the ridge penalty, the most steps (default 300), the key, the restarts (default 1) and the loss that ends them early.
 * @returns The weights, the loss, the extracted program and whether the solver converged.
 * @example
 * ```ts
 * import { dilpClauseText, diffIlp, learnProgram } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const task = {
 *   constants: ['a', 'b', 'c'],
 *   extensional: [{ name: 'succ', arity: 2, facts: [['a', 'b'], ['b', 'c']] }],
 *   intensional: [{ name: 'pred', arity: 2, templates: [{ maxBody: 1 }] }],
 *   examples: [
 *     { predicate: 'pred', args: ['b', 'a'], label: 1 },
 *     { predicate: 'pred', args: ['c', 'b'], label: 1 },
 *     { predicate: 'pred', args: ['a', 'b'], label: 0 },
 *     { predicate: 'pred', args: ['b', 'c'], label: 0 },
 *   ],
 * }
 * learnProgram(diffIlp(task, key(0)), { restarts: 3 }).program.clauses.map(dilpClauseText) // => ['pred(X,Y) :- succ(Y,X)']
 * ```
 */
export function learnProgram(
  ilp: DiffIlp,
  options: {
    readonly solver?: SpaceValue
    readonly l2?: number
    readonly maxSteps?: number
    readonly key?: Key
    readonly restarts?: number
    /** Stop restarting once a start reaches a loss at most this (default 0.01). */
    readonly acceptLoss?: number
  } = {},
): DilpFit {
  const key = options.key ?? makeKey(0)
  let best: DilpFit | null = null
  for (let r = 0; r < (options.restarts ?? 1); r++) {
    const alg = diffIlpAlgorithm(ilp, { ...options, key, start: r === 0 ? ilp.x0 : ilp.init(foldInKey(key, r)) })
    const run = runAlgorithm(alg as unknown as Algorithm<unknown>, { key, maxSteps: options.maxSteps ?? 300 })
    const s = run.state as { x: Float64Array; value: number }
    if (best === null || s.value < best.loss) {
      const theta = Array.from(s.x)
      best = {
        theta,
        loss: s.value,
        program: ilp.extract(theta),
        converged: run.status === 'converged',
        steps: run.steps,
      }
    }
  }
  return best as DilpFit
}
