/**
 * The neural theorem prover of Rocktaschel and Riedel (2017): backward chaining in which symbols are unified softly. Every
 * constant and predicate has an embedding; two symbols unify with the score of a Gaussian kernel of the distance between
 * their embeddings; a goal is proved against the facts and, to a depth, against rules, each proof's score is the minimum
 * of the unification scores along it, and the goal's score is the maximum over proofs. The whole proof network is built
 * from tensor operations, so the embeddings (and the predicate vectors of rule templates such as `#1(X,Y) :- #2(X,Z),
 * #3(Z,Y)`) train by gradient descent on facts and corrupted negatives: facts derivable by a rule from other facts score
 * high although they were never stated, and a template's learned vectors can be decoded to the nearest predicates,
 * which induces the rule. Proofs are batched as tensors (states by branches), a variable is bound by unifying with a fact
 * and goals with an unbound variable are proved against facts only.
 */

import { MdmlError } from '../base/index.ts'
import {
  type GradientPath,
  gradientContract,
  loggedSolver,
  type SmoothProblem,
  type SolverPath,
  smoothSolverConfig,
  solverContract,
} from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, normal, permutation } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  broadcastTo,
  clip,
  div,
  exp,
  fromStorage,
  item,
  log,
  matmul,
  maximum,
  max as maxReduce,
  minimum,
  mul,
  neg,
  reshape,
  sliceAxis,
  sub,
  sum,
  toFlat,
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

/** An atom: a predicate applied to constants (lower-case) and variables (upper-case). A predicate starting with `#` is a learnable template. */
export interface NtpAtom {
  readonly predicate: string
  readonly args: readonly string[]
}

/** A rule `head :- body`. */
export interface NtpRule {
  readonly head: NtpAtom
  readonly body: readonly NtpAtom[]
}

const isVariable = (t: string): boolean => /^[A-Z_]/.test(t)

/**
 * Parse an atom such as `parent(ann,bob)` or `#1(X,Y)`.
 *
 * @param text - The atom.
 * @returns The atom.
 * @throws MdmlError with code `value` on a syntax error.
 * @example
 * ```ts
 * import { parseNtpAtom } from 'mdml-engine/logic'
 * parseNtpAtom('parent(ann, bob)') // => { predicate: 'parent', args: ['ann', 'bob'] }
 * ```
 */
export function parseNtpAtom(text: string): NtpAtom {
  const m = /^\s*(#?\w+)\s*\(([^()]*)\)\s*$/.exec(text)
  if (m === null) throw new MdmlError('value', `parseNtpAtom: cannot read '${text}'`)
  return { predicate: m[1] as string, args: (m[2] as string).split(',').map((a) => a.trim()) }
}

/**
 * Parse a rule such as `grandparent(X,Y) :- parent(X,Z), parent(Z,Y)`.
 *
 * @param text - The rule.
 * @returns The rule.
 * @throws MdmlError with code `value` on a syntax error.
 * @example
 * ```ts
 * import { parseNtpRule } from 'mdml-engine/logic'
 * parseNtpRule('g(X,Y) :- p(X,Z), p(Z,Y)').body.length // => 2
 * ```
 */
export function parseNtpRule(text: string): NtpRule {
  const [head, body] = text.split(':-')
  if (head === undefined || body === undefined || body.trim() === '')
    throw new MdmlError('value', `parseNtpRule: cannot read '${text}'`)
  return {
    head: parseNtpAtom(head),
    body: body.split(/\)\s*,/).map((a, i, all) => parseNtpAtom(i < all.length - 1 ? `${a})` : a)),
  }
}

/** The configuration of a neural theorem prover. */
export interface NtpConfig {
  /** The facts, as atoms or text. */
  readonly facts: readonly (NtpAtom | string)[]
  /** The rules, as rules or text. */
  readonly rules: readonly (NtpRule | string)[]
  /** The embedding width (default 8). */
  readonly dimension?: number
  /** The proof depth: how many rule applications may nest (default 1). */
  readonly depth?: number
  /** The kernel width `mu` of `exp(-d^2 / (2 mu^2))` (default 1). */
  readonly kernelWidth?: number
  /** The scale of the initial embeddings (default 0.7). */
  readonly initScale?: number
}

/** A goal to learn from: an atom and whether it holds. */
export interface NtpExample {
  readonly goal: NtpAtom
  readonly label: 0 | 1
  /** The index of the fact to leave out while proving (a positive training fact must not prove itself). */
  readonly exclude?: number
}

/** A neural theorem prover over a knowledge base. */
export interface NeuralTheoremProver {
  /** The symbols (constants, predicates and templates) in embedding order. */
  readonly symbols: readonly string[]
  readonly dimension: number
  readonly facts: readonly NtpAtom[]
  readonly rules: readonly NtpRule[]
  /** The number of parameters (embeddings, flattened). */
  readonly size: number
  readonly x0: readonly number[]
  /** Fresh initial embeddings from a key (for restarts). */
  init(k: Key): number[]
  /** The proof score of a ground goal in [0, 1]. */
  score(theta: Value | readonly number[], goal: NtpAtom, exclude?: number): Value
  /** The mean cross-entropy of the examples (plus an optional ridge penalty on the embeddings), as a smooth problem. */
  problem(examples: readonly NtpExample[], options?: { readonly l2?: number }): SmoothProblem
  /** The embedding of a symbol. */
  embedding(theta: readonly number[], symbol: string): number[]
  /** For each template predicate, the nearest ordinary predicate by embedding distance. */
  decode(theta: readonly number[]): Readonly<Record<string, { readonly symbol: string; readonly distance: number }>>
}

interface Batch {
  /** Scores `[S, M]`: S independent states, M proof branches each. */
  readonly score: Value
  /** The bindings `[S, M, k]` of the variables bound so far. */
  readonly bind: ReadonlyMap<string, Value>
}

const EPS = 1e-7

/**
 * Build a neural theorem prover over a knowledge base. The returned object evaluates proof scores of ground goals and
 * states the training problem; its parameters are the embeddings of every symbol of the facts and rules plus one vector
 * per `#template` predicate.
 *
 * @param config - The facts, rules, embedding width, depth and kernel width.
 * @param k - The key the initial embeddings are drawn from.
 * @returns The prover.
 * @throws MdmlError with code `value` for a fact that is not ground, a rule whose body has no atoms, a non-positive
 *   width or depth, or an atom of the wrong shape.
 * @example
 * ```ts
 * import { neuralTheoremProver } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const ntp = neuralTheoremProver({ facts: ['p(a,b)'], rules: ['q(X,Y) :- p(X,Y)'] }, key(0))
 * ntp.symbols // => ['p', 'a', 'b', 'q']
 * ```
 */
export function neuralTheoremProver(config: NtpConfig, k: Key): NeuralTheoremProver {
  const dim = config.dimension ?? 8
  const depth = config.depth ?? 1
  const mu = config.kernelWidth ?? 1
  if (!(Number.isInteger(dim) && dim >= 1 && Number.isInteger(depth) && depth >= 0 && mu > 0))
    throw new MdmlError(
      'value',
      'neuralTheoremProver: need a positive width, a non-negative depth and a positive kernel width',
    )
  const facts = config.facts.map((f) => (typeof f === 'string' ? parseNtpAtom(f) : f))
  const rules = config.rules.map((r) => (typeof r === 'string' ? parseNtpRule(r) : r))
  for (const f of facts) {
    if (f.args.length !== 2 || f.args.some(isVariable) || f.predicate.startsWith('#'))
      throw new MdmlError(
        'value',
        `neuralTheoremProver: facts must be binary and ground, got ${f.predicate}(${f.args.join(',')})`,
      )
  }
  for (const r of rules) {
    if (r.body.length === 0) throw new MdmlError('value', 'neuralTheoremProver: a rule needs a body')
    for (const a of [r.head, ...r.body])
      if (a.args.length !== 2) throw new MdmlError('value', 'neuralTheoremProver: atoms must be binary')
  }
  const symbols: string[] = []
  const index = new Map<string, number>()
  const register = (s: string): void => {
    if (!index.has(s)) {
      index.set(s, symbols.length)
      symbols.push(s)
    }
  }
  for (const f of facts) {
    register(f.predicate)
    f.args.forEach(register)
  }
  for (const r of rules) {
    for (const a of [r.head, ...r.body]) {
      register(a.predicate)
      for (const t of a.args) if (!isVariable(t)) register(t)
    }
  }
  const n = symbols.length
  const size = n * dim
  const scale = config.initScale ?? 0.7
  const x0 = Array.from(normal(k, [n * dim], { scale }).data)
  const nFacts = facts.length
  const rowOf = (s: string): number => {
    const i = index.get(s)
    if (i === undefined) throw new MdmlError('value', `neuralTheoremProver: unknown symbol '${s}'`)
    return i
  }
  const selector = (pick: (f: NtpAtom) => string): Value => {
    const m = new Float64Array(nFacts * n)
    facts.forEach((f, i) => {
      m[i * n + rowOf(pick(f))] = 1
    })
    return fromStorage(m, [nFacts, n], 'float64')
  }
  const selPred = selector((f) => f.predicate)
  const selArg = [selector((f) => f.args[0] as string), selector((f) => f.args[1] as string)]
  const kernel = (a: Value, b: Value): Value => {
    const d = sub(a, b)
    return exp(div(neg(sum(mul(d, d), -1)), 2 * mu * mu))
  }
  const row = (E: Value, s: string): Value => reshape(sliceAxis(E, 0, rowOf(s), rowOf(s) + 1), [dim])
  const matrix = (theta: Value | readonly number[]): Value => {
    const t = Array.isArray(theta)
      ? fromStorage(Float64Array.from(theta as number[]), [size], 'float64')
      : (theta as Value)
    if ((t.shape[0] ?? 0) !== size) throw new MdmlError('shape', `neuralTheoremProver: expected ${size} parameters`)
    return reshape(t, [n, dim])
  }

  /** The embedding of a symbol, shaped to broadcast against states, branches, facts and width. */
  const symbolEmbedding = (E: Value, s: string): Value => reshape(row(E, s), [1, 1, 1, dim])

  /**
   * Prove one atom, whose predicate is an embedding `[S|1, M|1, 1, k]` and whose arguments are constants or variables,
   * extending each branch of a batch. A mask `[S, F]` switches facts off per state (leave-one-out training).
   */
  const orStep = (
    E: Value,
    predicate: Value,
    args: readonly string[],
    batch: Batch,
    d: number,
    mask: Value | null,
  ): Batch => {
    const [S, M] = batch.score.shape as [number, number]
    const P = reshape(matmul(selPred, E), [1, 1, nFacts, dim])
    const A = [matmul(selArg[0] as Value, E), matmul(selArg[1] as Value, E)]
    const wide = (x: Value): Value => reshape(x, [1, 1, nFacts, dim])
    let score: Value = minimum(reshape(batch.score, [S, M, 1]), kernel(predicate, P))
    if (mask !== null) score = mul(score, reshape(mask, [S, 1, nFacts]))
    const fresh = new Map<string, Value>()
    args.forEach((t, i) => {
      const column = A[i] as Value
      if (!isVariable(t)) {
        score = minimum(score, kernel(symbolEmbedding(E, t), wide(column)))
        return
      }
      const bound = batch.bind.get(t)
      if (bound !== undefined) {
        score = minimum(score, kernel(reshape(bound, [S, M, 1, dim]), wide(column)))
        return
      }
      const earlier = fresh.get(t)
      if (earlier !== undefined) score = minimum(score, kernel(wide(earlier), wide(column)))
      else fresh.set(t, column)
    })
    if (fresh.size === 0) {
      let best: Value = maxReduce(score, 2)
      if (d > 0) for (const rule of rules) best = maximum(best, ruleStep(E, predicate, args, rule, batch, d, mask))
      return { score: best, bind: batch.bind }
    }
    const bind = new Map<string, Value>()
    for (const [v, b] of batch.bind)
      bind.set(v, reshape(broadcastTo(reshape(b, [S, M, 1, dim]), [S, M, nFacts, dim]), [S, M * nFacts, dim]))
    for (const [v, column] of fresh)
      bind.set(v, reshape(broadcastTo(wide(column), [S, M, nFacts, dim]), [S, M * nFacts, dim]))
    return { score: reshape(score, [S, M * nFacts]), bind }
  }

  /** The score of a ground goal proved by one rule: head unification, then the body in the rule's own scope. */
  const ruleStep = (
    E: Value,
    predicate: Value,
    args: readonly string[],
    rule: NtpRule,
    batch: Batch,
    d: number,
    mask: Value | null,
  ): Value => {
    const [S, M] = batch.score.shape as [number, number]
    const flat = S * M
    const values = args.map((t) =>
      isVariable(t)
        ? reshape(batch.bind.get(t) as Value, [flat, dim])
        : broadcastTo(reshape(row(E, t), [1, dim]), [flat, dim]),
    )
    const goalPred = reshape(broadcastTo(predicate, [S, M, 1, dim]), [flat, dim])
    const wideRow = (s: string): Value => broadcastTo(reshape(row(E, s), [1, dim]), [flat, dim])
    let score: Value = kernel(goalPred, wideRow(rule.head.predicate))
    const local = new Map<string, Value>()
    rule.head.args.forEach((t, i) => {
      const v = values[i] as Value
      if (!isVariable(t)) score = minimum(score, kernel(v, wideRow(t)))
      else if (local.has(t)) score = minimum(score, kernel(local.get(t) as Value, v))
      else local.set(t, v)
    })
    // Every state of the rule's scope descends from one state and branch of the caller; its mask is that state's.
    const innerMask =
      mask === null ? null : reshape(broadcastTo(reshape(mask, [S, 1, nFacts]), [S, M, nFacts]), [flat, nFacts])
    let inner: Batch = {
      score: reshape(score, [flat, 1]),
      bind: new Map([...local].map(([v, b]) => [v, reshape(b, [flat, 1, dim])])),
    }
    for (const body of rule.body)
      inner = orStep(E, symbolEmbedding(E, body.predicate), body.args, inner, d - 1, innerMask)
    return minimum(reshape(batch.score, [S, M]), reshape(maxReduce(inner.score, 1), [S, M]))
  }

  const scoreAll = (
    theta: Value | readonly number[],
    goals: readonly NtpAtom[],
    excludes: readonly (number | undefined)[] = [],
  ): Value => {
    for (const goal of goals) {
      if (goal.args.length !== 2 || goal.args.some(isVariable))
        throw new MdmlError('value', 'neuralTheoremProver: a goal must be a ground binary atom')
    }
    const E = matrix(theta)
    const G = goals.length
    const gather = (names: readonly string[]): Value => {
      const m = new Float64Array(G * n)
      names.forEach((s, i) => {
        m[i * n + rowOf(s)] = 1
      })
      return fromStorage(m, [G, n], 'float64')
    }
    const hasMask = excludes.some((e) => e !== undefined)
    const mask = hasMask
      ? fromStorage(
          Float64Array.from({ length: G * nFacts }, (_, j) =>
            excludes[Math.floor(j / nFacts)] === j % nFacts ? 0 : 1,
          ),
          [G, nFacts],
          'float64',
        )
      : null
    const bind = new Map<string, Value>([
      ['_G0', reshape(matmul(gather(goals.map((g) => g.args[0] as string)), E), [G, 1, dim])],
      ['_G1', reshape(matmul(gather(goals.map((g) => g.args[1] as string)), E), [G, 1, dim])],
    ])
    const start: Batch = { score: fromStorage(new Float64Array(G).fill(1), [G, 1], 'float64'), bind }
    const predicate = reshape(matmul(gather(goals.map((g) => g.predicate)), E), [G, 1, 1, dim])
    const out = orStep(E, predicate, ['_G0', '_G1'], start, depth, mask)
    return reshape(maxReduce(out.score, 1), [G])
  }

  const score = (theta: Value | readonly number[], goal: NtpAtom, exclude?: number): Value =>
    reshape(scoreAll(theta, [goal], [exclude]), [])

  return {
    symbols,
    dimension: dim,
    facts,
    rules,
    size,
    x0,
    init: (kk) => Array.from(normal(kk, [n * dim], { scale }).data),
    score,
    problem: (examples, options = {}) => {
      const l2 = options.l2 ?? 0
      return {
        objective: (theta: Value): Value => {
          const scores = clip(
            scoreAll(
              theta,
              examples.map((e) => e.goal),
              examples.map((e) => e.exclude),
            ),
            EPS,
            1 - EPS,
          )
          const labels = fromStorage(
            Float64Array.from(examples, (e) => e.label),
            [examples.length],
            'float64',
          )
          const cross = add(mul(labels, log(scores)), mul(sub(1, labels), log(sub(1, scores))))
          let loss = neg(div(sum(cross), examples.length))
          if (l2 > 0) loss = add(loss, mul(0.5 * l2, sum(mul(theta, theta))))
          return loss
        },
        x0: [...x0],
      }
    },
    embedding: (theta, symbol) => theta.slice(rowOf(symbol) * dim, (rowOf(symbol) + 1) * dim),
    decode: (theta) => {
      const plain = symbols.filter((s) => !s.startsWith('#') && facts.some((f) => f.predicate === s))
      const out: Record<string, { symbol: string; distance: number }> = {}
      for (const t of symbols.filter((s) => s.startsWith('#'))) {
        const e = theta.slice(rowOf(t) * dim, (rowOf(t) + 1) * dim)
        let best = { symbol: '', distance: Number.POSITIVE_INFINITY }
        for (const p of plain) {
          const q = theta.slice(rowOf(p) * dim, (rowOf(p) + 1) * dim)
          const dist = Math.hypot(...e.map((x, i) => x - (q[i] ?? 0)))
          if (dist < best.distance) best = { symbol: p, distance: dist }
        }
        out[t] = best
      }
      return out
    },
  }
}

/** Options of {@link ntpTrainingSet}. */
export interface NtpTrainingOptions {
  /** Corrupted negatives per positive (default 2). */
  readonly negatives?: number
  /** Which facts to learn from (default all). */
  readonly only?: (fact: NtpAtom, index: number) => boolean
}

/**
 * The training examples of a knowledge base: every fact as a positive (proved with itself left out), and for each
 * positive some negatives made by replacing one argument by another constant, skipping those that are facts.
 *
 * @param prover - The prover.
 * @param k - The key the corruptions are drawn from.
 * @param options - The negatives per positive and a filter over the facts.
 * @returns The examples.
 * @example
 * ```ts
 * import { neuralTheoremProver, ntpTrainingSet } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const ntp = neuralTheoremProver({ facts: ['p(a,b)', 'p(b,c)'], rules: [] }, key(0))
 * ntpTrainingSet(ntp, key(1), { negatives: 1 }).filter((e) => e.label === 0).length // => 2
 * ```
 */
export function ntpTrainingSet(prover: NeuralTheoremProver, k: Key, options: NtpTrainingOptions = {}): NtpExample[] {
  const per = options.negatives ?? 2
  const constants = [...new Set(prover.facts.flatMap((f) => f.args))]
  const known = new Set(prover.facts.map((f) => `${f.predicate}(${f.args.join(',')})`))
  const out: NtpExample[] = []
  prover.facts.forEach((f, i) => {
    if (options.only !== undefined && !options.only(f, i)) return
    out.push({ goal: f, label: 1, exclude: i })
    const order = permutation(foldIn(k, i), constants.length * 2).data
    let made = 0
    for (const o of order) {
      if (made >= per) break
      const slot = (o as number) % 2
      const c = constants[Math.floor((o as number) / 2) % constants.length] as string
      const args = [...f.args]
      args[slot] = c
      const text = `${f.predicate}(${args.join(',')})`
      if (known.has(text) || c === f.args[slot]) continue
      out.push({ goal: { predicate: f.predicate, args }, label: 0 })
      made++
    }
  })
  return out
}

// ------------------------------------------------------------------------------------------------ training

/** The paths of {@link ntpAlgorithm}: the solver's, and the prover's. */
export type NtpPath = SolverPath | GradientPath | 'ntp/embeddings' | 'ntp/positive' | 'ntp/negative'

/** The log contract of {@link ntpAlgorithm}. */
export const ntpContract: Contract<unknown, NtpPath> = defineContract({
  ...solverContract,
  ...gradientContract,
  'ntp/embeddings': tensorSource({
    role: 'param',
    shape: ['symbols', 'dimension'],
    description: 'the symbol embeddings',
  }),
  'ntp/positive': scalar({ description: 'the mean proof score of the positive examples', direction: 'higher' }),
  'ntp/negative': scalar({ description: 'the mean proof score of the negative examples', direction: 'lower' }),
})

/**
 * Train a neural theorem prover as a traced algorithm over the solver slot: the cross-entropy of the examples minimised
 * by any solver of `smoothSolvers` (L-BFGS by default), logging the embeddings and the mean scores of the positives and
 * the negatives beside the solver's channels.
 *
 * @param prover - The prover.
 * @param examples - Positive and negative goals.
 * @param options - The solver, the ridge penalty and the key.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { neuralTheoremProver, ntpAlgorithm, ntpTrainingSet } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const ntp = neuralTheoremProver({ facts: ['p(a,b)', 'p(b,c)'], rules: [] }, key(0))
 * ntpAlgorithm(ntp, ntpTrainingSet(ntp, key(1))).name // => 'lbfgs'
 * ```
 */
export function ntpAlgorithm(
  prover: NeuralTheoremProver,
  examples: readonly NtpExample[],
  options: {
    readonly solver?: SpaceValue
    readonly l2?: number
    readonly key?: Key
    /** Where to start (default the prover's initial embeddings). */
    readonly start?: readonly number[]
  } = {},
): Algorithm<unknown, NtpPath> {
  const problem = prover.problem(examples, options.l2 === undefined ? {} : { l2: options.l2 })
  const inner = loggedSolver<'ntp/embeddings' | 'ntp/positive' | 'ntp/negative'>(
    options.start === undefined ? problem : { ...problem, x0: [...options.start] },
    options.solver ?? smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-6 }),
    options.key ?? makeKey(0),
    (x, ctx) => {
      ctx.log('ntp/embeddings', x)
      if (!ctx.wants('ntp/positive') && !ctx.wants('ntp/negative')) return
      const mean = (label: 0 | 1): number => {
        const xs = examples.filter((e) => e.label === label)
        return xs.length === 0
          ? 0
          : xs.reduce((a, e) => a + item(prover.score(Array.from(x), e.goal, e.exclude)), 0) / xs.length
      }
      ctx.log('ntp/positive', mean(1))
      ctx.log('ntp/negative', mean(0))
    },
  )
  return { ...inner, schema: contractSchema(ntpContract) }
}

/** A trained prover. */
export interface NtpFit {
  readonly theta: number[]
  readonly loss: number
  readonly converged: boolean
  readonly steps: number
}

/**
 * Train a prover on examples with the chosen solver.
 *
 * @param prover - The prover.
 * @param examples - Positive and negative goals.
 * @param options - The solver, the ridge penalty, the most steps (default 300), the key and the restarts.
 * @returns The embeddings, the loss and whether the solver converged (of the best start).
 * @example
 * ```ts
 * import { neuralTheoremProver, ntpTrainingSet, trainNtp } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const ntp = neuralTheoremProver({ facts: ['p(a,b)', 'p(b,c)', 'p(c,a)'], rules: [] }, key(0))
 * Number.isFinite(trainNtp(ntp, ntpTrainingSet(ntp, key(1)), { maxSteps: 20 }).loss) // => true
 * ```
 */
export function trainNtp(
  prover: NeuralTheoremProver,
  examples: readonly NtpExample[],
  options: {
    readonly solver?: SpaceValue
    readonly l2?: number
    readonly maxSteps?: number
    readonly key?: Key
    /** Starts: the prover's own initial embeddings, then fresh ones from the key; the lowest loss wins (default 1). */
    readonly restarts?: number
  } = {},
): NtpFit {
  const key = options.key ?? makeKey(0)
  let best: NtpFit | null = null
  for (let r = 0; r < (options.restarts ?? 1); r++) {
    const alg = ntpAlgorithm(prover, examples, {
      ...options,
      key,
      start: r === 0 ? prover.x0 : prover.init(foldIn(key, r)),
    })
    const run = runAlgorithm(alg as unknown as Algorithm<unknown>, { key, maxSteps: options.maxSteps ?? 300 })
    const s = run.state as { x: Float64Array; value: number }
    if (best === null || s.value < best.loss)
      best = { theta: Array.from(s.x), loss: s.value, converged: run.status === 'converged', steps: run.steps }
  }
  return best as NtpFit
}

/**
 * Read the proof scores of goals at trained embeddings.
 *
 * @param prover - The prover.
 * @param theta - The embeddings.
 * @param goals - Ground atoms.
 * @returns One score per goal.
 * @example
 * ```ts
 * import { ntpScores, neuralTheoremProver } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * const ntp = neuralTheoremProver({ facts: ['p(a,b)'], rules: [] }, key(0))
 * ntpScores(ntp, ntp.x0, [{ predicate: 'p', args: ['a', 'b'] }]) // => [1]
 * ```
 */
export function ntpScores(prover: NeuralTheoremProver, theta: readonly number[], goals: readonly NtpAtom[]): number[] {
  return goals.map((g) => Number(toFlat(prover.score(theta, g))[0]))
}
