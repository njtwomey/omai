/**
 * Algorithmic sequence tasks for sequence models: copying, reversal, sorting, Dyck-language recognition, multi-digit
 * addition, associative recall (the induction-head benchmark), parity, and modular arithmetic (the grokking benchmark),
 * each as padded input and target token sequences.
 */

import {
  type DatasetGenerator,
  type Generated,
  type GeneratorKnobs,
  type SequenceData,
  sequences,
} from '../data/index.ts'
import { solve as solveLinear } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal, randint, uniform } from '../random/index.ts'
import { bool, choice, dict, int, real } from '../space/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { stamp } from './_util.ts'

/** The special tokens shared by every task: padding, separator and answer markers. */
const SPECIAL = ['<pad>', '<sep>'] as const

function task(
  name: string,
  description: string,
  knobs: GeneratorKnobs,
  inputs: readonly (readonly number[])[],
  targets: readonly (readonly number[])[],
  vocabulary: readonly string[],
  solve: (input: readonly number[]) => number[],
): Generated<SequenceData> {
  return {
    data: sequences(inputs, targets, vocabulary, { padId: 0, meta: stamp(name, description, knobs) }),
    truth: {
      extra: {
        solve,
        accuracy: (predicted: readonly (readonly number[])[]) =>
          predicted.reduce((s, p, i) => s + (p.join(',') === (targets[i] ?? []).join(',') ? 1 : 0), 0) /
          Math.max(1, predicted.length),
      },
    },
  }
}

/** Random symbol strings: lengths uniform in [minLength, maxLength], symbols `offset .. offset + symbols - 1`. */
function strings(k: Key, n: number, minLength: number, maxLength: number, symbols: number, offset: number): number[][] {
  const lens = randint(foldIn(k, 0), [n], minLength, maxLength + 1).data
  return Array.from({ length: n }, (_, i) =>
    Array.from(randint(foldIn(foldIn(k, 1), i), [lens[i] ?? minLength], offset, offset + symbols).data),
  )
}

function symbolVocabulary(symbols: number): string[] {
  return [...SPECIAL, ...Array.from({ length: symbols }, (_, i) => `s${i}`)]
}

const lengthKnobs = {
  n: int(1, 1_000_000, { default: 1000, label: 'sequences', description: 'The number of sequences.' }),
  minLength: int(1, 1000, { default: 1, label: 'shortest sequence', description: 'The minimum sequence length.' }),
  maxLength: int(1, 1000, { default: 10, label: 'longest sequence', description: 'The maximum sequence length.' }),
  symbols: int(2, 1000, { default: 8, label: 'alphabet size', description: 'The number of distinct symbols.' }),
}

function stringTask(
  name: string,
  description: string,
  f: (s: readonly number[]) => number[],
): DatasetGenerator<SequenceData> {
  const generator: DatasetGenerator<SequenceData> = {
    name,
    kind: 'sequence',
    description,
    knobs: dict(lengthKnobs),
    generate: (k, knobs) => {
      const inputs = strings(
        k,
        knobs['n'] as number,
        knobs['minLength'] as number,
        Math.max(knobs['minLength'] as number, knobs['maxLength'] as number),
        knobs['symbols'] as number,
        SPECIAL.length,
      )
      return task(name, description, knobs, inputs, inputs.map(f), symbolVocabulary(knobs['symbols'] as number), f)
    },
  }
  return generator
}

/**
 * The copy task: the target repeats the input string.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { copyTask } from 'mdml-engine/datasets'
 * const { data } = generate(copyTask, { knobs: { n: 5 } })
 * Array.from(data.inputs.data).join() === Array.from(data.targets.data).join() // => true
 * ```
 */
export const copyTask: DatasetGenerator<SequenceData> = stringTask('copyTask', 'Copy the input string.', (s) => [...s])

/**
 * The reversal task: the target is the input reversed.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { reverseTask } from 'mdml-engine/datasets'
 * (generate(reverseTask).truth.extra?.['solve'] as (s: number[]) => number[])([2, 3, 4]) // => [4, 3, 2]
 * ```
 */
export const reverseTask: DatasetGenerator<SequenceData> = stringTask('reverseTask', 'Reverse the input string.', (s) =>
  [...s].reverse(),
)

/**
 * The sorting task: the target is the input symbols in increasing order.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { sortTask } from 'mdml-engine/datasets'
 * (generate(sortTask).truth.extra?.['solve'] as (s: number[]) => number[])([5, 2, 4]) // => [2, 4, 5]
 * ```
 */
export const sortTask: DatasetGenerator<SequenceData> = stringTask('sortTask', 'Sort the input symbols.', (s) =>
  [...s].sort((a, b) => a - b),
)

/** Whether a bracket string (opening id `2b`, closing `2b + 1` relative to the first bracket token) is balanced. */
function balanced(s: readonly number[], first: number): boolean {
  const stack: number[] = []
  for (const t of s) {
    const r = t - first
    if (r % 2 === 0) stack.push(r)
    else if (stack.pop() !== r - 1) return false
  }
  return stack.length === 0
}

/**
 * Recognising a Dyck language (balanced strings of `types` kinds of brackets): half the inputs are random balanced
 * strings of the given length, half are balanced strings with one bracket replaced (so almost always unbalanced);
 * the target is `yes` or `no`, decided exactly by the truth's stack recogniser.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { dyckTask } from 'mdml-engine/datasets'
 * generate(dyckTask, { knobs: { types: 2 } }).data.vocabulary // => ['<pad>', '<sep>', 'no', 'yes', '(', ')', '[', ']']
 * ```
 */
export const dyckTask: DatasetGenerator<SequenceData> = {
  name: 'dyckTask',
  kind: 'sequence',
  description: 'Decide whether a string of brackets is balanced (Dyck-k recognition).',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'strings', description: 'The number of bracket strings.', default: 1000 }),
    halfLength: int(1, 500, {
      description: 'The number of bracket pairs in each string.',
      default: 6,
      label: 'bracket pairs per string',
    }),
    types: int(1, 4, { description: 'The number of bracket kinds (Dyck-k).', default: 2, label: 'kinds of brackets' }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const pairs = knobs['halfLength'] as number
    const types = knobs['types'] as number
    const glyphs = ['(', ')', '[', ']', '{', '}', '<', '>'].slice(0, 2 * types)
    const vocab = [...SPECIAL, 'no', 'yes', ...glyphs]
    const first = 4
    const inputs: number[][] = []
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, i)
      // A random balanced string: open or close with equal probability while both keep the string completable.
      const u = uniform(foldIn(ki, 0), [2 * pairs]).data
      const kinds = randint(foldIn(ki, 1), [pairs], 0, types).data
      const s: number[] = []
      const open: number[] = []
      let opened = 0
      for (let t = 0; t < 2 * pairs; t++) {
        const remaining = 2 * pairs - t
        const canOpen = opened < pairs
        const mustClose = open.length === remaining
        if (canOpen && !mustClose && (open.length === 0 || (u[t] ?? 0) < 0.5)) {
          const b = kinds[opened] ?? 0
          opened++
          open.push(b)
          s.push(first + 2 * b)
        } else {
          const b = open.pop() ?? 0
          s.push(first + 2 * b + 1)
        }
      }
      if (i % 2 === 1) {
        const pos = randint(foldIn(ki, 2), [1], 0, s.length).data[0] ?? 0
        const repl = randint(foldIn(ki, 3), [1], 0, 2 * types - 1).data[0] ?? 0
        const cur = (s[pos] ?? first) - first
        s[pos] = first + (repl >= cur ? repl + 1 : repl)
      }
      inputs.push(s)
    }
    const solve = (s: readonly number[]) => [balanced(s, first) ? 3 : 2]
    return task('dyckTask', dyckTask.description, knobs, inputs, inputs.map(solve), vocab, solve)
  },
}

/**
 * Multi-digit addition: input `a + b` as digit tokens, target the digits of `a + b`, optionally with every number
 * written least significant digit first (which makes the carries local, a common trick for sequence models).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { additionTask } from 'mdml-engine/datasets'
 * (generate(additionTask).truth.extra?.['solve'] as (s: number[]) => number[])([3, 4, 12, 5, 6]) // => [6, 8]
 * ```
 */
export const additionTask: DatasetGenerator<SequenceData> = {
  name: 'additionTask',
  kind: 'sequence',
  description: 'Add two multi-digit numbers given as digit tokens.',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'problems', description: 'The number of addition problems.', default: 1000 }),
    digits: int(1, 50, {
      description: 'The largest number of digits in each operand.',
      default: 3,
      label: 'maximum digits per operand',
    }),
    reverseDigits: bool({
      description: 'Write numbers least significant digit first (easier to learn left to right).',
      default: false,
      label: 'least significant digit first',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const digits = knobs['digits'] as number
    const rev = knobs['reverseDigits'] === true
    const vocab = [...SPECIAL, ...Array.from({ length: 10 }, (_, d) => String(d)), '+']
    const plus = vocab.length - 1
    const toTokens = (x: bigint) => {
      const t = Array.from(x.toString(), (c) => 2 + Number(c))
      return rev ? t.reverse() : t
    }
    const fromTokens = (t: readonly number[]) =>
      BigInt((rev ? [...t].reverse() : t).map((d) => String(d - 2)).join('') || '0')
    const draw = (key: Key) => {
      const len = (randint(foldIn(key, 0), [1], 1, digits + 1).data[0] ?? 1) as number
      const ds = randint(foldIn(key, 1), [len], 0, 10).data
      return BigInt(
        Array.from(ds)
          .join('')
          .replace(/^0+(?=\d)/, ''),
      )
    }
    const inputs: number[][] = []
    for (let i = 0; i < n; i++) {
      const a = draw(foldIn(foldIn(k, 0), i))
      const b = draw(foldIn(foldIn(k, 1), i))
      inputs.push([...toTokens(a), plus, ...toTokens(b)])
    }
    const solve = (s: readonly number[]) => {
      const at = s.indexOf(plus)
      return toTokens(fromTokens(s.slice(0, at)) + fromTokens(s.slice(at + 1)))
    }
    return task('additionTask', additionTask.description, knobs, inputs, inputs.map(solve), vocab, solve)
  },
}

/**
 * Associative recall: a sequence of key-value pairs followed by a separator and a query key; the target is the value
 * stored with that key (keys are distinct within a sequence), the induction-head benchmark.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { associativeRecall } from 'mdml-engine/datasets'
 * generate(associativeRecall, { knobs: { pairs: 3 } }).data.inputs.shape[1] // => 8
 * ```
 */
export const associativeRecall: DatasetGenerator<SequenceData> = {
  name: 'associativeRecall',
  kind: 'sequence',
  description: 'Recall the value paired with a query key earlier in the sequence.',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'sequences', description: 'The number of sequences.', default: 1000 }),
    pairs: int(1, 26, { label: 'pairs', description: 'Key-value pairs per sequence before the query.', default: 5 }),
    keys: int(2, 26, { description: 'The number of distinct keys.', default: 26, label: 'size of the key alphabet' }),
    values: int(2, 100, {
      description: 'The number of distinct values.',
      default: 10,
      label: 'size of the value alphabet',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const nk = knobs['keys'] as number
    const pairs = Math.min(knobs['pairs'] as number, nk)
    const nv = knobs['values'] as number
    const keyTokens = Array.from({ length: nk }, (_, i) => String.fromCharCode(97 + i))
    const vocab = [...SPECIAL, ...keyTokens, ...Array.from({ length: nv }, (_, i) => `v${i}`)]
    const keyOf = (i: number) => SPECIAL.length + i
    const valueToken = (i: number) => SPECIAL.length + nk + i
    const inputs: number[][] = []
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, i)
      const order = Array.from(uniform(foldIn(ki, 0), [nk]).data, (u, j) => [u, j] as const)
        .sort((a, b) => a[0] - b[0])
        .slice(0, pairs)
        .map(([, j]) => j)
      const vals = randint(foldIn(ki, 1), [pairs], 0, nv).data
      const q = randint(foldIn(ki, 2), [1], 0, pairs).data[0] ?? 0
      const s: number[] = []
      for (const [p, key] of order.entries()) s.push(keyOf(key), valueToken(vals[p] ?? 0))
      s.push(1, keyOf(order[q] ?? 0))
      inputs.push(s)
    }
    const solve = (s: readonly number[]) => {
      const query = s[s.length - 1]
      for (let i = 0; i + 1 < s.length - 2; i += 2) if (s[i] === query) return [s[i + 1] ?? 0]
      return [0]
    }
    return task('associativeRecall', associativeRecall.description, knobs, inputs, inputs.map(solve), vocab, solve)
  },
}

/**
 * Parity of a bit string: the target is the parity bit, or with `running` the parity of every prefix (a sequence
 * labelling version that is easy for recurrent and hard for attention-only models).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { parityTask } from 'mdml-engine/datasets'
 * (generate(parityTask, { knobs: { running: true } }).truth.extra?.['solve'] as (s: number[]) => number[])([3, 3, 2]) // => [3, 2, 2]
 * ```
 */
export const parityTask: DatasetGenerator<SequenceData> = {
  name: 'parityTask',
  kind: 'sequence',
  description: 'Compute the parity of a bit string (or of every prefix).',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'strings', description: 'The number of bit strings.', default: 1000 }),
    minLength: int(1, 10_000, { label: 'shortest string', description: 'The minimum string length.', default: 1 }),
    maxLength: int(1, 10_000, { label: 'longest string', description: 'The maximum string length.', default: 16 }),
    running: bool({
      description: 'Label the parity of every prefix, not only of the whole string.',
      default: false,
      label: 'label every prefix',
    }),
  }),
  generate: (k, knobs) => {
    const vocab = [...SPECIAL, '0', '1']
    const inputs = strings(
      k,
      knobs['n'] as number,
      knobs['minLength'] as number,
      Math.max(knobs['minLength'] as number, knobs['maxLength'] as number),
      2,
      2,
    )
    const running = knobs['running'] === true
    const solve = (s: readonly number[]) => {
      let p = 0
      const out: number[] = []
      for (const t of s) {
        p ^= t - 2
        out.push(2 + p)
      }
      return running ? out : [2 + p]
    }
    return task('parityTask', parityTask.description, knobs, inputs, inputs.map(solve), vocab, solve)
  },
}

/** The operations of {@link modularArithmetic}. */
export type ModularOperation = '+' | '-' | '*' | '/'

/**
 * Modular arithmetic `a op b mod p` (the grokking benchmark of Power et al., 2022): inputs `[a, op, b, =]`, target
 * `[c]`; division uses the modular inverse of `b` (so `b` is never 0 for division). With `all`, the dataset is the
 * whole operation table (shuffled), otherwise `n` random equations.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { modularArithmetic } from 'mdml-engine/datasets'
 * generate(modularArithmetic, { knobs: { modulus: 7, all: true } }).data.inputs.shape // => [49, 4]
 * ```
 */
export const modularArithmetic: DatasetGenerator<SequenceData> = {
  name: 'modularArithmetic',
  kind: 'sequence',
  description: 'Equations a op b = c modulo a prime p.',
  knobs: dict({
    modulus: int(2, 10_000, {
      description: 'The modulus p; division needs a prime.',
      default: 97,
      label: 'modulus p (prime for division)',
    }),
    operation: choice(['+', '-', '*', '/'], {
      label: 'operation',
      description: 'The operation tabulated.',
      default: '+',
    }),
    all: bool({
      description: 'Generate the whole p × p table, instead of n random equations.',
      default: true,
      label: 'the whole table',
    }),
    n: int(1, 1_000_000, {
      label: 'equations',
      description: 'The number of random equations when not generating the whole table.',
      default: 1000,
    }),
  }),
  generate: (k, knobs) => {
    const p = knobs['modulus'] as number
    const op = knobs['operation'] as ModularOperation
    const vocab = [...SPECIAL, ...Array.from({ length: p }, (_, i) => String(i)), op, '=']
    const opToken = SPECIAL.length + p
    const eq = opToken + 1
    const num = (x: number) => SPECIAL.length + x
    const power = (b: number, e: number) => {
      let r = 1
      let base = b % p
      let ex = e
      while (ex > 0) {
        if (ex & 1) r = (r * base) % p
        base = (base * base) % p
        ex >>= 1
      }
      return r
    }
    const apply = (a: number, b: number) => {
      switch (op) {
        case '+':
          return (a + b) % p
        case '-':
          return (((a - b) % p) + p) % p
        case '*':
          return (a * b) % p
        case '/':
          return (a * power(b, p - 2)) % p
      }
    }
    const lowB = op === '/' ? 1 : 0
    let pairs: [number, number][] = []
    if (knobs['all'] === true) {
      for (let a = 0; a < p; a++) for (let b = lowB; b < p; b++) pairs.push([a, b])
      const order = Array.from(uniform(foldIn(k, 0), [pairs.length]).data, (u, i) => [u, i] as const).sort(
        (x, y) => x[0] - y[0],
      )
      pairs = order.map(([, i]) => pairs[i] ?? [0, 0])
    } else {
      const n = knobs['n'] as number
      const a = randint(foldIn(k, 1), [n], 0, p).data
      const b = randint(foldIn(k, 2), [n], lowB, p).data
      pairs = Array.from(a, (x, i) => [x, b[i] ?? lowB])
    }
    const inputs = pairs.map(([a, b]) => [num(a), opToken, num(b), eq])
    const solve = (s: readonly number[]) => [num(apply((s[0] ?? 0) - SPECIAL.length, (s[2] ?? 0) - SPECIAL.length))]
    return task('modularArithmetic', modularArithmetic.description, knobs, inputs, inputs.map(solve), vocab, solve)
  },
}

/**
 * In-context linear regression as token sequences (the transformer-theory benchmark): each task draws a weight vector
 * `w ~ N(0, I)`, then a context of `context` pairs `(x, y = w . x + noise)` and a query `x`; values are quantised to
 * `levels` bins on `[-range, range]` and written as tokens, `x_1 y_1 x_2 y_2 ... x_q` with each `x` taking `features`
 * tokens, and the target is the quantised noiseless `w . x_q`. The truth has every task's weights, the
 * least-squares and the Bayes (ridge) prediction from the (unquantised) context, and `solve`, which decodes a token
 * sequence and returns the least-squares prediction as a token.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { inContextRegression } from 'mdml-engine/datasets'
 * generate(inContextRegression, { knobs: { n: 5, context: 4, features: 1 } }).data.inputs.shape // => [5, 9]
 * ```
 */
export const inContextRegression: DatasetGenerator<SequenceData> = {
  name: 'inContextRegression',
  kind: 'sequence',
  description: 'Random linear regression tasks written as token sequences, with least-squares and Bayes predictions.',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'tasks', description: 'The number of regression tasks.', default: 1000 }),
    context: int(1, 200, {
      label: 'context pairs',
      description: 'The number of (x, y) pairs shown before the query.',
      default: 8,
    }),
    features: int(1, 8, { label: 'features', description: 'The dimension of x.', default: 1 }),
    noise: real(0, 5, {
      label: 'noise standard deviation',
      description: 'The noise on the context outputs.',
      default: 0.1,
    }),
    levels: int(8, 4096, { label: 'quantisation levels', description: 'The number of value tokens.', default: 128 }),
    range: real(0.5, 100, {
      label: 'value range',
      description: 'Values are clipped to [-range, range] before quantising.',
      default: 4,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = knobs['context'] as number
    const d = knobs['features'] as number
    const sd = knobs['noise'] as number
    const levels = knobs['levels'] as number
    const range = knobs['range'] as number
    const vocab = [...SPECIAL, ...Array.from({ length: levels }, (_, b) => `v${b}`)]
    const width = (2 * range) / levels
    const toBin = (v: number): number => Math.min(levels - 1, Math.max(0, Math.floor((v + range) / width)))
    const token = (v: number): number => SPECIAL.length + toBin(v)
    const centre = (t: number): number => -range + (t - SPECIAL.length + 0.5) * width
    // The least-squares prediction at x_q from a context of rows (xs, ys), with the ridge term `lambda`.
    const ridge = (
      xs: readonly (readonly number[])[],
      ys: readonly number[],
      xq: readonly number[],
      lambda: number,
    ): number => {
      const a = Array.from({ length: d }, (_, i) =>
        Array.from(
          { length: d },
          (_, j) => xs.reduce((s, x) => s + (x[i] ?? 0) * (x[j] ?? 0), 0) + (i === j ? lambda : 0),
        ),
      )
      const b = Array.from({ length: d }, (_, i) => xs.reduce((s, x, r) => s + (x[i] ?? 0) * (ys[r] ?? 0), 0))
      return toFlat(solveLinear(tensor(a), tensor(b))).reduce((s, w, i) => s + w * (xq[i] ?? 0), 0)
    }
    const inputs: number[][] = []
    const targets: number[][] = []
    const weights: number[][] = []
    const lsPrediction: number[] = []
    const bayesPrediction: number[] = []
    const noiseless: number[] = []
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, i)
      const z = normal(foldIn(ki, 0), [d + (m + 1) * d + m]).data
      const w = Array.from(z.subarray(0, d))
      const xs = Array.from({ length: m + 1 }, (_, r) => Array.from(z.subarray(d + r * d, d + (r + 1) * d)))
      const eps = z.subarray(d + (m + 1) * d)
      const ys = xs.slice(0, m).map((x, r) => x.reduce((s, v, j) => s + v * (w[j] ?? 0), 0) + sd * (eps[r] ?? 0))
      const xq = xs[m] ?? []
      const truthQ = xq.reduce((s, v, j) => s + v * (w[j] ?? 0), 0)
      const seq: number[] = []
      for (let r = 0; r < m; r++) {
        for (const v of xs[r] ?? []) seq.push(token(v))
        seq.push(token(ys[r] ?? 0))
      }
      for (const v of xq) seq.push(token(v))
      inputs.push(seq)
      targets.push([token(truthQ)])
      weights.push(w)
      noiseless.push(truthQ)
      lsPrediction.push(ridge(xs.slice(0, m), ys, xq, 1e-9))
      bayesPrediction.push(ridge(xs.slice(0, m), ys, xq, (sd * sd) / 1))
    }
    const solve = (input: readonly number[]): number[] => {
      const stride = d + 1
      const rows = Math.floor(input.length / stride)
      const xsIn = Array.from({ length: rows }, (_, r) =>
        Array.from({ length: d }, (_, j) => centre(input[r * stride + j] ?? SPECIAL.length)),
      )
      const ysIn = Array.from({ length: rows }, (_, r) => centre(input[r * stride + d] ?? SPECIAL.length))
      const xq = Array.from({ length: d }, (_, j) => centre(input[rows * stride + j] ?? SPECIAL.length))
      return [token(ridge(xsIn, ysIn, xq, 1e-9))]
    }
    const out = task('inContextRegression', inContextRegression.description, knobs, inputs, targets, vocab, solve)
    return {
      ...out,
      truth: {
        ...out.truth,
        noiseScale: sd,
        extra: {
          ...(out.truth.extra ?? {}),
          weights,
          lsPrediction,
          bayesPrediction,
          noiselessTarget: noiseless,
          binWidth: width,
          decode: centre,
        },
      },
    }
  },
}

/** A task whose targets sit at fixed positions of the input: the rest of the target row is padding and unsupervised. */
function alignedTask(
  name: string,
  description: string,
  knobs: GeneratorKnobs,
  inputs: readonly (readonly number[])[],
  targets: readonly (readonly number[])[],
  supervised: readonly (readonly boolean[])[],
  vocabulary: readonly string[],
  solve: (input: readonly number[]) => number[],
): Generated<SequenceData> {
  return {
    data: sequences(inputs, targets, vocabulary, {
      padId: 0,
      targetMask: supervised.map((row) => row.map((b) => (b ? 1 : 0))),
      meta: stamp(name, description, knobs),
    }),
    truth: {
      extra: {
        solve,
        supervised,
        accuracy: (predicted: readonly (readonly number[])[]) => {
          let right = 0
          let total = 0
          for (const [i, t] of targets.entries())
            for (const [j, on] of (supervised[i] ?? []).entries()) {
              if (!on) continue
              total++
              if (predicted[i]?.[j] === t[j]) right++
            }
          return right / Math.max(1, total)
        },
      },
    },
  }
}

/** `count` distinct positions out of `length`, in increasing order. */
function scatter(k: Key, length: number, count: number): number[] {
  return Array.from(uniform(k, [length]).data, (u, j) => [u, j] as const)
    .sort((a, b) => a[0] - b[0])
    .slice(0, count)
    .map(([, j]) => j)
    .sort((a, b) => a - b)
}

/**
 * Selective copying (the task of the Mamba paper, Gu and Dao 2023, after the copying task of Arjovsky et al.): `tokens`
 * content symbols sit at random positions among noise tokens in a sequence of `length`, followed by `tokens` blank
 * slots (`<sep>`); at the slots the target is the content symbols in order. Recalling them needs the model to choose
 * what to keep from the content of the input, which a time-invariant (fixed kernel) model cannot do. With `fixed` the
 * content sits in the first positions (the vanilla copying task, solvable by a fixed lag).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { selectiveCopying } from 'mdml-engine/datasets'
 * generate(selectiveCopying, { knobs: { length: 12, tokens: 3 } }).data.inputs.shape[1] // => 15
 * ```
 */
export const selectiveCopying: DatasetGenerator<SequenceData> = {
  name: 'selectiveCopying',
  kind: 'sequence',
  description: 'Copy the content symbols scattered among noise, in order, at the end of the sequence.',
  knobs: dict({
    n: int(1, 1_000_000, { default: 1000, label: 'sequences', description: 'The number of sequences.' }),
    length: int(2, 1000, { default: 16, label: 'length', description: 'Positions holding content and noise.' }),
    tokens: int(1, 100, { default: 4, label: 'content symbols', description: 'Symbols to copy.' }),
    symbols: int(2, 100, { default: 6, label: 'alphabet size', description: 'Distinct content symbols.' }),
    fixed: bool({
      default: false,
      label: 'fixed positions',
      description: 'Content in the first positions (vanilla copying).',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const length = knobs['length'] as number
    const tokens = Math.min(knobs['tokens'] as number, length)
    const symbols = knobs['symbols'] as number
    const fixed = knobs['fixed'] === true
    // 0 padding, 1 blank slot, 2 noise, then the content symbols.
    const vocab = [...SPECIAL, 'noise', ...Array.from({ length: symbols }, (_, i) => `s${i}`)]
    const first = SPECIAL.length + 1
    const inputs: number[][] = []
    const targets: number[][] = []
    const supervised: boolean[][] = []
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, i)
      const where = fixed ? Array.from({ length: tokens }, (_, j) => j) : scatter(foldIn(ki, 0), length, tokens)
      const content = Array.from(randint(foldIn(ki, 1), [tokens], first, first + symbols).data)
      const seq = new Array<number>(length).fill(2)
      where.forEach((pos, j) => {
        seq[pos] = content[j] ?? first
      })
      inputs.push([...seq, ...new Array<number>(tokens).fill(1)])
      targets.push([...new Array<number>(length).fill(0), ...content])
      supervised.push([...new Array<boolean>(length).fill(false), ...new Array<boolean>(tokens).fill(true)])
    }
    const solve = (input: readonly number[]): number[] => {
      const body = input.slice(0, length)
      return [...new Array<number>(length).fill(0), ...body.filter((t) => t >= first)]
    }
    return alignedTask(
      'selectiveCopying',
      selectiveCopying.description,
      knobs,
      inputs,
      targets,
      supervised,
      vocab,
      solve,
    )
  },
}

/**
 * Induction heads (the task of Olsson et al. 2022, used in the Mamba paper): a sequence of random tokens in which a
 * marker token occurs once, followed by a token; the sequence ends with the marker again and the target (at the last
 * position) is the token that followed the marker the first time. Solving it needs a mechanism that finds the earlier
 * marker and copies what came after it.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { inductionHeads } from 'mdml-engine/datasets'
 * generate(inductionHeads, { knobs: { length: 10 } }).data.inputs.shape[1] // => 10
 * ```
 */
export const inductionHeads: DatasetGenerator<SequenceData> = {
  name: 'inductionHeads',
  kind: 'sequence',
  description: 'After a marker seen once earlier, recall the token that followed it.',
  knobs: dict({
    n: int(1, 1_000_000, { default: 1000, label: 'sequences', description: 'The number of sequences.' }),
    length: int(4, 1000, { default: 16, label: 'length', description: 'Sequence length, the final marker included.' }),
    symbols: int(2, 100, { default: 6, label: 'alphabet size', description: 'Distinct ordinary tokens.' }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const length = knobs['length'] as number
    const symbols = knobs['symbols'] as number
    // 0 padding, 1 unused separator, 2 marker, then the ordinary tokens.
    const vocab = [...SPECIAL, 'marker', ...Array.from({ length: symbols }, (_, i) => `s${i}`)]
    const first = SPECIAL.length + 1
    const inputs: number[][] = []
    const targets: number[][] = []
    const supervised: boolean[][] = []
    for (let i = 0; i < n; i++) {
      const ki = foldIn(k, i)
      const seq = Array.from(randint(foldIn(ki, 0), [length], first, first + symbols).data)
      const at = randint(foldIn(ki, 1), [1], 0, length - 2).data[0] ?? 0
      seq[at] = 2
      seq[length - 1] = 2
      inputs.push(seq)
      targets.push([...new Array<number>(length - 1).fill(0), seq[at + 1] ?? first])
      supervised.push([...new Array<boolean>(length - 1).fill(false), true])
    }
    const solve = (input: readonly number[]): number[] => {
      const at = input.indexOf(2)
      return [...new Array<number>(input.length - 1).fill(0), input[at + 1] ?? 0]
    }
    return alignedTask('inductionHeads', inductionHeads.description, knobs, inputs, targets, supervised, vocab, solve)
  },
}
