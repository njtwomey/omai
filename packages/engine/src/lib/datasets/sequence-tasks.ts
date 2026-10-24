import { type DatasetGenerator, type Generated, type Knobs, type SequenceData, sequences } from '../data/index.ts'
import { foldIn, type Key, randint, uniform } from '../random/index.ts'
import { bool, choice, dict, int } from '../space/index.ts'
import { stamp } from './_util.ts'

/** The special tokens shared by every task: padding, separator and answer markers. */
const SPECIAL = ['<pad>', '<sep>'] as const

function task(
  name: string,
  description: string,
  knobs: Knobs,
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
  n: int(1, 1_000_000, { default: 1000 }),
  minLength: int(1, 1000, { default: 1 }),
  maxLength: int(1, 1000, { default: 10 }),
  symbols: int(2, 1000, { default: 8 }),
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
 * import { generate } from 'whitebox-engine/data'
 * import { copyTask } from 'whitebox-engine/datasets'
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
 * import { generate } from 'whitebox-engine/data'
 * import { reverseTask } from 'whitebox-engine/datasets'
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
 * import { generate } from 'whitebox-engine/data'
 * import { sortTask } from 'whitebox-engine/datasets'
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
 * import { generate } from 'whitebox-engine/data'
 * import { dyckTask } from 'whitebox-engine/datasets'
 * generate(dyckTask, { knobs: { types: 2 } }).data.vocabulary // => ['<pad>', '<sep>', 'no', 'yes', '(', ')', '[', ']']
 * ```
 */
export const dyckTask: DatasetGenerator<SequenceData> = {
  name: 'dyckTask',
  kind: 'sequence',
  description: 'Decide whether a string of brackets is balanced (Dyck-k recognition).',
  knobs: dict({
    n: int(1, 1_000_000, { default: 1000 }),
    halfLength: int(1, 500, { default: 6, label: 'bracket pairs per string' }),
    types: int(1, 4, { default: 2, label: 'kinds of brackets' }),
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
 * import { generate } from 'whitebox-engine/data'
 * import { additionTask } from 'whitebox-engine/datasets'
 * (generate(additionTask).truth.extra?.['solve'] as (s: number[]) => number[])([3, 4, 12, 5, 6]) // => [6, 8]
 * ```
 */
export const additionTask: DatasetGenerator<SequenceData> = {
  name: 'additionTask',
  kind: 'sequence',
  description: 'Add two multi-digit numbers given as digit tokens.',
  knobs: dict({
    n: int(1, 1_000_000, { default: 1000 }),
    digits: int(1, 50, { default: 3, label: 'maximum digits per operand' }),
    reverseDigits: bool({ default: false, label: 'least significant digit first' }),
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
 * import { generate } from 'whitebox-engine/data'
 * import { associativeRecall } from 'whitebox-engine/datasets'
 * generate(associativeRecall, { knobs: { pairs: 3 } }).data.inputs.shape[1] // => 8
 * ```
 */
export const associativeRecall: DatasetGenerator<SequenceData> = {
  name: 'associativeRecall',
  kind: 'sequence',
  description: 'Recall the value paired with a query key earlier in the sequence.',
  knobs: dict({
    n: int(1, 1_000_000, { default: 1000 }),
    pairs: int(1, 26, { default: 5 }),
    keys: int(2, 26, { default: 26, label: 'size of the key alphabet' }),
    values: int(2, 100, { default: 10, label: 'size of the value alphabet' }),
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
 * import { generate } from 'whitebox-engine/data'
 * import { parityTask } from 'whitebox-engine/datasets'
 * (generate(parityTask, { knobs: { running: true } }).truth.extra?.['solve'] as (s: number[]) => number[])([3, 3, 2]) // => [3, 2, 2]
 * ```
 */
export const parityTask: DatasetGenerator<SequenceData> = {
  name: 'parityTask',
  kind: 'sequence',
  description: 'Compute the parity of a bit string (or of every prefix).',
  knobs: dict({
    n: int(1, 1_000_000, { default: 1000 }),
    minLength: int(1, 10_000, { default: 1 }),
    maxLength: int(1, 10_000, { default: 16 }),
    running: bool({ default: false, label: 'label every prefix' }),
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
 * import { generate } from 'whitebox-engine/data'
 * import { modularArithmetic } from 'whitebox-engine/datasets'
 * generate(modularArithmetic, { knobs: { modulus: 7, all: true } }).data.inputs.shape // => [49, 4]
 * ```
 */
export const modularArithmetic: DatasetGenerator<SequenceData> = {
  name: 'modularArithmetic',
  kind: 'sequence',
  description: 'Equations a op b = c modulo a prime p.',
  knobs: dict({
    modulus: int(2, 10_000, { default: 97, label: 'modulus p (prime for division)' }),
    operation: choice(['+', '-', '*', '/'], { default: '+' }),
    all: bool({ default: true, label: 'the whole table' }),
    n: int(1, 1_000_000, { default: 1000 }),
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
