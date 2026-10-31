/**
 * Dung's abstract argumentation frameworks: arguments and an attack relation, and the sets of arguments that can be
 * accepted together. Conflict-free and admissible sets (those that defend themselves against every attacker), the
 * characteristic function, the grounded extension (its least fixed point, the sceptical core, computed as a labelling
 * that marks arguments in, out and undecided one round per step), complete, preferred (maximal admissible) and stable
 * (conflict-free sets attacking everything outside them) extensions, with enumeration by subsets for frameworks of up to
 * 20 arguments.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Algorithm, channels, type TraceSchema } from '../trace/index.ts'

/** An argumentation framework: arguments and attacks `[attacker, attacked]`. */
export interface ArgumentationFramework {
  readonly arguments: readonly string[]
  readonly attacks: readonly (readonly [string, string])[]
}

/**
 * A framework, checked: attacks must name known arguments.
 *
 * @param args - The arguments.
 * @param attacks - Pairs `[a, b]` meaning a attacks b.
 * @returns The framework.
 * @throws MdmlError with code `value` for an unknown argument or a repeated argument name.
 * @example
 * ```ts
 * import { framework } from 'mdml-engine/logic'
 * framework(['a', 'b'], [['a', 'b']]).attacks.length // => 1
 * ```
 */
export function framework(
  args: readonly string[],
  attacks: readonly (readonly [string, string])[],
): ArgumentationFramework {
  if (new Set(args).size !== args.length) throw new MdmlError('value', 'argument names must be distinct')
  for (const [a, b] of attacks)
    if (!args.includes(a) || !args.includes(b))
      throw new MdmlError('value', `attack ${a} -> ${b} names an unknown argument`)
  return { arguments: [...args], attacks: attacks.map(([a, b]) => [a, b] as const) }
}

const attackersOf = (f: ArgumentationFramework, a: string): string[] =>
  f.attacks.filter(([, t]) => t === a).map(([s]) => s)

/**
 * Whether no argument of the set attacks another of it.
 *
 * @param f - The framework.
 * @param set - A set of arguments.
 * @returns True when the set is conflict-free.
 * @example
 * ```ts
 * import { conflictFree, framework } from 'mdml-engine/logic'
 * const f = framework(['a', 'b', 'c'], [['a', 'b']])
 * [conflictFree(f, ['a', 'c']), conflictFree(f, ['a', 'b'])] // => [true, false]
 * ```
 */
export function conflictFree(f: ArgumentationFramework, set: readonly string[]): boolean {
  return !f.attacks.some(([a, b]) => set.includes(a) && set.includes(b))
}

/**
 * Whether a set defends an argument: every attacker of the argument is attacked by some member of the set.
 *
 * @param f - The framework.
 * @param set - The defending set.
 * @param a - The argument.
 * @returns True when every attack on `a` is counter-attacked from the set.
 * @example
 * ```ts
 * import { defends, framework } from 'mdml-engine/logic'
 * defends(framework(['a', 'b', 'c'], [['a', 'b'], ['b', 'c']]), ['a'], 'c') // => true
 * ```
 */
export function defends(f: ArgumentationFramework, set: readonly string[], a: string): boolean {
  return attackersOf(f, a).every((b) => attackersOf(f, b).some((c) => set.includes(c)))
}

/**
 * The characteristic function: the arguments a set defends.
 *
 * @param f - The framework.
 * @param set - A set of arguments.
 * @returns The arguments defended by the set.
 * @example
 * ```ts
 * import { characteristic, framework } from 'mdml-engine/logic'
 * characteristic(framework(['a', 'b'], [['a', 'b']]), []) // => ['a']
 * ```
 */
export function characteristic(f: ArgumentationFramework, set: readonly string[]): string[] {
  return f.arguments.filter((a) => defends(f, set, a))
}

/**
 * Whether a set is admissible: conflict-free and defending each of its members.
 *
 * @param f - The framework.
 * @param set - A set of arguments.
 * @returns True when admissible.
 * @example
 * ```ts
 * import { admissible, framework } from 'mdml-engine/logic'
 * const f = framework(['a', 'b'], [['a', 'b'], ['b', 'a']])
 * [admissible(f, ['a']), admissible(f, ['a', 'b'])] // => [true, false]
 * ```
 */
export function admissible(f: ArgumentationFramework, set: readonly string[]): boolean {
  return conflictFree(f, set) && set.every((a) => defends(f, set, a))
}

function subsets(f: ArgumentationFramework): string[][] {
  if (f.arguments.length > 20) throw new MdmlError('value', 'extension enumeration supports at most 20 arguments')
  const n = f.arguments.length
  const out: string[][] = []
  for (let mask = 0; mask < 1 << n; mask++) out.push(f.arguments.filter((_, i) => (mask >> i) & 1))
  return out
}

const subset = (a: readonly string[], b: readonly string[]): boolean => a.every((x) => b.includes(x))
const sameSet = (a: readonly string[], b: readonly string[]): boolean => a.length === b.length && subset(a, b)

/**
 * The grounded extension: the least fixed point of the characteristic function, reached by applying it from the empty set.
 *
 * @param f - The framework.
 * @returns The grounded extension, in the framework's argument order.
 * @example
 * ```ts
 * import { framework, groundedExtension } from 'mdml-engine/logic'
 * groundedExtension(framework(['a', 'b', 'c'], [['a', 'b'], ['b', 'c']])) // => ['a', 'c']
 * ```
 */
export function groundedExtension(f: ArgumentationFramework): string[] {
  let s: string[] = []
  for (;;) {
    const next = characteristic(f, s)
    if (sameSet(next, s)) return s
    s = next
  }
}

/**
 * The complete extensions: admissible sets containing every argument they defend.
 *
 * @param f - The framework.
 * @returns The complete extensions.
 * @throws MdmlError with code `value` for more than 20 arguments.
 * @example
 * ```ts
 * import { completeExtensions, framework } from 'mdml-engine/logic'
 * completeExtensions(framework(['a', 'b'], [['a', 'b'], ['b', 'a']])).map((e) => e.join()) // => ['', 'a', 'b']
 * ```
 */
export function completeExtensions(f: ArgumentationFramework): string[][] {
  return subsets(f).filter((s) => admissible(f, s) && sameSet(characteristic(f, s), s))
}

/**
 * The preferred extensions: the maximal admissible sets. Every framework has at least one (possibly the empty set).
 *
 * @param f - The framework.
 * @returns The preferred extensions.
 * @throws MdmlError with code `value` for more than 20 arguments.
 * @example
 * ```ts
 * import { framework, preferredExtensions } from 'mdml-engine/logic'
 * preferredExtensions(framework(['a', 'b'], [['a', 'b'], ['b', 'a']])).map((e) => e.join()) // => ['a', 'b']
 * ```
 */
export function preferredExtensions(f: ArgumentationFramework): string[][] {
  const adm = subsets(f).filter((s) => admissible(f, s))
  return adm.filter((s) => !adm.some((t) => t.length > s.length && subset(s, t)))
}

/**
 * The stable extensions: conflict-free sets that attack every argument outside them. A framework may have none (an odd
 * cycle).
 *
 * @param f - The framework.
 * @returns The stable extensions.
 * @throws MdmlError with code `value` for more than 20 arguments.
 * @example
 * ```ts
 * import { framework, stableExtensions } from 'mdml-engine/logic'
 * stableExtensions(framework(['a', 'b', 'c'], [['a', 'b'], ['b', 'c'], ['c', 'a']])) // => []
 * ```
 */
export function stableExtensions(f: ArgumentationFramework): string[][] {
  return subsets(f).filter(
    (s) =>
      conflictFree(f, s) && f.arguments.every((a) => s.includes(a) || attackersOf(f, a).some((b) => s.includes(b))),
  )
}

/** Whether an argument is accepted: sceptically (in every extension) or credulously (in some) under a semantics. */
export type Semantics = 'grounded' | 'complete' | 'preferred' | 'stable'

/**
 * The extensions under a semantics.
 *
 * @param f - The framework.
 * @param semantics - Which semantics.
 * @returns The extensions (one for grounded).
 * @throws MdmlError with code `value` for more than 20 arguments (except grounded).
 * @example
 * ```ts
 * import { extensions, framework } from 'mdml-engine/logic'
 * extensions(framework(['a', 'b'], [['a', 'b']]), 'grounded') // => [['a']]
 * ```
 */
export function extensions(f: ArgumentationFramework, semantics: Semantics): string[][] {
  return semantics === 'grounded'
    ? [groundedExtension(f)]
    : semantics === 'complete'
      ? completeExtensions(f)
      : semantics === 'preferred'
        ? preferredExtensions(f)
        : stableExtensions(f)
}

/**
 * Whether an argument is accepted: in every extension (sceptical) or some extension (credulous).
 *
 * @param f - The framework.
 * @param a - The argument.
 * @param semantics - Which semantics.
 * @param mode - `sceptical` or `credulous`.
 * @returns True when accepted (a sceptical test over no extension is vacuously true).
 * @example
 * ```ts
 * import { accepted, framework } from 'mdml-engine/logic'
 * const f = framework(['a', 'b'], [['a', 'b'], ['b', 'a']])
 * [accepted(f, 'a', 'preferred', 'credulous'), accepted(f, 'a', 'preferred', 'sceptical')] // => [true, false]
 * ```
 */
export function accepted(
  f: ArgumentationFramework,
  a: string,
  semantics: Semantics,
  mode: 'sceptical' | 'credulous',
): boolean {
  const ext = extensions(f, semantics)
  return mode === 'sceptical' ? ext.every((e) => e.includes(a)) : ext.some((e) => e.includes(a))
}

/** The label of an argument in a labelling. */
export type Label = 'in' | 'out' | 'undecided'

/** The state of the grounded labelling. */
export interface LabellingState {
  readonly labels: Readonly<Record<string, Label>>
  readonly round: number
  readonly changed: boolean
}

const LABELLING_SCHEMA: TraceSchema = channels({
  metrics: {
    in: { description: 'arguments labelled in' },
    out: { description: 'arguments labelled out' },
    undecided: { description: 'arguments still undecided' },
  },
  messages: {
    round: { description: 'the arguments labelled in this round', payload: { in: 'newly in', out: 'newly out' } },
  },
})

/**
 * The grounded labelling as a traced algorithm, one round per step: an undecided argument is labelled in when all its
 * attackers are out (it is defended), and out when some attacker is in. Arguments never labelled stay undecided; when a
 * round changes nothing the labelling is complete and its `in` arguments are the grounded extension.
 *
 * @param f - The framework.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { framework, groundedAlgorithm } from 'mdml-engine/logic'
 * groundedAlgorithm(framework(['a'], [])).name // => 'grounded-labelling'
 * ```
 */
export function groundedAlgorithm(f: ArgumentationFramework): Algorithm<LabellingState> {
  const last: { in: string[]; out: string[] } = { in: [], out: [] }
  return {
    name: 'grounded-labelling',
    config: { arguments: [...f.arguments] },
    schema: LABELLING_SCHEMA,
    init: () => ({
      labels: Object.fromEntries(f.arguments.map((a) => [a, 'undecided' as Label])),
      round: 0,
      changed: true,
    }),
    step: (s) => {
      const labels: Record<string, Label> = { ...s.labels }
      const becomeIn = f.arguments.filter(
        (a) => labels[a] === 'undecided' && attackersOf(f, a).every((b) => labels[b] === 'out'),
      )
      for (const a of becomeIn) labels[a] = 'in'
      const becomeOut = f.arguments.filter(
        (a) => labels[a] === 'undecided' && attackersOf(f, a).some((b) => labels[b] === 'in'),
      )
      for (const a of becomeOut) labels[a] = 'out'
      last.in = becomeIn
      last.out = becomeOut
      return { labels, round: s.round + 1, changed: becomeIn.length + becomeOut.length > 0 }
    },
    done: (s) => (s.changed ? undefined : 'converged'),
    observe: (s) => {
      const count = (l: Label): number => Object.values(s.labels).filter((x) => x === l).length
      return {
        metrics: { in: count('in'), out: count('out'), undecided: count('undecided') },
        messages:
          s.round === 0
            ? []
            : [{ kind: 'round', payload: { in: [...last.in], out: [...last.out] } as unknown as Json }],
      }
    },
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as LabellingState,
  }
}
