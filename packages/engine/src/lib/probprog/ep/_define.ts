/**
 * Declaring an atom: `defineAtom` checks what an atom says of itself (a name and its factors, an `ep` made of EP knobs, `extends`
 * together with `recovery`) and gives it its default knobs and the problem its knobs make, the three parts composed. Internal to the
 * folder: the atoms of `_atoms/` are written with it, and a user looks an atom up by name.
 */

import { OmaiError } from '../../base/index.ts'
import { defaults } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { checkKnobKeys, epRunKnobs } from '../inference/index.ts'
import type { AtomParts, AtomProblem, EpAtom } from './atom-types.ts'

/** What an atom declares: an `EpAtom` without what `defineAtom` makes of it (the default knobs, the composed problem, `expected` defaulted to `runs`). */
export interface AtomSpec<K extends Config> extends Omit<EpAtom<K>, 'expected' | 'problem' | 'defaults'> {
  readonly expected?: 'runs' | 'planned'
}

/**
 * The three parts of an atom as one problem record, field by field (a getter stays a getter, so a reference that is computed
 * when first read is not computed here).
 *
 * @param parts - The parts.
 * @returns The composed problem.
 * @example
 * ```ts
 * import { composeProblem, gaussianAtom } from 'omai-engine/probprog/ep'
 * Object.keys(composeProblem(gaussianAtom.parts(gaussianAtom.defaults))).includes('equations') // => true
 * ```
 */
export function composeProblem(parts: AtomParts): AtomProblem {
  const out = {}
  for (const part of [parts.build, parts.reference ?? {}, parts.present])
    Object.defineProperties(out, Object.getOwnPropertyDescriptors(part))
  return out as AtomProblem
}

/**
 * Declare an atom.
 *
 * @param spec - What the atom declares.
 * @returns The atom, with its default knobs.
 * @throws OmaiError with code `value` for an atom with no name, no factors, a `planned` atom with no plan, one that
 * gives `extends` without `recovery` (or the reverse), or an `ep` with a key that is not an EP knob.
 */
export function defineAtom<K extends Config>(spec: AtomSpec<K>): EpAtom<K> {
  if (spec.name === '' || spec.factors.length === 0)
    throw new OmaiError('value', 'defineAtom: an atom has a name and the factors it exercises')
  if (spec.expected === 'planned' && (spec.plan === undefined || spec.plan === ''))
    throw new OmaiError('value', `defineAtom: the planned atom '${spec.name}' says where its rule is planned`)
  if (spec.ep !== undefined) checkKnobKeys(`defineAtom: '${spec.name}'`, spec.ep, epRunKnobs('ep'))
  if ((spec.extends === undefined) !== (spec.recovery === undefined))
    throw new OmaiError('value', `defineAtom: '${spec.name}' gives both 'extends' and 'recovery' or neither`)
  return {
    ...spec,
    expected: spec.expected ?? 'runs',
    problem: (knobs) => composeProblem(spec.parts(knobs)),
    defaults: defaults(spec.knobs) as K,
  }
}
