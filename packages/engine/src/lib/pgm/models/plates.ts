import { WhiteboxError } from '../../base/index.ts'
import { topologicalOrder } from '../graphs.ts'
import { type BayesNet, type BayesNode, bayesNet } from './bayes-net.ts'

/** A template variable of a plate model: repeated once per element of each of its plates, with a shared CPT. */
export interface PlateVariable {
  readonly name: string
  readonly states: readonly string[]
  /** The plates the variable is indexed by. */
  readonly plates: readonly string[]
  /** Template parents; their plates must be among this variable's plates. */
  readonly parents: readonly string[]
  /** The shared CPT over [...parents, node], row-major (the first parent slowest). */
  readonly cpt: readonly number[]
}

/**
 * A plate model (a template Bayes network in plate notation; Buntine 1994, Koller and Friedman chapter 6): template
 * variables indexed by plates (students, courses, documents ...) with CPTs shared by all their instances.
 *
 * @concept plate-models
 */
export interface PlateModel {
  readonly kind: 'plateModel'
  readonly plates: readonly string[]
  readonly variables: readonly PlateVariable[]
}

/**
 * Build and validate a plate model: parents must exist and sit in a subset of the child's plates, the template must be
 * acyclic, and CPT sizes must match.
 *
 * @param variables - The template variables.
 * @returns The model (variables in topological order, plates in first-use order).
 * @throws WhiteboxError with code `value` for unknown parents, parents outside the child's plates, cycles, or CPTs of
 * the wrong size.
 * @concept plate-models
 * @example
 * ```ts
 * import { plateModel } from 'whitebox-engine/pgm'
 * const m = plateModel([
 *   { name: 'skill', states: ['lo', 'hi'], plates: ['student'], parents: [], cpt: [0.5, 0.5] },
 *   { name: 'pass', states: ['no', 'yes'], plates: ['student', 'exam'], parents: ['skill'], cpt: [0.6, 0.4, 0.1, 0.9] },
 * ])
 * m.plates // => ['student', 'exam']
 * ```
 */
export function plateModel(variables: readonly PlateVariable[]): PlateModel {
  const byName = new Map(variables.map((v) => [v.name, v]))
  if (byName.size !== variables.length) throw new WhiteboxError('value', 'plateModel: variable names must be distinct')
  for (const v of variables) {
    for (const p of v.parents) {
      const parent = byName.get(p)
      if (parent === undefined) throw new WhiteboxError('value', `plateModel: ${v.name} has unknown parent ${p}`)
      for (const plate of parent.plates) {
        if (!v.plates.includes(plate))
          throw new WhiteboxError('value', `plateModel: parent ${p} is in plate ${plate}, which ${v.name} is not in`)
      }
    }
    const rows = v.parents.reduce((k, p) => k * (byName.get(p)?.states.length ?? 1), 1)
    if (v.cpt.length !== rows * v.states.length)
      throw new WhiteboxError('value', `plateModel: ${v.name} needs ${rows * v.states.length} CPT entries`)
  }
  let ordered: PlateVariable[]
  try {
    ordered = topologicalOrder(variables)
  } catch (e) {
    throw new WhiteboxError('value', `plateModel: ${(e as Error).message}`)
  }
  const plates = [...new Set(variables.flatMap((v) => v.plates))]
  return { kind: 'plateModel', plates, variables: ordered }
}

/**
 * The name of a ground instance of a template variable: `name[i1,i2]` with the indices in the variable's plate order
 * (just `name` for a variable in no plate).
 *
 * @param variable - The template variable.
 * @param index - The element of each plate.
 * @returns The ground name.
 * @example
 * ```ts
 * import { groundName } from 'whitebox-engine/pgm'
 * groundName({ name: 'grade', states: [], plates: ['student', 'course'], parents: [], cpt: [] }, { student: 'ann', course: 'cs1' }) // => 'grade[ann,cs1]'
 * ```
 */
export function groundName(variable: PlateVariable, index: Readonly<Record<string, string>>): string {
  if (variable.plates.length === 0) return variable.name
  return `${variable.name}[${variable.plates.map((p) => index[p] ?? '?').join(',')}]`
}

/** Options of {@link groundPlateModel}. */
export interface GroundingOptions {
  /**
   * For a variable whose plates form a relation, the index tuples that exist (in the variable's plate order), for
   * example which students took which courses; by default every combination exists.
   */
  readonly relations?: Readonly<Record<string, readonly (readonly string[])[]>>
}

/** Every index tuple of some plates. */
function product(
  plates: readonly string[],
  elements: Readonly<Record<string, readonly string[]>>,
): Record<string, string>[] {
  let out: Record<string, string>[] = [{}]
  for (const p of plates) {
    const es = elements[p]
    if (es === undefined) throw new WhiteboxError('value', `ground: no elements for plate ${p}`)
    out = out.flatMap((t) => es.map((e) => ({ ...t, [p]: e })))
  }
  return out
}

/**
 * Ground a plate model against a set of objects: one Bayes-network node per template variable and element of its
 * plates (or per tuple of a given relation), each sharing its template's CPT, with parents found by projecting the
 * child's index onto the parent's plates.
 *
 * @param model - The plate model.
 * @param elements - The elements of each plate.
 * @param options - Relations restricting which index tuples exist.
 * @returns The ground Bayes network.
 * @throws WhiteboxError with code `value` for plates without elements or relation tuples of the wrong length.
 * @concept plate-models
 * @example
 * ```ts
 * import { plateModel, groundPlateModel } from 'whitebox-engine/pgm'
 * const m = plateModel([
 *   { name: 'skill', states: ['lo', 'hi'], plates: ['student'], parents: [], cpt: [0.5, 0.5] },
 *   { name: 'pass', states: ['no', 'yes'], plates: ['student', 'exam'], parents: ['skill'], cpt: [0.6, 0.4, 0.1, 0.9] },
 * ])
 * groundPlateModel(m, { student: ['ann', 'bo'], exam: ['e1'] }).nodes.map((n) => n.name) // => ['skill[ann]', 'skill[bo]', 'pass[ann,e1]', 'pass[bo,e1]']
 * ```
 */
export function groundPlateModel(
  model: PlateModel,
  elements: Readonly<Record<string, readonly string[]>>,
  options: GroundingOptions = {},
): BayesNet {
  const byName = new Map(model.variables.map((v) => [v.name, v]))
  const nodes: BayesNode[] = []
  for (const v of model.variables) {
    const relation = options.relations?.[v.name]
    const tuples =
      relation === undefined
        ? product(v.plates, elements)
        : relation.map((t) => {
            if (t.length !== v.plates.length)
              throw new WhiteboxError('value', `ground: ${v.name} needs ${v.plates.length}-tuples`)
            return Object.fromEntries(v.plates.map((p, i) => [p, t[i] ?? '']))
          })
    for (const index of tuples) {
      nodes.push({
        name: groundName(v, index),
        states: v.states,
        parents: v.parents.map((p) => groundName(byName.get(p) as PlateVariable, index)),
        cpt: v.cpt,
      })
    }
  }
  return bayesNet(nodes)
}

/** One observation of a template variable: the plate elements it belongs to and the observed state. */
export interface PlateObservation {
  readonly variable: string
  readonly index: Readonly<Record<string, string>>
  readonly state: string
}

/**
 * Ground observations against a plate model: the evidence of the ground network, by ground name.
 *
 * @param model - The plate model.
 * @param observations - Observations of template variables at plate elements.
 * @returns Evidence (state names) by ground variable.
 * @throws WhiteboxError with code `value` for unknown variables or states.
 * @concept plate-models
 * @example
 * ```ts
 * import { plateModel, groundEvidence } from 'whitebox-engine/pgm'
 * const m = plateModel([{ name: 'skill', states: ['lo', 'hi'], plates: ['student'], parents: [], cpt: [0.5, 0.5] }])
 * groundEvidence(m, [{ variable: 'skill', index: { student: 'ann' }, state: 'hi' }]) // => { 'skill[ann]': 'hi' }
 * ```
 */
export function groundEvidence(model: PlateModel, observations: readonly PlateObservation[]): Record<string, string> {
  const out: Record<string, string> = {}
  for (const o of observations) {
    const v = model.variables.find((x) => x.name === o.variable)
    if (v === undefined) throw new WhiteboxError('value', `groundEvidence: unknown variable ${o.variable}`)
    if (!v.states.includes(o.state))
      throw new WhiteboxError('value', `groundEvidence: ${o.variable} has no state ${o.state}`)
    out[groundName(v, o.index)] = o.state
  }
  return out
}

/**
 * Maximum-likelihood estimates of a plate model's shared CPTs from fully observed ground data: the counts of every
 * instance of a template family are pooled (parameter sharing), and unobserved parent configurations get uniform
 * rows.
 *
 * @param model - The plate model (its CPTs are replaced).
 * @param elements - The elements of each plate.
 * @param observations - Observations of every ground variable that is used.
 * @param options - Relations restricting which index tuples exist.
 * @returns The model with fitted CPTs.
 * @concept plate-models
 * @example
 * ```ts
 * import { plateModel, fitPlateModel } from 'whitebox-engine/pgm'
 * const m = plateModel([{ name: 'coin', states: ['h', 't'], plates: ['toss'], parents: [], cpt: [0.5, 0.5] }])
 * const obs = ['h', 'h', 't', 'h'].map((state, i) => ({ variable: 'coin', index: { toss: String(i) }, state }))
 * fitPlateModel(m, { toss: ['0', '1', '2', '3'] }, obs).variables[0]?.cpt // => [0.75, 0.25]
 * ```
 */
export function fitPlateModel(
  model: PlateModel,
  elements: Readonly<Record<string, readonly string[]>>,
  observations: readonly PlateObservation[],
  options: GroundingOptions = {},
): PlateModel {
  const evidence = groundEvidence(model, observations)
  const byName = new Map(model.variables.map((v) => [v.name, v]))
  const variables = model.variables.map((v) => {
    const k = v.states.length
    const rows = v.parents.reduce((r, p) => r * (byName.get(p)?.states.length ?? 1), 1)
    const counts = Array.from({ length: rows }, () => new Array<number>(k).fill(0))
    const relation = options.relations?.[v.name]
    const tuples =
      relation === undefined
        ? product(v.plates, elements)
        : relation.map((t) => Object.fromEntries(v.plates.map((p, i) => [p, t[i] ?? ''])))
    for (const index of tuples) {
      const own = evidence[groundName(v, index)]
      if (own === undefined) continue
      let j = 0
      let complete = true
      for (const p of v.parents) {
        const pv = byName.get(p) as PlateVariable
        const s = evidence[groundName(pv, index)]
        if (s === undefined) {
          complete = false
          break
        }
        j = j * pv.states.length + pv.states.indexOf(s)
      }
      if (!complete) continue
      const row = counts[j] as number[]
      const x = v.states.indexOf(own)
      row[x] = (row[x] ?? 0) + 1
    }
    const cpt = counts.flatMap((row) => {
      const n = row.reduce((a, b) => a + b, 0)
      return n > 0 ? row.map((c) => c / n) : row.map(() => 1 / k)
    })
    return { ...v, cpt }
  })
  return { ...model, variables }
}
