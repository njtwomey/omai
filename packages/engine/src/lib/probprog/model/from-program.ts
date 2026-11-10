/**
 * Lifting a program to a spec, as a best effort that says what it keeps: `fromProgram` runs a `sample` and `observe`
 * program under a trace several times, at different random values, and rebuilds a `ModelSpec` of its site structure: the
 * names, the distribution families, the plates with their sizes, the observed values and the parameters as constants where
 * they do not move with the random values. A parameter that does move (`loc` of a site that depends on another site) is
 * not recoverable from a trace, so it becomes an opaque `custom` node that names the sites it depends on and is evaluated by
 * running the program again with those values substituted. The spec can be drawn (`toStructure`), analysed (`analyseSpec`,
 * `structureReport`) and run (`toProgram`), but anything that reads expressions (conjugacy through a mean, message passing)
 * sees the opaque nodes and says so. A program whose structure depends on random values (a site that exists only on one
 * branch, a plate whose size is drawn) is refused, with the first difference found.
 *
 * @index function fromProgram: lift a program to a ModelSpec of its site structure; refuses a varying one.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { broadcastTo, type NestedArray, primalOf, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { unexpanded } from '../_expand.ts'
import { isolated } from '../_stack.ts'
import { analyse } from '../analysis.ts'
import { type ProgramTrace, type Site, seed, substitute, traceOf } from '../handlers.ts'
import type { Model } from '../primitives.ts'
import { validateSpec } from './check.ts'
import { type CustomFn, type CustomFns, ModelDefinition } from './definition.ts'
import { distributionFamilies, type FamilyDef } from './dists/families.ts'
import {
  type DimSpec,
  type DistSpec,
  type ExprSpec,
  MODEL_SPEC_SCHEMA,
  type ModelNode,
  type ModelSpec,
  OPERATOR_REGISTRY,
} from './spec.ts'

/** Options of {@link fromProgram}. */
export interface LiftProgramOptions {
  /** The key of the prototype run and the base of the perturbed runs (default key(0)). */
  readonly key?: Key
  /** How many runs, at different random values, the structure is compared across (default 4, at least 2). */
  readonly trials?: number
  /** The spec's name (default `program`). */
  readonly name?: string
}

/** A parameter, deterministic value or factor that moves with the random values and so became an opaque `custom` node. */
export interface OpaqueLift {
  /** The node it belongs to. */
  readonly node: string
  /** The parameter of the node's distribution (`loc`), or `value` for a deterministic node. */
  readonly parameter: string
  /** The name of the `custom` function. */
  readonly custom: string
  /** The latent and param sites it was found to depend on. */
  readonly dependsOn: readonly string[]
}

/** A program lifted to a spec. */
export interface LiftedProgram {
  /** The spec with its `custom` functions (which run the program again), ready for `bind`, `toProgram` and the analyses. */
  readonly definition: ModelDefinition
  readonly spec: ModelSpec
  /** The parameters recovered as constants, as `site.parameter`. */
  readonly constants: readonly string[]
  /** What could not be recovered as an expression: a parameter that depends on other sites. */
  readonly opaque: readonly OpaqueLift[]
  /** What the lift does not capture, in sentences. */
  readonly notes: readonly string[]
}

interface SiteShape {
  readonly names: readonly string[]
  readonly sizes: readonly number[]
}

function platesOf(site: Site): SiteShape {
  if (site.type === 'plate' || site.type === 'markov') return { names: [], sizes: [] }
  const frames = [...site.plates].sort((a, b) => a.dim - b.dim)
  frames.forEach((f, i) => {
    if (f.dim !== -(frames.length - i))
      throw new OmaiError(
        'value',
        `fromProgram: the plates of '${site.name}' do not use consecutive batch dimensions`,
        {
          site: site.name,
        },
      )
    if (f.subsampleSize !== f.size)
      throw new OmaiError('value', `fromProgram: the plate '${f.name}' of '${site.name}' is subsampled`, {
        site: site.name,
      })
  })
  return { names: frames.map((f) => f.name), sizes: frames.map((f) => f.size) }
}

const kindOf = (s: Site): string =>
  s.type === 'sample' ? (s.isFactor ? 'factor' : s.isObserved ? 'observed' : 'latent') : s.type

/** The structure of one run as lines that must agree across runs. */
function signature(t: ProgramTrace): string[] {
  const out: string[] = []
  for (const s of t.sites.values()) {
    if (s.type === 'plate' || s.type === 'markov') continue
    const p = platesOf(s)
    const family = s.type === 'sample' ? unexpanded(s.fn).name : ''
    out.push(
      `${s.name}: ${kindOf(s)} ${family} plates [${p.names.map((n, i) => `${n}=${p.sizes[i]}`).join(', ')}] value shape [${s.value.shape.join(', ')}]`,
    )
  }
  return out
}

const nested = (v: Value): NestedArray => toArray(primalOf(v))

function rankOf(x: unknown): number {
  let r = 0
  let v = x
  while (Array.isArray(v)) {
    r += 1
    v = v[0]
  }
  return r
}

/** The number of trailing axes a parameter of a family has of its own (a categorical's `probs` has one). */
function paramRank(def: FamilyDef, p: string): number {
  const group = def.oneOf.find((g) => g.includes(p))
  for (const q of [p, ...(group ?? [])]) {
    const e = def.example[q]
    if (e !== undefined) return rankOf(e)
  }
  return 0
}

/** The parameters of a family that a distribution holds, one of each group of alternatives. */
function paramNames(def: FamilyDef, held: Readonly<Record<string, Value>>): string[] {
  const out: string[] = []
  for (const p of def.params) {
    const group = def.oneOf.find((g) => g.includes(p))
    if (group !== undefined) {
      if (p === group.find((q) => held[q] !== undefined)) out.push(p)
    } else if (held[p] !== undefined) out.push(p)
  }
  return out
}

const same = (a: Value, b: Value): boolean => {
  const x = toFlat(primalOf(a))
  const y = toFlat(primalOf(b))
  return x.length === y.length && x.every((v, i) => Math.abs(v - (y[i] as number)) <= 1e-12 * (1 + Math.abs(v)))
}

/**
 * Lift a program to a spec of its site structure (a best-effort tracing lift). The program runs under a trace `trials`
 * times at different random values; the structure of every run must be the same (the sites, their kinds and families, the
 * plates and their sizes, the value shapes), or the lift is refused with the first difference. The spec holds each latent,
 * observed, param and deterministic site with its family, plates, observed value and parameters; a parameter that is the
 * same in every run is a constant, and one that moves is an opaque `custom` node that names the latent and param sites it
 * depends on and is evaluated by running the program with those values substituted (so `toProgram` of the lifted spec has
 * the program's density). What is not recovered: any expression (`mu + tau * z` is `custom(mu, tau, z)`), constraints
 * beyond a family's support, the markov structure (a loop is unrolled into the sites it made), and anything the trace did
 * not run (a branch the runs did not take). Detection of a structure that depends on random values is by sampling: a
 * branch taken with small probability can be missed, so raise `trials` for programs that branch.
 *
 * @param program - The program.
 * @param data - Its data.
 * @param options - The key, the number of runs and the spec's name.
 * @returns The lifted program, with what became opaque and what the lift does not capture.
 * @throws OmaiError with code `value` when the structure depends on random values, for a site that is a factor or is scaled or masked, uses a
 *   family or plate the spec cannot express (a subsampled plate, a plate size that changes, non-consecutive dimensions), or
 *   a name that is used for two plates of different sizes.
 * @concept model-spec
 * @index function fromProgram: lift a sample/observe program to a ModelSpec of its site structure; refuses a varying structure.
 * @example
 * ```ts
 * import { observe, plate, sample } from 'omai-engine/probprog'
 * import { fromProgram, structureReport } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/dists'
 * const program = () => { const mu = sample('mu', normal(0, 2)); plate('n', 3, () => observe('y', normal(mu, 1), [1, 2, 3])) }
 * const lifted = fromProgram(program, undefined)
 * // The mean of y moves with mu, so it is an opaque custom node and no conjugacy is claimed through it.
 * [lifted.spec.nodes.map((n) => n.name), lifted.opaque.map((o) => o.custom), structureReport(lifted.definition).conjugateBlocks.length] // => [['mu', 'y'], ['y.loc'], 0]
 * ```
 */
export function fromProgram<D = void>(program: Model<D>, data: D, options: LiftProgramOptions = {}): LiftedProgram {
  const key = options.key ?? makeKey(0)
  const trials = Math.max(2, options.trials ?? 4)
  const families = distributionFamilies()
  const a = analyse(program, data, { key })
  const proto = a.prototype
  const runs: ProgramTrace[] = [proto]
  for (let k = 1; k < trials; k++) runs.push(traceOf(seed(program, foldIn(key, 1000 + k)), data))
  const base = signature(proto)
  runs.forEach((t, k) => {
    const sig = signature(t)
    const i = base.findIndex((line, j) => line !== sig[j])
    if (base.length !== sig.length || i >= 0)
      throw new OmaiError(
        'value',
        `fromProgram: the program's structure depends on random values: run 0 has '${base[Math.max(i, 0)] ?? '(no site)'}' where run ${k} has '${sig[Math.max(i, 0)] ?? '(no site)'}'`,
        { run: k },
      )
  })

  const dims = new Map<string, number>()
  const nodes: ModelNode[] = []
  const customs: Record<string, CustomFn> = {}
  const constants: string[] = []
  const opaque: OpaqueLift[] = []
  const parentsOf = new Map(a.sites.map((s) => [s.name, s.parents]))
  const roleOf = new Map(a.sites.map((s) => [s.name, s.role]))
  const order = [...proto.sites.values()].filter((s) => s.type !== 'plate' && s.type !== 'markov')
  const isValueNode = (name: string): boolean => {
    const r = roleOf.get(name)
    return r === 'latent' || r === 'param'
  }

  /** The latent and param sites a site depends on, through deterministic sites. */
  const ancestors = (name: string): string[] => {
    const before = order
      .slice(
        0,
        order.findIndex((s) => s.name === name),
      )
      .map((s) => s.name)
    if (!a.dependenciesExact) return before.filter(isValueNode)
    const out = new Set<string>()
    const seen = new Set<string>()
    const walk = (n: string): void => {
      for (const p of parentsOf.get(n) ?? []) {
        if (seen.has(p)) continue
        seen.add(p)
        if (isValueNode(p)) out.add(p)
        else if (roleOf.get(p) === 'deterministic') walk(p)
      }
    }
    walk(name)
    return before.filter((n) => out.has(n))
  }

  const opaqueExpr = (
    site: string,
    parameter: string,
    values: readonly Value[],
    batch: SiteShape,
    rank: number,
    read: (t: ProgramTrace) => Value,
  ): ExprSpec => {
    const first = values[0] as Value
    if (values.every((v) => same(first, v))) {
      constants.push(`${site}.${parameter}`)
      // No batch axes of its own: it broadcasts over the plates when the spec runs.
      if (first.shape.length <= rank) return { kind: 'const', value: nested(first) }
      const event = first.shape.slice(first.shape.length - rank)
      return {
        kind: 'const',
        value: nested(broadcastTo(primalOf(first), [...batch.sizes, ...event])),
        dims: batch.names,
      }
    }
    const dep = ancestors(site)
    const custom = `${site}.${parameter}`
    opaque.push({ node: site, parameter, custom, dependsOn: dep })
    customs[custom] = (args) => {
      const given: Record<string, Value> = {}
      args.forEach((x, i) => {
        given[dep[i] as string] = x.value
      })
      // Run apart from the handlers of the program this node is part of, and keep the values tracked for autodiff.
      const v = isolated(() => read(traceOf(seed(substitute(program, given), makeKey(0)), data)))
      const event = v.shape.slice(v.shape.length - rank)
      return broadcastTo(v, [...batch.sizes, ...event])
    }
    return {
      kind: 'custom',
      name: custom,
      args: dep.map((name) => ({ kind: 'ref', name })),
      dims: batch.names,
      eventRank: rank,
    }
  }

  for (const s of order) {
    const batch = platesOf(s)
    batch.names.forEach((n, i) => {
      const size = batch.sizes[i] as number
      if (dims.has(n) && dims.get(n) !== size)
        throw new OmaiError('value', `fromProgram: the plate name '${n}' is used for sizes ${dims.get(n)} and ${size}`)
      dims.set(n, size)
    })
    const at = (t: ProgramTrace): Site => t.sites.get(s.name) as Site
    if (s.type === 'sample') {
      if (s.isFactor)
        throw new OmaiError('value', `fromProgram: '${s.name}' is a factor site; factors are not lifted`, {
          site: s.name,
        })
      if (s.scale !== 1 || s.mask !== undefined)
        throw new OmaiError('value', `fromProgram: '${s.name}' is scaled or masked, which the spec cannot express`, {
          site: s.name,
        })
      const fam = unexpanded(s.fn)
      const def = families[fam.name]
      if (def === undefined)
        throw new OmaiError(
          'value',
          `fromProgram: '${s.name}' is a ${fam.name}, which is not a family of the spec layer (${Object.keys(families).join(', ')})`,
          { site: s.name },
        )
      const params: Record<string, ExprSpec> = {}
      for (const p of paramNames(def, fam.params)) {
        const rank = paramRank(def, p)
        params[p] = opaqueExpr(
          s.name,
          p,
          runs.map((t) => unexpanded((at(t) as typeof s).fn).params[p] as Value),
          batch,
          rank,
          (t) => unexpanded((at(t) as typeof s).fn).params[p] as Value,
        )
      }
      const dist: DistSpec = { family: fam.name, params }
      const eventRank = fam.eventShape.length
      if (s.isObserved) {
        if (!runs.every((t) => same(s.value, (at(t) as typeof s).value)))
          throw new OmaiError('value', `fromProgram: the observed value of '${s.name}' changes between runs`, {
            site: s.name,
          })
        nodes.push({
          kind: 'observed',
          name: s.name,
          dims: batch.names,
          eventRank,
          dist,
          value: { kind: 'const', value: nested(s.value), dims: batch.names },
        })
      } else nodes.push({ kind: 'latent', name: s.name, dims: batch.names, eventRank, dist })
    } else if (s.type === 'param') {
      nodes.push({
        kind: 'param',
        name: s.name,
        dims: batch.names,
        eventRank: s.eventDim,
        init: nested(s.value) as never,
      })
    } else if (s.type === 'deterministic') {
      const rank = s.value.shape.length
      const expr = opaqueExpr(
        s.name,
        'value',
        runs.map((t) => (at(t) as typeof s).value),
        { names: [], sizes: [] },
        rank,
        (t) => (at(t) as typeof s).value,
      )
      nodes.push({ kind: 'derived', name: s.name, dims: [], eventRank: rank, expr, record: true })
    }
  }
  const dimSpecs: DimSpec[] = [...dims].map(([name, size]) => ({ name, size }))
  const spec: ModelSpec = {
    schema: MODEL_SPEC_SCHEMA,
    name: options.name ?? 'program',
    registry: OPERATOR_REGISTRY,
    dims: dimSpecs,
    nodes,
    customs: Object.keys(customs),
  }
  validateSpec(spec)
  const notes = [
    'the spec holds the sites, families, plates and observed values of the traced runs; expressions are not recovered',
    ...(opaque.length > 0
      ? [
          `${opaque.length} parameter${opaque.length === 1 ? '' : 's'} moved with the random values and became opaque custom nodes (${opaque
            .map((o) => o.custom)
            .join(', ')}): conjugacy through them and message passing do not see through`,
        ]
      : []),
    'a branch that none of the runs took is not in the spec',
  ]
  return {
    definition: new ModelDefinition(spec, customs as CustomFns),
    spec,
    constants,
    opaque,
    notes,
  }
}
