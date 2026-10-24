import { type NodeRole, type Structure, type StructureGroup, type StructureNode, structure } from '../scene/index.ts'
import { analyse, type ProgramAnalysis, type SiteRole } from './analysis.ts'
import type { Model } from './primitives.ts'

const ROLE: Readonly<Record<SiteRole, NodeRole>> = {
  latent: 'latent',
  observed: 'observed',
  factor: 'factor',
  param: 'parameter',
  deterministic: 'deterministic',
}

/**
 * Describe a program as a graphical model (section 3.13 of the design), so any program draws like a Bayes network:
 * a node per site with its role (latent, observed, factor, parameter, deterministic), distribution and shape, a
 * directed dependency edge per traced dependency, nested plate groups with their sizes, and a repeat group per markov
 * loop.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - An analysis already made.
 * @returns The structure.
 * @concept structures
 * @example
 * ```ts
 * import { observe, plate, toStructure, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * const model = () => { const mu = sample('mu', normal(0, 1)); plate('data', 3, () => observe('y', normal(mu, 1), [1, 2, 3])) }
 * const s = toStructure(model, undefined)
 * s.nodes.map((n) => [n.id, n.role, n.group ?? '']) // => [['mu', 'latent', ''], ['y', 'observed', 'plate:data']]
 * s.groups // => [{ id: 'plate:data', label: 'data', kind: 'plate', count: 3 }]
 * ```
 */
export function toStructure<D = void>(
  model: Model<D>,
  data: D,
  options: { readonly analysis?: ProgramAnalysis } = {},
): Structure {
  return analysisStructure(options.analysis ?? analyse(model, data))
}

/**
 * The structure of an analysed program (see {@link toStructure}).
 *
 * @param a - The analysis.
 * @returns The structure.
 * @concept structures
 * @example
 * ```ts
 * import { analyse, analysisStructure, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * analysisStructure(analyse(() => { sample('x', normal(0, 1)) }, undefined)).type // => 'probabilistic-program'
 * ```
 */
export function analysisStructure(a: ProgramAnalysis): Structure {
  const groups: StructureGroup[] = []
  const seen = new Set<string>()
  const plateParents = new Map<string, string | undefined>()
  for (const site of a.prototype.sites.values()) {
    if (site.type === 'plate') {
      const outer = site.plates[0]?.name
      plateParents.set(site.name, outer === undefined ? undefined : `plate:${outer}`)
    }
  }
  for (const p of a.plates) {
    const id = `plate:${p.name}`
    if (seen.has(id)) continue
    seen.add(id)
    const parent = plateParents.get(p.name)
    groups.push({ id, label: p.name, kind: 'plate', ...(parent === undefined ? {} : { parent }), count: p.size })
  }
  for (const m of a.markov) groups.push({ id: `markov:${m.name}`, label: m.name, kind: 'repeat', count: m.length })
  const nodes: StructureNode[] = a.sites.map((s) => {
    const inner = a.prototype.sites.get(s.name)
    const plates = inner === undefined || inner.type === 'markov' ? [] : inner.plates
    const group =
      plates[0] !== undefined
        ? `plate:${plates[0].name}`
        : s.markov !== undefined
          ? `markov:${s.markov.name}`
          : undefined
    const badges: Record<string, string | number> = {}
    if (s.shape.length > 0) badges['shape'] = `[${s.shape.join(', ')}]`
    if (s.cardinality !== undefined) badges['states'] = s.cardinality
    if (s.enumerated) badges['enumerated'] = 'yes'
    return {
      id: s.name,
      label: s.name,
      role: ROLE[s.role],
      ...(group === undefined ? {} : { group }),
      ...(s.distribution === undefined ? {} : { detail: s.distribution }),
      ...(Object.keys(badges).length === 0 ? {} : { badges }),
    }
  })
  const edges = a.edges.map((e) => ({ from: e.from, to: e.to, directed: true, kind: 'dependency' as const }))
  return structure('probabilistic-program', nodes, edges, {
    groups,
    title: 'Probabilistic program',
    direction: 'down',
    ...(a.dependenciesExact
      ? {}
      : { annotations: [{ text: 'dependencies assumed from execution order', kind: 'warning' as const }] }),
  })
}
