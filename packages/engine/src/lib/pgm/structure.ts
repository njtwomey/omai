import {
  type NodeRole,
  type Structure,
  type StructureEdge,
  type StructureGroup,
  type StructureNode,
  structure,
} from '../scene/index.ts'
import type { Dag } from './graphs.ts'
import type { BayesNet } from './models/bayes-net.ts'
import type { GeneralCrf, LinearChainCrf } from './models/crf.ts'
import type { DynamicBayesNet } from './models/dynamic.ts'
import type { FactorGraph } from './models/factor-graph.ts'
import type { InfluenceDiagram } from './models/influence.ts'
import type { Gmrf, GmrfInformation, LinearGaussianBayesNet } from './models/linear-gaussian.ts'
import type { MarkovNetwork } from './models/markov-network.ts'
import type { PlateModel } from './models/plates.ts'
import type { HiddenMarkovModel, LinearDynamicalSystem } from './models/state-space.ts'

/** Every model type of `pgm` that describes itself as a {@link Structure}. */
export type PgmModel =
  | BayesNet
  | Dag
  | MarkovNetwork
  | FactorGraph
  | LinearGaussianBayesNet
  | Gmrf
  | GmrfInformation
  | DynamicBayesNet
  | HiddenMarkovModel
  | LinearDynamicalSystem
  | LinearChainCrf
  | GeneralCrf
  | PlateModel
  | InfluenceDiagram

const directed = (from: string, to: string, kind: StructureEdge['kind'] = 'dependency'): StructureEdge => ({
  from,
  to,
  directed: true,
  kind,
})
const undirected = (from: string, to: string): StructureEdge => ({ from, to, directed: false, kind: 'dependency' })

function variableNode(
  name: string,
  observed: readonly string[],
  detail?: string,
  badges?: Record<string, number>,
  group?: string,
): StructureNode {
  const role: NodeRole = observed.includes(name) ? 'observed' : 'latent'
  return {
    id: name,
    label: name,
    role,
    ...(detail === undefined ? {} : { detail }),
    ...(badges === undefined ? {} : { badges }),
    ...(group === undefined ? {} : { group }),
  }
}

/** Two slices of a repeated template: nodes `<name>_<t-1>` and `<name>_<t>`, grouped by slice. */
function slices(
  type: string,
  title: string,
  names: readonly { name: string; role: NodeRole; detail?: string }[],
  intra: readonly [string, string][],
  inter: readonly [string, string][],
): Structure {
  const groups: StructureGroup[] = [
    { id: 'slice-prev', label: 't - 1', kind: 'repeat' },
    { id: 'slice-now', label: 't', kind: 'repeat' },
  ]
  const nodes: StructureNode[] = ['prev', 'now'].flatMap((s) =>
    names.map((n) => ({
      id: `${n.name}_${s === 'prev' ? 't-1' : 't'}`,
      label: n.name,
      role: n.role,
      group: `slice-${s}`,
      ...(n.detail === undefined ? {} : { detail: n.detail }),
    })),
  )
  const edges: StructureEdge[] = [
    ...['t-1', 't'].flatMap((t) => intra.map(([a, b]) => directed(`${a}_${t}`, `${b}_${t}`))),
    ...inter.map(([a, b]) => directed(`${a}_t-1`, `${b}_t`)),
  ]
  return structure(type, nodes, edges, { groups, title, direction: 'right' })
}

/**
 * Any graphical model of `pgm` as a {@link Structure}, so it can be drawn (and its inference animated from the
 * `message` events of a trace): Bayes networks and DAGs with directed edges, Markov networks and Gaussian Markov
 * random fields with undirected edges, factor graphs with `factor` nodes, dynamic models (DBNs, HMMs, linear
 * dynamical systems, linear-chain CRFs) as two grouped slices, plate models with `plate` groups, and influence
 * diagrams with `decision` and `utility` nodes.
 *
 * @param model - The model.
 * @param observed - Variables to show as observed.
 * @returns The structure.
 * @concept structures
 * @example
 * ```ts
 * import { bayesNet, toStructure } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * toStructure(bn).edges // => [{ from: 'a', to: 'b', directed: true, kind: 'dependency' }]
 * ```
 */
export function toStructure(model: PgmModel, observed: readonly string[] = []): Structure {
  switch (model.kind) {
    case 'bayesNet':
      return structure(
        'bayes-net',
        model.nodes.map((n) =>
          variableNode(n.name, observed, n.states.join(', '), { states: n.states.length, cpt: n.cpt.length }),
        ),
        model.nodes.flatMap((n) => n.parents.map((p) => directed(p, n.name))),
        { title: 'Bayes network', direction: 'down' },
      )
    case 'dag':
      return structure(
        'dag',
        model.nodes.map((n) => variableNode(n.name, observed)),
        model.nodes.flatMap((n) => n.parents.map((p) => directed(p, n.name))),
        { title: 'DAG', direction: 'down' },
      )
    case 'linearGaussianBayesNet':
      return structure(
        'linear-gaussian',
        model.nodes.map((n) => variableNode(n.name, observed, `N(${n.intercept} + w.pa, ${n.variance})`)),
        model.nodes.flatMap((n) =>
          n.parents.map((p, i) => ({ ...directed(p, n.name), label: String(n.weights[i] ?? 0) })),
        ),
        { title: 'Linear-Gaussian network', direction: 'down' },
      )
    case 'markovNetwork':
      return structure(
        'markov-network',
        model.variables.map((v) => variableNode(v.name, observed, v.states.join(', '), { states: v.states.length })),
        model.edges.map(([a, b]) => undirected(a, b)),
        { title: 'Markov network' },
      )
    case 'gmrf':
    case 'gmrfInformation': {
      const edges: StructureEdge[] = []
      for (let i = 0; i < model.variables.length; i++)
        for (let j = i + 1; j < model.variables.length; j++)
          if ((model.precision[i]?.[j] ?? 0) !== 0)
            edges.push(undirected(model.variables[i] ?? '', model.variables[j] ?? ''))
      return structure(
        'gmrf',
        model.variables.map((v) => variableNode(v, observed)),
        edges,
        { title: 'Gaussian Markov random field' },
      )
    }
    case 'factorGraph':
      return structure(
        'factor-graph',
        [
          ...model.variables.map((v) => variableNode(v.name, observed, v.states.join(', '))),
          ...model.factors.map(
            (f, i): StructureNode => ({
              id: `f${i}`,
              label: `f${i}`,
              role: 'factor',
              detail: f.variables.join(', '),
              badges: { entries: f.logValues.length },
            }),
          ),
        ],
        model.factors.flatMap((f, i) => f.variables.map((v) => undirected(`f${i}`, v))),
        { title: 'Factor graph' },
      )
    case 'dynamicBayesNet':
      return slices(
        'dynamic-bayes-net',
        'Dynamic Bayes network',
        model.transition.map((n) => ({ name: n.name, role: observed.includes(n.name) ? 'observed' : 'latent' })),
        model.transition.flatMap((n) => n.parents.map((p): [string, string] => [p, n.name])),
        model.transition.flatMap((n) => n.previous.map((p): [string, string] => [p, n.name])),
      )
    case 'hmm':
      return slices(
        'hmm',
        'Hidden Markov model',
        [
          { name: 'z', role: 'latent', detail: model.states.join(', ') },
          { name: 'x', role: 'observed', detail: model.emission.kind },
        ],
        [['z', 'x']],
        [['z', 'z']],
      )
    case 'lds':
      return slices(
        'linear-dynamical-system',
        'Linear dynamical system',
        [
          { name: 'x', role: 'latent', detail: `state (${model.initialMean.length})` },
          { name: 'y', role: 'observed', detail: `observation (${model.observation.length})` },
        ],
        [['x', 'y']],
        [['x', 'x']],
      )
    case 'linearChainCrf': {
      const ts = ['t-1', 't', 't+1']
      return structure(
        'linear-chain-crf',
        ts.flatMap((t): StructureNode[] => [
          { id: `y_${t}`, label: `y ${t}`, role: 'latent', detail: model.labels.join(', ') },
          { id: `x_${t}`, label: `x ${t}`, role: 'observed', badges: { attributes: model.attributes.length } },
        ]),
        [
          ...ts.map((t) => directed(`x_${t}`, `y_${t}`, 'data')),
          undirected('y_t-1', 'y_t'),
          undirected('y_t', 'y_t+1'),
        ],
        { title: 'Linear-chain CRF', direction: 'right' },
      )
    }
    case 'crf':
      return structure(
        'crf',
        model.variables.map((v) => variableNode(v.name, observed, v.states.join(', '))),
        [],
        { title: 'Conditional random field' },
      )
    case 'plateModel': {
      const groupOf = (plates: readonly string[]): string | undefined =>
        plates.length === 0 ? undefined : `plate:${plates.join('×')}`
      const groups: StructureGroup[] = [
        ...new Set(model.variables.map((v) => groupOf(v.plates)).filter((g): g is string => g !== undefined)),
      ].map((id) => ({ id, label: id.slice('plate:'.length).split('×').join(' × '), kind: 'plate' as const }))
      return structure(
        'plate-model',
        model.variables.map((v) => variableNode(v.name, observed, v.states.join(', '), undefined, groupOf(v.plates))),
        model.variables.flatMap((v) => v.parents.map((p) => directed(p, v.name))),
        { groups, title: 'Plate model', direction: 'down' },
      )
    }
    case 'influenceDiagram':
      return structure(
        'influence-diagram',
        [
          ...model.chance.map((n) => variableNode(n.name, observed, n.states.join(', '))),
          ...model.decisions.map(
            (d): StructureNode => ({ id: d.name, label: d.name, role: 'decision', detail: d.states.join(', ') }),
          ),
          ...model.utilities.map((u): StructureNode => ({ id: u.name, label: u.name, role: 'utility' })),
        ],
        [
          ...model.chance.flatMap((n) => n.parents.map((p) => directed(p, n.name))),
          ...model.decisions.flatMap((d) => d.parents.map((p) => directed(p, d.name, 'data'))),
          ...model.utilities.flatMap((u) => u.parents.map((p) => directed(p, u.name))),
        ],
        { title: 'Influence diagram', direction: 'down' },
      )
    default:
      return structure('unknown', [], [])
  }
}
