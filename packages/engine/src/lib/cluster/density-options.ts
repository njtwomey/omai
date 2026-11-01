/**
 * The slot of HDBSCAN that varies as an option table: how flat clusters are selected from the condensed tree, either by
 * excess of mass (the default: a parent replaces its children when it is more stable than they are together) or as
 * the leaves of the tree (many small, homogeneous clusters). The table is one `union` space for controls, grids and
 * tuners, and builds the rule from a configuration.
 */

import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'

/** How clusters are chosen from the condensed tree. */
export type ClusterSelectionRule = 'eom' | 'leaf'

/** What a selection rule does, for validity rules and selectors. */
export interface ClusterSelectionProperties {
  /** Whether the rule can return clusters nested at different depths of the tree. */
  readonly mixedDepth: boolean
}

function rule(
  description: string,
  name: ClusterSelectionRule,
  mixedDepth: boolean,
): Option<ClusterSelectionRule, ClusterSelectionProperties> {
  return { description, properties: { mixedDepth }, space: dict({}), make: () => name }
}

/**
 * The cluster selection rules of HDBSCAN as one slot: `eom` (excess of mass: walk up from the leaves, keeping a
 * cluster when its stability exceeds the sum of its children's) and `leaf` (every leaf of the condensed tree).
 *
 * @concept density-clustering
 * @example
 * ```ts
 * import { hdbscanSelections } from 'mdml-engine/cluster'
 * hdbscanSelections.make({ kind: 'leaf' }) // => 'leaf'
 * ```
 */
export const hdbscanSelections: OptionTable<ClusterSelectionRule, ClusterSelectionProperties> = optionTable(
  {
    eom: rule('excess of mass: keep a cluster when it is more stable than its descendants together', 'eom', true),
    leaf: rule('the leaves of the condensed tree: many small, homogeneous clusters', 'leaf', false),
  },
  {
    default: 'eom',
    label: 'cluster selection',
    description: 'How flat clusters are chosen from the condensed tree.',
  },
)
