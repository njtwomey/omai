import type { HubEntry } from '../../entry/entry.ts'
import { miningEntry } from './_mining.ts'

/** FP-growth: frequent itemsets from a prefix tree, without candidates. */
export const fpGrowthEntry: HubEntry = miningEntry({
  name: 'fp-growth',
  title: 'FP-growth',
  summary: 'Compress the baskets into a prefix tree in two passes, then mine each item’s prefix paths: no candidates.',
  miner: 'fp-growth',
  boardTitle: 'The FP-tree growing, then each item mined (item:count)',
  method: { file: 'methods/patterns/itemsets.ts', name: 'fpGrowth' },
  references: ['Han, Pei and Yin (2000), Mining frequent patterns without candidate generation.'],
})
