import type { HubEntry } from '../../entry/entry.ts'
import { miningEntry } from './_mining.ts'

/** Apriori: frequent itemsets level by level. */
export const aprioriEntry: HubEntry = miningEntry({
  name: 'apriori',
  title: 'Apriori',
  summary: 'Count itemsets level by level; a candidate with an infrequent subset is pruned before it is ever counted.',
  miner: 'apriori',
  boardTitle: 'The itemset lattice, level by level (item:count)',
  method: { file: 'methods/patterns/itemsets.ts', name: 'apriori' },
  references: ['Agrawal and Srikant (1994), Fast algorithms for mining association rules.'],
})
