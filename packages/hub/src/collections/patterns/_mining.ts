// Itemset-mining hub entries (private to this collection): the baskets and thresholds as controls, the miner played,
// and the association rules of what it found.
import { apriori, associationRules, fpGrowth, marketBaskets, TEXTBOOK_BASKETS } from 'whitebox-engine/methods/patterns'
import { choice, dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry, type MethodRef } from '../../entry/entry.ts'

/** What an itemset-mining entry declares. */
export interface MiningEntrySpec {
  readonly name: string
  readonly title: string
  readonly summary: string
  readonly miner: 'apriori' | 'fp-growth'
  readonly boardTitle: string
  readonly method: MethodRef
  readonly references: readonly string[]
}

/**
 * The hub entry of an itemset miner with its association rules.
 *
 * @param spec - The miner and its texts.
 * @returns The entry.
 */
export function miningEntry(spec: MiningEntrySpec): HubEntry {
  return defineEntry({
    id: `patterns/${spec.name}`,
    title: spec.title,
    summary: spec.summary,
    collection: 'patterns',
    tags: ['frequent itemsets', 'association rules', 'market baskets'],
    datasets: [],
    params: dict({
      baskets: choice(['textbook (Han and Kamber)', 'market baskets'], { default: 'market baskets', label: 'baskets' }),
      transactions: int(10, 200, { default: 40, label: 'market baskets' }),
      minSupport: real(0.05, 0.6, { default: 0.2, label: 'minimum support' }),
      minConfidence: real(0.1, 1, { default: 0.6, label: 'minimum confidence' }),
    }),
    run: ({ params, key }) => {
      const textbook = params['baskets'] !== 'market baskets'
      const t = textbook ? TEXTBOOK_BASKETS : marketBaskets(key, Number(params['transactions']))
      const minSupport = textbook ? Math.max(Number(params['minSupport']), 2 / 9) : Number(params['minSupport'])
      const mined = spec.miner === 'apriori' ? apriori(t, { minSupport }) : fpGrowth(t, { minSupport })
      const rules = associationRules(mined.itemsets, { minConfidence: Number(params['minConfidence']) })
      return { ...mined, rules: rules.rules, exposition: [...mined.exposition, ...rules.exposition] }
    },
    views: [
      { record: 'boardSteps', title: spec.boardTitle },
      { record: 'coefficientTable', title: 'Rules by lift (support s, confidence c)', area: 'below' },
    ],
    methods: [spec.method, { file: 'methods/patterns/association-rules.ts', name: 'associationRules' }],
    references: spec.references,
  })
}
