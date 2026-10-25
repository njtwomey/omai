/**
 * The prototypes entry: MMD-critic's prototypes and criticisms of a dataset, on its principal components or as images.
 */

import { prototypesAndCriticisms } from 'mdml-engine/methods/explain'
import { dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Prototypes and criticisms. */
export const prototypes: HubEntry = defineEntry({
  id: 'explainability/prototypes',
  title: 'Prototypes and criticisms',
  summary: 'The rows that summarise a dataset and the rows the summary misses, chosen by maximum mean discrepancy.',
  collection: 'explainability',
  tags: ['prototypes', 'criticisms', 'MMD-critic', 'example-based', 'data summaries'],
  datasets: ['classificationTabular'],
  dataset: 'iris',
  maxRows: 600,
  params: dict({
    prototypes: int(1, 20, { default: 6, label: 'prototypes' }),
    criticisms: int(0, 10, { default: 3, label: 'criticisms' }),
  }),
  run: ({ data, params }) => {
    const learn = learnDataOf(data)
    const names = (learn as { readonly featureNames?: readonly string[] }).featureNames
    const images = names?.length === 64 && names[0]?.startsWith('pixel_') === true
    return prototypesAndCriticisms(learn, {
      nPrototypes: Number(params['prototypes']),
      nCriticisms: Number(params['criticisms']),
      maxRows: 300,
      ...(images ? { imageShape: [8, 8] as const } : {}),
    })
  },
  views: [{ record: 'prototypes', view: 'PrototypeView', title: 'Prototypes (diamonds) and criticisms (crosses)' }],
  methods: [{ file: 'methods/explain/counterfactuals.ts', name: 'prototypesAndCriticisms' }],
  references: [
    'Kim, Khanna and Koyejo (2016), Examples are not enough, learn to criticize! Criticism for interpretability.',
  ],
})
