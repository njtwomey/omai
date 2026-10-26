/**
 * The TCAV entry: whether a LeNet's decision for a digit rests on concepts given as other digits' images.
 */

import { tcavStudy } from 'mdml-engine/methods/explain'
import { lenet } from 'mdml-engine/methods/neural/vision'
import { dict, int } from 'mdml-engine/space'
import { asTensor, type TensorLike, toFlat } from 'mdml-engine/tensor'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Testing with concept activation vectors. */
export const tcav: HubEntry = defineEntry({
  id: 'explainability/tcav',
  title: 'Concept activation vectors (TCAV)',
  summary:
    'Does a network’s “9” rest on 4-like and 7-like strokes? Concept directions at a hidden layer, tested against random.',
  collection: 'explainability',
  tags: ['TCAV', 'concepts', 'concept activation vectors', 'neural networks', 'probing'],
  datasets: ['classificationTabular'],
  dataset: 'digits',
  maxRows: 900,
  params: dict({
    target: int(0, 9, { default: 9, label: 'class explained' }),
    randoms: int(3, 20, { default: 8, label: 'random sets' }),
    epochs: int(1, 20, { default: 4, label: 'training epochs' }),
  }),
  run: ({ data, params, key }) => {
    const learn = learnDataOf(data)
    const net = lenet({ epochs: Number(params['epochs']), snapshots: 2 }).fit(learn, { key, explain: false })
    const labels = toFlat(asTensor(learn.y as TensorLike))
    const target = Number(params['target'])
    const rowsOf = (d: number): number[] => labels.flatMap((v, i) => (v === d ? [i] : [])).slice(0, 40)
    const concepts = [0, 1, 4, 7].filter((c) => c !== target).map((c) => ({ name: `${c}-like`, rows: rowsOf(c) }))
    return tcavStudy(net, learn, concepts, { classIndex: target, nRandom: Number(params['randoms']), key })
  },
  views: [{ record: 'concepts', view: 'ConceptView', title: 'TCAV scores against random directions' }],
  methods: [{ file: 'methods/explain/concepts.ts', name: 'tcavStudy' }],
  references: [
    'Kim et al. (2018), Interpretability beyond feature attribution: quantitative testing with concept activation vectors (TCAV).',
  ],
})
