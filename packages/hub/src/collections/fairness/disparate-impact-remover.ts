/**
 * Hub entry: preprocessing: repair features so their distribution is the same in every group.
 */

import type { Exposition } from 'whitebox-engine/exposition'
import { fairnessAudit, disparateImpactRemover as repair } from 'whitebox-engine/methods/fairness'
import { dict, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { fairnessDataOf, fitClassifier } from './_shared.ts'

/** Preprocessing: repair features so their distribution is the same in every group. */
export const disparateImpactRemover: HubEntry = defineEntry({
  id: 'fairness/disparate-impact-remover',
  title: 'Disparate-impact remover',
  summary: 'Move each feature towards the groups’ median distribution, keeping ranks within groups; refit unaware.',
  collection: 'fairness',
  tags: ['fairness', 'preprocessing', 'quantiles', 'repair'],
  datasets: ['fairnessTabular'],
  params: dict({ repairLevel: real(0, 1, { default: 1, label: 'repair level' }) }),
  run: (input) => {
    const data = fairnessDataOf(input)
    const r = repair(data, { repairLevel: input.params['repairLevel'] as number })
    // Both models are fitted without the protected column, so only the features carry the group.
    const audit = (d: typeof data, label: string): readonly Exposition[] =>
      fairnessAudit(fitClassifier(d, false, input), d, { dropSensitive: true, label }).exposition
    return { ...r, exposition: [...r.exposition, ...audit(r.data, 'repaired'), ...audit(data, 'original')] }
  },
  views: [
    { record: 'groupReport', index: 0, title: 'Repaired features, model without the attribute' },
    { record: 'curves', title: 'The most group-dependent feature, before and after', area: 'side' },
    { record: 'groupReport', index: 1, title: 'Original features, model without the attribute', area: 'below' },
  ],
  methods: [{ file: 'methods/fairness/preprocessing.ts', name: 'disparateImpactRemover' }],
  references: ['Feldman et al. (2015), Certifying and removing disparate impact'],
})
