// Fairness hub entries (private to this collection): the fairness data of a run, and a classifier fitted with or
// without the protected attribute.

import type { Fitted } from 'whitebox-engine/learn'
import { type FairnessData, sensitiveGroups, withoutColumns } from 'whitebox-engine/methods/fairness'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { bool } from 'whitebox-engine/space'
import type { HubRunInput } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** The control for whether the model sees the protected attribute. */
export const awareParam: ReturnType<typeof bool> = bool({
  default: true,
  label: 'model sees the protected attribute',
  description: 'Fit with the protected column (off: fairness through unawareness, which proxies can defeat).',
})

/** The run's data as fairness data. */
export function fairnessDataOf(input: HubRunInput): FairnessData {
  return learnDataOf(input.data) as FairnessData
}

/** Logistic regression fitted to the data, with or without its protected column. */
export function fitClassifier(data: FairnessData, aware: boolean, input: HubRunInput): Fitted {
  const train = aware ? data : withoutColumns(data, [sensitiveGroups(data).column ?? -1])
  return logisticRegression().fit(train, { key: input.key, explain: false })
}
