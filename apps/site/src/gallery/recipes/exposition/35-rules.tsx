import { breastCancer, diabetes } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { randomForestClassifier } from 'mdml-engine/methods/ensemble'
import { anchorExplanation, logisticScorecard, ruleFitExplanation } from 'mdml-engine/methods/explain'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, RuleListView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Rule lists',
  question: 'How do I show a model, or one prediction, as a few readable rules?',
  explain:
    'Three `rules` records: `ruleFitExplanation` grows shallow trees, turns their paths into rules and keeps the few a lasso needs (RuleFit), each bar its importance coloured by its coefficient’s sign with its support beside it; `anchorExplanation` finds the intervals around one tumour that keep the forest’s prediction with 95% precision, drawn as precision and coverage; and `logisticScorecard` turns a logistic model into points per feature bin, as credit scorecards are read. `RuleListView` writes each rule beside its bar.',
}

// region
const reg = diabetes()
const cancer = breastCancer()
const forest = randomForestClassifier({ nEstimators: 30, maxDepth: 5 }).fit(cancer, { key: key(1) })
const logistic = logisticRegression({ C: 1 }).fit(cancer)
type Kind = 'RuleFit' | 'anchor' | 'scorecard'

export default function RulesRecipe(): ReactNode {
  const [kind, setKind] = useState<Kind>('RuleFit')
  const record = useMemo(() => {
    const r =
      kind === 'RuleFit'
        ? ruleFitExplanation(reg, { nTrees: 20, key: key(2) })
        : kind === 'anchor'
          ? anchorExplanation(forest, cancer, 7, { key: key(3) })
          : logisticScorecard(logistic, cancer, { bins: 3 })
    return recordsOf(r.exposition, 'rules')[0]
  }, [kind])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <Choice label="rules" value={kind} onChange={setKind} options={['RuleFit', 'anchor', 'scorecard'] as Kind[]} />
      </ControlPanel>
      {record !== undefined && <RuleListView record={record} maxRules={kind === 'scorecard' ? 15 : 10} />}
    </div>
  )
}
// endregion
