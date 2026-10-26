import { generate } from 'mdml-engine/data'
import { blobs } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import type { ClassificationScore } from 'mdml-engine/learn'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { conformalClassification } from 'mdml-engine/methods/trust'
import { key } from 'mdml-engine/random'
import { Choice, ConformalView, ControlPanel, NumberField, Toggle } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Conformal prediction sets',
  question: 'How can a classifier say “one of these classes” with a guaranteed hit rate?',
  explain:
    '`conformalClassification(estimator, data, { alpha, score })` (MAPIE’s classifier) keeps every class whose conformity score is within the calibration rows’ quantile: with LAC, the classes whose probability is at least one minus the quantile; with APS, the most probable classes until their mass passes it. The set holds the true class for at least 1 − α of new rows. Where blobs overlap the sets grow; a smaller α buys coverage with larger sets, and sets of size 0 appear when α is large. Calibrating per class (Mondrian) covers every class, not just on average.',
}

export default function ConformalSetsRecipe(): ReactNode {
  const [alpha, setAlpha] = useState(0.1)
  const [score, setScore] = useState<ClassificationScore>('lac')
  const [spread, setSpread] = useState(2)
  const [byClass, setByClass] = useState(false)
  const data = useMemo(
    () => generate(blobs, { key: key(3), knobs: { n: 900, centers: 4, clusterStd: spread } }).data,
    [spread],
  )
  const record = useMemo(
    () =>
      recordsOf(
        conformalClassification(logisticRegression(), data, { alpha, score, classConditional: byClass, key: key(1) })
          .exposition,
        'conformal',
      )[0],
    [data, alpha, score, byClass],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="α (miscoverage)" value={alpha} onChange={setAlpha} min={0.01} max={0.5} step={0.01} />
        <Choice label="score" value={score} onChange={setScore} options={['lac', 'aps'] as ClassificationScore[]} />
        <NumberField label="blob spread" value={spread} onChange={setSpread} min={0.5} max={5} step={0.25} />
        <Toggle label="calibrate per class" value={byClass} onChange={setByClass} />
      </ControlPanel>
      {record !== undefined && <ConformalView record={record} />}
    </div>
  )
}
