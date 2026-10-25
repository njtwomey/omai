import { generate } from 'mdml-engine/data'
import { makeClassification } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { type ShiftKind, shiftStudy } from 'mdml-engine/methods/trust'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, CurvesView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Dataset shift',
  question: 'Which kinds of drift can be detected from the inputs alone?',
  explain:
    '`shiftStudy(estimator, data, { kind })` fits on reference rows, then shifts the current rows by growing amounts. Two detectors look only at the features: per-feature Kolmogorov–Smirnov tests (−log10 p on the right axis) and a domain classifier trying to tell reference from current rows (an AUC of 0.5 means it cannot). Covariate shift moves the features: both detectors fire as the model’s accuracy drops. Concept shift flips labels while the features stay put: accuracy falls and neither detector notices — detecting it needs labels.',
}

export default function ShiftRecipe(): ReactNode {
  const [kind, setKind] = useState<ShiftKind>('covariate')
  const data = useMemo(
    () =>
      generate(makeClassification, { key: key(8), knobs: { n: 1000, nFeatures: 5, nInformative: 3, nRedundant: 0 } })
        .data,
    [],
  )
  const record = useMemo(
    () => recordsOf(shiftStudy(logisticRegression(), data, { kind, key: key(1) }).exposition, 'curves')[0],
    [data, kind],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <Choice label="shift" value={kind} onChange={setKind} options={['covariate', 'concept'] as ShiftKind[]} />
      </ControlPanel>
      {record !== undefined && <CurvesView record={record} />}
    </div>
  )
}
