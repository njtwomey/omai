import { digits } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { randomForestClassifier } from 'mdml-engine/methods/ensemble'
import { explainImage, type ImageMethod } from 'mdml-engine/methods/explain'
import { key } from 'mdml-engine/random'
import { AttributionImageView, Choice, ControlPanel, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Attribution maps over an image',
  question: 'How do I show which pixels of an image a classifier’s prediction rests on?',
  explain:
    '`explainImage` perturbs one image and watches the predicted class’s probability: occlusion slides a 2 by 2 window and credits the pixels it covers with the drop, ablation and SHAP switch 2 by 2 superpixels off (SHAP shares out their Shapley values), RISE averages random smooth masks weighted by the score, and LIME fits a linear model over the superpixels. The record is laid out as an image, so `AttributionImageView` draws the input beside its map: red pixels raise the probability, blue ones lower it.',
}

// region
const data = digits()
const forest = randomForestClassifier({ nEstimators: 40, maxDepth: 8 }).fit(data, { key: key(1) })
const methods: ImageMethod[] = ['occlusion', 'ablation', 'shap', 'rise', 'lime']

export default function AttributionMapRecipe(): ReactNode {
  const [row, setRow] = useState(5)
  const [method, setMethod] = useState<ImageMethod>('occlusion')
  const record = useMemo(
    () =>
      recordsOf(
        explainImage(forest, data, row, { method, nMasks: 400, key: key(2) }).exposition,
        'featureAttributions',
      )[0],
    [row, method],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="digit (row)" value={row} onChange={setRow} min={0} max={1796} integer />
        <Choice label="explainer" value={method} onChange={setMethod} options={methods} />
      </ControlPanel>
      {record !== undefined && <AttributionImageView record={record} />}
    </div>
  )
}
// endregion
