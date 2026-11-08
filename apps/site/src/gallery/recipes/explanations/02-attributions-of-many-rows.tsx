import { kernelShap } from 'omai-engine/explain'
import { featureAttributions } from 'omai-engine/exposition'
import { tensor } from 'omai-engine/tensor'
import {
  AttributionBarView,
  Choice,
  ControlPanel,
  DependenceScatterView,
  ExpositionView,
  NumberField,
  SeedInput,
  Select,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { CREDIT_FEATURES, creditPredict, creditRows, creditScore } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Attributions of many rows',
  question: 'How do I show what a model uses overall, and how a feature acts across rows?',
  explain:
    'A `featureAttributions` record with many rows is drawn by `ExpositionView` as a beeswarm: a row per feature, a dot per explained row at its attribution, coloured by the feature’s own value. `AttributionBarView` reduces it to mean |attribution| per feature, and `DependenceScatterView` plots one feature’s values against its attributions, coloured by a second feature, so a vertical spread at one value shows an interaction. In the toy credit model debt matters more when income is low: choose `debt` and colour by `income`.',
  shows: ['featureAttributions', 'BeeswarmView', 'AttributionBarView', 'DependenceScatterView'],
}

type Style = 'beeswarm' | 'bars' | 'dependence'

export default function AttributionsOfManyRows(): ReactNode {
  const [rowsExplained, setRows] = useState(80)
  const [seed, setSeed] = useState(2)
  const [style, setStyle] = useState<Style>('beeswarm')
  const [feature, setFeature] = useState('debt')
  const [colour, setColour] = useState('income')
  const record = useMemo(() => {
    const rows = creditRows(seed, rowsExplained + 40)
    const background = rows.slice(rowsExplained)
    const x = rows.slice(0, rowsExplained)
    const shap = kernelShap(creditPredict, tensor(x), tensor(background))
    return featureAttributions({
      method: 'kernel',
      features: CREDIT_FEATURES,
      output: 'log odds of repaying',
      baseValue: shap.baseValue,
      values: shap.values,
      data: x,
      predictions: x.map(creditScore),
    })
  }, [seed, rowsExplained])
  const index = (name: string): number => CREDIT_FEATURES.indexOf(name)
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <SeedInput value={seed} onChange={setSeed} />
        <NumberField
          label="rows explained"
          value={rowsExplained}
          onChange={setRows}
          min={10}
          max={300}
          step={10}
          integer
        />
        <Choice label="view" value={style} options={['beeswarm', 'bars', 'dependence'] as const} onChange={setStyle} />
        {style === 'dependence' && (
          <>
            <Select label="feature" value={feature} options={CREDIT_FEATURES} onChange={setFeature} />
            <Select label="coloured by" value={colour} options={CREDIT_FEATURES} onChange={setColour} />
          </>
        )}
      </ControlPanel>
      {style === 'beeswarm' && <ExpositionView record={record} title="Every row's attributions, by feature" />}
      {style === 'bars' && <AttributionBarView record={record} title="Mean |attribution|" />}
      {style === 'dependence' && (
        <DependenceScatterView
          record={record}
          feature={index(feature)}
          colourBy={index(colour)}
          title={`${feature} against its attribution`}
        />
      )}
    </div>
  )
}
