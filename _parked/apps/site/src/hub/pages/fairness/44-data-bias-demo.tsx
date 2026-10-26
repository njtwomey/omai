import { generate } from 'mdml-engine/data'
import { biasedCredit } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { dataBiasAudit, withoutColumns } from 'mdml-engine/methods/fairness'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { key } from 'mdml-engine/random'
import { ControlPanel, DataBiasView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Bias in data',
  question: 'How do I show the bias a dataset carries before any model is fitted?',
  explain:
    '`biasedCredit` generates credit data with knobs for each kind of bias, and keeps the clean labels as truth. `dataBiasAudit(data, { cleanLabels, populationShares, model })` returns a `dataBias` record: each group’s share of the data against the population’s (representation), its base rate (the labels’ statistical parity difference and disparate impact beneath), the share of its clean labels recorded otherwise (label bias), how well each feature tells the groups apart (proxies: AUC and mutual information, and a linear model of all of them), and a model’s selection rate per group with its bias amplification. `DataBiasView` draws the four panels. Raise the proxy strength: the postcode becomes a proxy, and a model fitted without the group still learns it.',
}

export default function DataBiasRecipe(): ReactNode {
  const [labelBias, setLabelBias] = useState(0.3)
  const [proxy, setProxy] = useState(2)
  const [share, setShare] = useState(0.2)
  const record = useMemo(() => {
    const { data, truth } = generate(biasedCredit, {
      key: key(5),
      knobs: { n: 1500, labelBias, proxyStrength: proxy, minorityShare: share },
    })
    const model = logisticRegression().fit(withoutColumns(data, ['group']))
    const audit = dataBiasAudit(data, {
      cleanLabels: truth?.latents?.['cleanLabels'] as number[],
      populationShares: { majority: 0.5, minority: 0.5 },
      model,
    })
    return recordsOf(audit.exposition, 'dataBias')[0]
  }, [labelBias, proxy, share])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="label bias" value={labelBias} onChange={setLabelBias} min={0} max={0.9} step={0.05} />
        <NumberField label="proxy strength" value={proxy} onChange={setProxy} min={0} max={4} step={0.25} />
        <NumberField label="minority share" value={share} onChange={setShare} min={0.05} max={0.5} step={0.05} />
      </ControlPanel>
      {record !== undefined && <DataBiasView record={record} />}
    </div>
  )
}
