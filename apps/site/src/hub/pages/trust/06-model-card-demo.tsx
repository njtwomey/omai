import { generate } from 'mdml-engine/data'
import { biasedCredit } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { modelCard } from 'mdml-engine/methods/trust'
import { key } from 'mdml-engine/random'
import { figure } from 'mdml-render/figure'
import { CalibrationView, ControlPanel, GroupReportView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Model card',
  question: 'How do I summarise what a model does well, for whom, and where it fails?',
  explain:
    '`modelCard(model, data, { intendedUse })` returns a model card (Mitchell et al., 2019) as plain data — details, evaluation data, performance overall and by group, calibration, robustness and the caveats its checks raise — with its evidence as exposition records: the group report and the reliability diagram here. Raise the label bias and the card grows a caveat about the gap between groups.',
}

const REPORT = figure({ width: 940, height: 480 })

const fmt = (v: number | string | undefined): string =>
  v === undefined ? '' : typeof v === 'string' ? v : Number.isInteger(v) ? String(v) : v.toFixed(3)

export default function ModelCardRecipe(): ReactNode {
  const [labelBias, setLabelBias] = useState(0.3)
  const result = useMemo(() => {
    const data = generate(biasedCredit, { key: key(11), knobs: { n: 1200, labelBias } }).data
    return modelCard(logisticRegression().fit(data), data, {
      title: 'Credit decisions (logistic regression)',
      intendedUse: 'Illustration only: a model of synthetic loan outcomes, not for decisions about people.',
    })
  }, [labelBias])
  const report = recordsOf(result.exposition, 'groupReport')[0]
  const calibration = recordsOf(result.exposition, 'calibration')[0]
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="label bias" value={labelBias} onChange={setLabelBias} min={0} max={0.9} step={0.05} />
      </ControlPanel>
      <article className="flex max-w-3xl flex-col gap-3 rounded border border-[var(--wb-grid)] p-4 text-sm">
        <h3 className="text-base font-semibold">{result.card.title}</h3>
        {result.card.sections.map((s) => (
          <section key={s.heading}>
            <h4 className="font-semibold">{s.heading}</h4>
            <ul className="ml-4 list-disc">
              {s.items.map((item) => (
                <li key={item.label}>
                  {item.label}
                  {item.value === undefined ? '' : `: ${fmt(item.value)}`}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </article>
      {report !== undefined && <GroupReportView record={report} figure={REPORT} />}
      {calibration !== undefined && <CalibrationView record={calibration} />}
    </div>
  )
}
