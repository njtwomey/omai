/**
 * Example "The metrics of a classifier" (a README example): a support vector classifier is fitted as a session (its tape
 * is kept) and scored on 300 held-out points of the two-moons data. Everything shown is the library's own: the ROC
 * curve, the precision-recall curve and the precision-recall gain curve from `omai-engine/metrics` drawn by the render
 * package's charts, and the contingency table (the confusion matrix with its margins and rates) drawn by the render
 * package's `ContingencyTable` component at the decision threshold 0.
 */
import { generate } from 'omai-engine/data'
import { moons } from 'omai-engine/datasets'
import {
  averagePrecisionScore,
  precisionRecallCurve,
  prGainCurve,
  rocAucScore,
  rocCurve,
  thresholdConfusion,
} from 'omai-engine/metrics'
import { key } from 'omai-engine/random'
import { svc } from 'omai-engine/svm'
import { toFlat } from 'omai-engine/tensor'
import { precisionRecallChart, prGainChart, rocChart } from 'omai-render/charts'
import { ContingencyTable, ThemeProvider } from 'omai-render/react'
import { categoricalColour } from 'omai-render/theme'
import { INK } from '../_shared/colours.ts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { TABLE_STYLE } from '../_shared/table-style.ts'

export const meta: ExampleModule['meta'] = {
  title: 'The metrics of a classifier',
  summary:
    'The ROC curve, the precision-recall curve, the precision-recall gain curve and the contingency table of a support vector classifier on held-out data.',
  readme: false,
  alt: 'The ROC, precision-recall and precision-recall gain curves of a classifier, and its contingency table with totals and rates',
}

interface Result {
  readonly roc: ReturnType<typeof rocCurve>
  readonly pr: ReturnType<typeof precisionRecallCurve>
  readonly gain: ReturnType<typeof prGainCurve>
  readonly table: ReturnType<typeof thresholdConfusion>
  readonly auc: number
  readonly ap: number
  readonly prevalence: number
}

export function run(): Outcome<Result> {
  const train = generate(moons, { knobs: { n: 120, noise: 0.25 }, key: key(0) }).data
  const test = generate(moons, { knobs: { n: 300, noise: 0.25 }, key: key(9) }).data
  const fit = svc.with({ kernel: 'rbf', C: 1 }).session(train, { key: key(1) })
  fit.advance()
  const scores = fit.fittedAt(fit.t).decisionFunction(test.x)
  const labels = Array.from(toFlat(test.y))
  const result: Result = {
    roc: rocCurve(test.y, scores),
    pr: precisionRecallCurve(test.y, scores),
    gain: prGainCurve(test.y, scores),
    table: thresholdConfusion(test.y, scores, 0),
    auc: rocAucScore(test.y, scores).estimate.value,
    ap: averagePrecisionScore(test.y, scores).estimate.value,
    prevalence: labels.filter((l) => l === 1).length / labels.length,
  }
  const matrix = Array.from(toFlat(result.table.matrix))
  return {
    result,
    tapes: { session: fit },
    notes: [
      `AUC ${result.auc.toFixed(3)}, average precision ${result.ap.toFixed(3)}, prevalence ${result.prevalence.toFixed(3)}`,
      `contingency table at threshold 0: ${matrix.join(' ')} (rows actual, columns predicted)`,
    ],
  }
}

export function View({ result }: { result: Result }) {
  const roc = rocChart([{ name: `AUC ${result.auc.toFixed(3)}`, curve: result.roc }])
  const pr = precisionRecallChart([{ name: `AP ${result.ap.toFixed(3)}`, curve: result.pr }], {
    prevalence: result.prevalence,
  })
  const gain = prGainChart([{ name: 'precision-recall gain', curve: result.gain }])
  return (
    <Figure width={1200} height={700} title={meta.alt}>
      <Panel x={0} width={400} height={330} title="ROC curve" still={roc} />
      <Panel x={400} width={400} height={330} title="precision-recall curve" still={pr} />
      <Panel x={800} width={400} height={330} title="precision-recall gain curve" still={gain} />
      <text x={20} y={392} fontFamily="sans-serif" fontSize={13} fontWeight={600} fill={INK}>
        contingency table at decision threshold 0
      </text>
      <foreignObject x={20} y={402} width={900} height={290}>
        <div className="wb" {...{ xmlns: 'http://www.w3.org/1999/xhtml' }}>
          <style>{TABLE_STYLE}</style>
          <ThemeProvider mode="light">
            <ContingencyTable
              table={result.table}
              classNames={['class 0', 'class 1']}
              positive={1}
              classColours={[categoricalColour(0), categoricalColour(1)]}
            />
          </ThemeProvider>
        </div>
      </foreignObject>
    </Figure>
  )
}
