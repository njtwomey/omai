import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { biasedCredit } from '../../lib/datasets/index.ts'
import { type Exposition, recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { withoutColumns } from './groups.ts'
import {
  adversarialDebiasingFit,
  fairConstrainedFit,
  gerryFairFit,
  metaFairFit,
  prejudiceRemoverFit,
} from './in-processing.ts'
import { calibratedEqOdds, rejectOption } from './post-processing.ts'
import { fairRepresentation, massaging, optimisedPreprocessing, preferentialSampling } from './preprocessing.ts'

const { data } = generate(biasedCredit, { key: key(6), knobs: { n: 300, labelBias: 0.4 } })
const model = logisticRegression().fit(withoutColumns(data, ['group']))

/** Two group reports, before and after, each with both groups. */
function beforeAfter(records: readonly Exposition[]): void {
  const reports = recordsOf(records, 'groupReport')
  expect(reports).toHaveLength(2)
  for (const r of reports) expect(r.groups).toEqual(['majority', 'minority'])
}

describe('mitigation methods (smoke)', () => {
  it('pre-processing: massaging, preferential sampling, optimised labels, fair representations', () => {
    beforeAfter(massaging(data).exposition)
    const p = preferentialSampling(data)
    beforeAfter(p.exposition)
    expect(recordsOf(p.exposition, 'coefficientTable')).toHaveLength(1)
    beforeAfter(optimisedPreprocessing(data, { epsilon: 0.05 }).exposition)
    beforeAfter(fairRepresentation(data, { prototypes: 3, maxIterations: 60 }).exposition)
  })

  it('in-processing: prejudice remover, adversarial debiasing, Zafar, meta fair, GerryFair', () => {
    beforeAfter(prejudiceRemoverFit(data, { eta: 5 }).exposition)
    const a = adversarialDebiasingFit(data, { steps: 40 })
    beforeAfter(a.exposition)
    expect(recordsOf(a.exposition, 'curves')).toHaveLength(1)
    beforeAfter(fairConstrainedFit(data).exposition)
    beforeAfter(metaFairFit(data, { tau: 0.9 }).exposition)
    const g = gerryFairFit(data, { iterations: 4 })
    beforeAfter(g.exposition)
    expect(g.fit.steps).toHaveLength(3)
  })

  it('post-processing: calibrated equalised odds and reject option', () => {
    const c = calibratedEqOdds(model, data, { dropSensitive: true })
    beforeAfter(c.exposition)
    expect(Math.min(...c.fit.mixRates)).toBe(0)
    const r = rejectOption(model, data, { dropSensitive: true, thresholds: [0.1, 0.9, 9], margins: 5 })
    beforeAfter(r.exposition)
    expect(recordsOf(r.exposition, 'curves')[0]?.series[0]?.y).toHaveLength(9)
  })
})
