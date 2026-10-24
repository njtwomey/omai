import { metricValue } from '../../testing/metrics.ts'
import { goldenSuite } from '../../testing/suite.ts'
import * as M from './index.ts'

const all = M as unknown as Record<string, Parameters<typeof metricValue>[0]>
const runners = (names: readonly string[]): Record<string, ReturnType<typeof metricValue>> =>
  Object.fromEntries(names.map((n) => [n, metricValue(all[n] as Parameters<typeof metricValue>[0])]))

goldenSuite(
  'regression metrics against sklearn',
  'metrics/regression',
  runners([
    'meanSquaredError',
    'rootMeanSquaredError',
    'meanAbsoluteError',
    'medianAbsoluteError',
    'r2Score',
    'explainedVarianceScore',
    'meanAbsolutePercentageError',
    'meanSquaredLogError',
    'maxError',
    'adjustedR2Score',
    'meanLogCoshError',
    'meanHuberLoss',
  ]),
)

goldenSuite(
  'deviance and D^2 metrics against sklearn',
  'metrics/deviance',
  runners([
    'meanPinballLoss',
    'd2AbsoluteErrorScore',
    'd2PinballScore',
    'meanPoissonDeviance',
    'meanGammaDeviance',
    'meanTweedieDeviance',
    'd2TweedieScore',
  ]),
)
