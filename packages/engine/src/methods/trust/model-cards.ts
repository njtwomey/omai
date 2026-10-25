import { calibrationRecord, type Explained, type Exposition, recordsOf } from '../../lib/exposition/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import { accuracyScore, rocAucScore } from '../../lib/metrics/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import {
  binaryLabels,
  type FairnessData,
  fairnessAudit,
  positiveScores,
  type SensitiveSpec,
} from '../fairness/index.ts'
import { reliabilityCurve } from './calibration.ts'
import { robustnessStudy } from './robustness.ts'

/** One section of a model card: a heading and lines of text, each optionally with a number. */
export interface ModelCardSection {
  readonly heading: string
  readonly items: readonly { readonly label: string; readonly value?: number | string }[]
}

/** A model card (Mitchell et al., 2019) as data. */
export interface ModelCard {
  readonly title: string
  readonly sections: readonly ModelCardSection[]
}

/** Options of {@link modelCard}. */
export interface ModelCardOptions {
  readonly title?: string
  /** What the model is for and what it is not for. */
  readonly intendedUse?: string
  /** The protected attribute to report by (default the data's first, if any). */
  readonly sensitive?: SensitiveSpec
  readonly key?: Key
}

/**
 * A model card for a fitted binary classifier (Mitchell et al., 2019): its details, evaluation data, overall
 * performance, performance by group with the fairness disparities, calibration, robustness to noise and missing values,
 * and the caveats the numbers imply (gaps over 0.1, calibration error over 0.05, accuracy falling by over 0.1 under
 * one standard deviation of noise). The card is plain data; its evidence is the exposition: the group report, the
 * reliability diagram and the robustness curves.
 *
 * @param model - A fitted binary classifier with probabilities.
 * @param data - Labelled evaluation rows, with a protected attribute for the group sections.
 * @param options - Title, intended use, attribute and key.
 * @returns The card, with the records behind it.
 */
export function modelCard(
  model: Fitted,
  data: FairnessData,
  options: ModelCardOptions = {},
): Explained<{ readonly card: ModelCard }> {
  const labels = binaryLabels(data, 'modelCard')
  const { scores } = positiveScores(model, data.x)
  const predicted = scores.map((s) => (s >= 0.5 ? 1 : 0))
  const accuracy = accuracyScore(labels, predicted).estimate.value
  const auc = rocAucScore(labels, scores).estimate.value
  const hasGroups = options.sensitive !== undefined || data.meta?.sensitive !== undefined
  const audit = hasGroups
    ? fairnessAudit(model, data, options.sensitive === undefined ? {} : { sensitive: options.sensitive })
    : undefined
  const reliability = reliabilityCurve('model', labels, scores, { nBins: 10 })
  const robust = robustnessStudy(model, data, {
    noise: [0, 0.5, 1],
    missing: [0, 0.2],
    repeats: 2,
    key: options.key ?? makeKey(0),
  })
  const caveats: { label: string }[] = []
  if (audit !== undefined) {
    if (audit.disparities.demographicParityDifference > 0.1)
      caveats.push({
        label: `Selection rates differ by ${audit.disparities.demographicParityDifference.toFixed(2)} across ${audit.attribute}.`,
      })
    if (audit.disparities.equalizedOddsDifference > 0.1)
      caveats.push({
        label: `Error rates differ by ${audit.disparities.equalizedOddsDifference.toFixed(2)} across ${audit.attribute}.`,
      })
  }
  if (reliability.ece > 0.05)
    caveats.push({
      label: `Probabilities are miscalibrated (ECE ${reliability.ece.toFixed(3)}); recalibrate before using them as risks.`,
    })
  const drop = (robust.noise[0] ?? 0) - (robust.noise[2] ?? 0)
  if (drop > 0.1) caveats.push({ label: `Accuracy falls by ${drop.toFixed(2)} under one standard deviation of noise.` })
  if (caveats.length === 0) caveats.push({ label: 'No check crossed its threshold.' })
  const sections: ModelCardSection[] = [
    {
      heading: 'Model details',
      items: [
        { label: 'estimator', value: model.estimator },
        ...(model.nFeaturesIn === undefined ? [] : [{ label: 'features', value: model.nFeaturesIn }]),
      ],
    },
    ...(options.intendedUse === undefined
      ? []
      : [{ heading: 'Intended use', items: [{ label: options.intendedUse }] }]),
    {
      heading: 'Evaluation data',
      items: [
        { label: 'rows', value: labels.length },
        { label: 'positive share', value: labels.reduce((a, b) => a + b, 0) / Math.max(1, labels.length) },
      ],
    },
    {
      heading: 'Performance',
      items: [
        { label: 'accuracy at 0.5', value: accuracy },
        { label: 'ROC AUC', value: auc },
      ],
    },
    ...(audit === undefined
      ? []
      : [
          {
            heading: `Performance by ${audit.attribute}`,
            items: [
              ...audit.groups.map((g) => ({ label: `${g}: accuracy`, value: audit.rates[g]?.accuracy ?? Number.NaN })),
              ...audit.groups.map((g) => ({
                label: `${g}: selection rate`,
                value: audit.rates[g]?.selectionRate ?? Number.NaN,
              })),
              { label: 'demographic parity difference', value: audit.disparities.demographicParityDifference },
              { label: 'equalised odds difference', value: audit.disparities.equalizedOddsDifference },
            ],
          },
        ]),
    {
      heading: 'Calibration',
      items: [
        { label: 'expected calibration error', value: reliability.ece },
        { label: 'maximum calibration error', value: reliability.mce },
      ],
    },
    {
      heading: 'Robustness',
      items: [
        { label: 'accuracy with 1 sd of noise', value: robust.noise[2] ?? Number.NaN },
        { label: 'accuracy with 20% missing', value: robust.missing[1] ?? Number.NaN },
      ],
    },
    { heading: 'Caveats', items: caveats },
  ]
  const exposition: Exposition[] = [
    ...(audit === undefined ? [] : recordsOf(audit.exposition, 'groupReport')),
    calibrationRecord({ nBins: 10, strategy: 'uniform', curves: [reliability], split: 'evaluation' }),
    ...robust.exposition,
  ]
  return { card: { title: options.title ?? `Model card: ${model.estimator}`, sections }, exposition }
}
