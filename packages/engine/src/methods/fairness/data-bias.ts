/**
 * Data-bias audits: what a dataset holds about a protected attribute before (or around) a model, as a `dataBias`
 * record: representation, base rates and their gaps, label bias against clean labels, proxies, and a model's bias
 * amplification.
 */

import { type DataBias, dataBias, type Explained } from '../../lib/exposition/index.ts'
import { baseRates, biasAmplification, labelBias, proxyScores, representation } from '../../lib/fairness/index.ts'
import { type Fitted, featureNamesOf, matrixView } from '../../lib/learn/index.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveSpec,
  sensitiveGroups,
  withoutColumns,
} from './groups.ts'

/** Options of {@link dataBiasAudit}. */
export interface DataBiasAuditOptions {
  /** The protected attribute (default the dataset's first sensitive attribute). */
  readonly sensitive?: SensitiveSpec
  /** The reference group by name (default the largest). */
  readonly reference?: string
  /** The population's share of each group, by name (for sampling bias). */
  readonly populationShares?: Readonly<Record<string, number>>
  /** Clean labels per row (1 positive), when known: label bias is measured against them. */
  readonly cleanLabels?: readonly number[]
  /** A fitted classifier: its bias amplification is measured (scored without the protected column when it is one). */
  readonly model?: Fitted
  /** The model's decision threshold (default 0.5 for probabilities, 0 for decision values). */
  readonly threshold?: number
  /** Whether the model was fitted without the protected column (default true when it is a column). */
  readonly modelWithoutSensitive?: boolean
  readonly label?: string
}

/** The result of {@link dataBiasAudit}. */
export interface DataBiasAudit {
  readonly groups: readonly string[]
  readonly baseRates: readonly number[]
  /** The base rate furthest from the reference's, minus the reference's. */
  readonly statisticalParityDifference: number
  /** The feature that best predicts the group, and its AUC. */
  readonly strongestProxy?: { readonly feature: string; readonly auc: number }
}

/**
 * An audit of the bias in a dataset (AIF360's dataset metrics, Aequitas's group counts): each group's representation
 * (and its ratio to the population when known), base rate and the label disparities (statistical parity difference,
 * disparate impact), label flips against clean labels when known, which features act as proxies for the group (each
 * one's AUC and mutual information, and a linear model of all of them), and, given a model, how much it amplifies the
 * base-rate gap. Exposition: a `dataBias` record.
 *
 * @param data - Labelled data holding the attribute.
 * @param options - The attribute, reference, population shares, clean labels and a model.
 * @returns The base rates, the gap and the strongest proxy, with the record.
 */
export function dataBiasAudit(data: FairnessData, options: DataBiasAuditOptions = {}): Explained<DataBiasAudit> {
  const groups = sensitiveGroups(data, options.sensitive)
  const named = groups.index.map((g) => groups.names[g] ?? String(g))
  const labels = binaryLabels(data, 'dataBiasAudit')
  const rep = representation(
    named,
    options.populationShares === undefined ? {} : { populationShares: options.populationShares },
  )
  const rates = baseRates(labels, named, options.reference === undefined ? {} : { reference: options.reference })
  // Proxies: every other feature column.
  const m = matrixView(data.x, 'dataBiasAudit')
  const names = featureNamesOf(data, m.d)
  const keep = Array.from({ length: m.d }, (_, j) => j).filter((j) => j !== groups.column)
  const rows = Array.from({ length: m.n }, (_, i) => keep.map((j) => m.data[i * m.d + j] ?? 0))
  const proxies =
    keep.length === 0 || rep.groups.length < 2
      ? undefined
      : proxyScores(rows, named, { featureNames: keep.map((j) => names[j] ?? `x${j}`) })
  const flips = options.cleanLabels === undefined ? undefined : labelBias(labels, options.cleanLabels, named)
  let amplification: DataBias['amplification']
  if (options.model !== undefined) {
    const drop = (options.modelWithoutSensitive ?? groups.column !== undefined) && groups.column !== undefined
    const x = drop ? withoutColumns(data, [groups.column ?? -1]).x : data.x
    const { scores, probability } = positiveScores(options.model, x)
    const t = options.threshold ?? (probability ? 0.5 : 0)
    const a = biasAmplification(
      labels,
      scores.map((s) => (s >= t ? 1 : 0)),
      named,
    )
    amplification = { selectionRates: a.selectionRates, value: a.value }
  }
  const ref = rates.reference
  const record = dataBias({
    label: options.label ?? 'data bias',
    attribute: groups.attribute,
    groups: rep.groups,
    reference: ref,
    counts: rep.counts,
    shares: rep.shares,
    ...(rep.populationShares === undefined ? {} : { populationShares: rep.populationShares }),
    baseRates: rates.rates,
    overallBaseRate: rates.overall,
    disparities: [
      {
        name: 'statistical parity difference (labels)',
        value: rates.statisticalParityDifference,
        ideal: 0,
        kind: 'difference',
      },
      { name: 'disparate impact (labels)', value: rates.disparateImpact, ideal: 1, kind: 'ratio' },
    ],
    ...(flips === undefined
      ? {}
      : { labelFlips: { positiveToNegative: flips.positiveToNegative, negativeToPositive: flips.negativeToPositive } }),
    ...(proxies === undefined
      ? {}
      : {
          proxies: {
            features: proxies.features,
            auc: proxies.auc,
            dependence: proxies.dependence,
            ...(proxies.combinedAuc === undefined ? {} : { combinedAuc: proxies.combinedAuc }),
          },
        }),
    ...(amplification === undefined ? {} : { amplification }),
  })
  const best = proxies === undefined ? -1 : proxies.auc.indexOf(Math.max(...proxies.auc))
  return {
    groups: rep.groups,
    baseRates: rates.rates,
    statisticalParityDifference: rates.statisticalParityDifference,
    ...(proxies === undefined || best < 0
      ? {}
      : { strongestProxy: { feature: proxies.features[best] ?? '', auc: proxies.auc[best] ?? 0.5 } }),
    exposition: [record],
  }
}
