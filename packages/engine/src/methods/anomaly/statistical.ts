import { WhiteboxError } from '../../lib/base/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  type FitOptions,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { histogram } from '../../lib/stats/index.ts'
import { type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { covarianceEstimator, minCovDet, squaredMahalanobis } from '../covariance/index.ts'
import { pca } from '../decomposition/index.ts'
import { contaminationThreshold, type OutlierVerbs, outlierExposition, outlierVerbs } from './combine.ts'

const rowsOf = (x: TensorLike, who: string, d?: number): number[][] => {
  const m = matrixView(x, who, d === undefined ? {} : { nFeatures: d })
  return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
}

/** Fit a scorer-with-threshold detector: training scores, the contamination threshold and the exposition. */
function scorerFit<P extends { readonly threshold: number }>(
  name: string,
  data: LearnData,
  contamination: number,
  params: (threshold: number) => P,
  scores: (p: P) => (x: TensorLike) => number[],
  evidence: string,
) {
  const d = matrixView(data.x, name).d
  const draft = params(0)
  const train = scores(draft)(data.x)
  const threshold = contaminationThreshold(train, contamination)
  const p = params(threshold)
  const fitted = { estimator: name, kind: 'outlier' as const, params: p, ...outlierVerbs(scores(p), threshold, d) }
  return {
    params: p,
    exposition: outlierExposition(fitted, data.x, name, threshold, { name: evidence, values: train }, train),
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Elliptic envelope and Mahalanobis distances

/** Hyperparameters of the Mahalanobis detectors. */
export type EllipticEnvelopeHyper = {
  /** The expected fraction of outliers (default 0.1). */
  readonly contamination?: number
  /** A robust (MCD, the elliptic envelope) or empirical location and covariance (default robust). */
  readonly robust?: boolean
  readonly supportFraction?: number
}

/** The parameters of a fitted elliptic envelope. */
export type EllipticEnvelopeParams = {
  readonly location: number[]
  readonly precision: number[][]
  readonly threshold: number
}

/** A fitted Mahalanobis detector. */
export type FittedEllipticEnvelope = ExplainedFitted<EllipticEnvelopeParams, OutlierVerbs>

/**
 * The elliptic envelope (sklearn.covariance.EllipticEnvelope): squared Mahalanobis distances under the minimum
 * covariance determinant (or, non-robust, the empirical covariance), thresholded at the contamination quantile. Its
 * exposition is an `anomalyScores` record with the distances and, for one or two features, a `decisionSurface`.
 *
 * @param hyper - Contamination, robustness and the MCD support fraction.
 * @returns The estimator.
 */
export function ellipticEnvelope(
  hyper: EllipticEnvelopeHyper = {},
): Estimator<EllipticEnvelopeHyper, FittedEllipticEnvelope> {
  const scores = (p: EllipticEnvelopeParams) => (x: TensorLike) => squaredMahalanobis(x, p.location, p.precision)
  return defineExplainedEstimator<EllipticEnvelopeHyper, EllipticEnvelopeParams, OutlierVerbs>(
    {
      name: 'elliptic-envelope',
      kind: 'outlier',
      hyperSpace: (h) =>
        dict({
          contamination: real(0.01, 0.5, { default: h.contamination ?? 0.1 }),
          robust: choice([true, false], { default: h.robust ?? true }),
        }),
      fit: (data, h, options: FitOptions) => {
        const cov =
          h.robust === false
            ? covarianceEstimator().fit(data).params
            : minCovDet(h.supportFraction === undefined ? {} : { supportFraction: h.supportFraction }).fit(
                data,
                options,
              ).params
        return scorerFit(
          h.robust === false ? 'mahalanobis' : 'ellipticEnvelope',
          data,
          h.contamination ?? 0.1,
          (threshold) => ({ location: cov.location, precision: cov.precision, threshold }),
          scores,
          'squaredMahalanobis',
        )
      },
      methods: (p) => outlierVerbs(scores(p), p.threshold, p.location.length),
    },
    hyper,
  )
}

/** The elliptic envelope as a class. */
export class EllipticEnvelope extends EstimatorClass<EllipticEnvelopeHyper, FittedEllipticEnvelope> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link ellipticEnvelope}.
   */
  constructor(hyper: EllipticEnvelopeHyper = {}) {
    super(ellipticEnvelope(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// PCA reconstruction error

/** Hyperparameters of the PCA reconstruction detector. */
export type PcaDetectorHyper = { readonly nComponents?: number; readonly contamination?: number }

/** The parameters of a fitted PCA detector. */
export type PcaDetectorParams = {
  readonly mean: number[]
  readonly components: number[][]
  readonly threshold: number
}

/**
 * The PCA reconstruction-error detector: a row's score is its squared distance to the subspace of the leading
 * principal components (decomposition's PCA). Its exposition is an `anomalyScores` record.
 *
 * @param hyper - Components kept and contamination (default 0.1).
 * @returns The estimator.
 */
export function pcaReconstructionDetector(
  hyper: PcaDetectorHyper = {},
): Estimator<PcaDetectorHyper, ExplainedFitted<PcaDetectorParams, OutlierVerbs>> {
  const scores = (p: PcaDetectorParams) => (x: TensorLike) =>
    rowsOf(x, 'pcaReconstructionDetector', p.mean.length).map((r) => {
      const c = r.map((v, j) => v - (p.mean[j] ?? 0))
      const total = c.reduce((a, v) => a + v * v, 0)
      const kept = p.components.reduce((a, w) => a + w.reduce((s, v, j) => s + v * (c[j] ?? 0), 0) ** 2, 0)
      return Math.max(0, total - kept)
    })
  return defineExplainedEstimator<PcaDetectorHyper, PcaDetectorParams, OutlierVerbs>(
    {
      name: 'pca-reconstruction-detector',
      kind: 'outlier',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 20, { default: h.nComponents ?? 1, label: 'components' }),
          contamination: real(0.01, 0.5, { default: h.contamination ?? 0.1 }),
        }),
      fit: (data, h) => {
        const p = pca({ nComponents: h.nComponents ?? 1 }).fit(data).params
        return scorerFit(
          'pcaReconstruction',
          data,
          h.contamination ?? 0.1,
          (threshold) => ({ mean: p.mean, components: p.components, threshold }),
          scores,
          'reconstructionError',
        )
      },
      methods: (p) => outlierVerbs(scores(p), p.threshold, p.mean.length),
    },
    hyper,
  )
}

// ---------------------------------------------------------------------------------------------------------------
// HBOS and z-scores

/** Hyperparameters of HBOS. */
export type HbosHyper = { readonly bins?: number; readonly contamination?: number }

/** The parameters of a fitted HBOS: per-feature bin edges and densities. */
export type HbosParams = { readonly edges: number[][]; readonly densities: number[][]; readonly threshold: number }

/**
 * Histogram-based outlier scores (Goldstein and Dengel 2012): per-feature histograms (stats' histogram densities,
 * normalised to a maximum of one), a row's score the sum over features of log(1 / height of its bin). Its exposition
 * is an `anomalyScores` record and, for one or two features, a `decisionSurface`.
 *
 * @param hyper - Bins per feature (default 10) and contamination (default 0.1).
 * @returns The estimator.
 */
export function hbos(hyper: HbosHyper = {}): Estimator<HbosHyper, ExplainedFitted<HbosParams, OutlierVerbs>> {
  const scores = (p: HbosParams) => (x: TensorLike) =>
    rowsOf(x, 'hbos', p.edges.length).map((r) =>
      r.reduce((a, v, j) => {
        const e = p.edges[j] ?? []
        const dens = p.densities[j] ?? []
        let b = -1
        if (v >= (e[0] ?? 0) && v <= (e.at(-1) ?? 0)) {
          b = e.findIndex((edge, t) => t > 0 && v <= edge) - 1
          if (b < 0) b = 0
        }
        const height = b < 0 ? 0 : (dens[b] ?? 0)
        return a + Math.log(1 / Math.max(height, 1e-12))
      }, 0),
    )
  return defineExplainedEstimator<HbosHyper, HbosParams, OutlierVerbs>(
    {
      name: 'hbos',
      kind: 'outlier',
      hyperSpace: (h) =>
        dict({
          bins: int(2, 100, { default: h.bins ?? 10, label: 'bins' }),
          contamination: real(0.01, 0.5, { default: h.contamination ?? 0.1 }),
        }),
      fit: (data, h) => {
        const rows = rowsOf(data.x, 'hbos')
        const d = rows[0]?.length ?? 0
        const edges: number[][] = []
        const densities: number[][] = []
        for (let j = 0; j < d; j++) {
          const hist = histogram(
            rows.map((r) => r[j] ?? 0),
            { bins: h.bins ?? 10, density: true },
          )
          const dens = toFlat(hist.counts)
          const top = Math.max(...dens, 1e-300)
          edges.push(toFlat(hist.edges))
          densities.push(dens.map((v) => v / top))
        }
        return scorerFit(
          'hbos',
          data,
          h.contamination ?? 0.1,
          (threshold) => ({ edges, densities, threshold }),
          scores,
          'logInverseDensity',
        )
      },
      methods: (p) => outlierVerbs(scores(p), p.threshold, p.edges.length),
    },
    hyper,
  )
}

/** Hyperparameters of the z-score detector. */
export type ZScoreHyper = {
  /** A row is flagged when its largest |z| over features exceeds this (default 3). */
  readonly threshold?: number
  /** Median and MAD instead of mean and standard deviation (default false). */
  readonly robust?: boolean
}

/** The parameters of a fitted z-score detector. */
export type ZScoreParams = { readonly centre: number[]; readonly scale: number[]; readonly threshold: number }

/**
 * The z-score detector: a row's score is its largest absolute standardised value over features (mean and standard
 * deviation, or median and 1.4826 MAD). Its exposition is an `anomalyScores` record with the flagging feature as
 * evidence.
 *
 * @param hyper - The threshold and robustness.
 * @returns The estimator.
 */
export function zScoreDetector(
  hyper: ZScoreHyper = {},
): Estimator<ZScoreHyper, ExplainedFitted<ZScoreParams, OutlierVerbs>> {
  const zs = (p: ZScoreParams, x: TensorLike): number[][] =>
    rowsOf(x, 'zScoreDetector', p.centre.length).map((r) =>
      r.map((v, j) => Math.abs(v - (p.centre[j] ?? 0)) / (p.scale[j] || 1)),
    )
  const scores = (p: ZScoreParams) => (x: TensorLike) => zs(p, x).map((r) => Math.max(...r))
  return defineExplainedEstimator<ZScoreHyper, ZScoreParams, OutlierVerbs>(
    {
      name: 'z-score-detector',
      kind: 'outlier',
      hyperSpace: (h) =>
        dict({
          threshold: real(1, 10, { default: h.threshold ?? 3 }),
          robust: choice([true, false], { default: h.robust ?? false }),
        }),
      fit: (data, h) => {
        const rows = rowsOf(data.x, 'zScoreDetector')
        if (rows.length < 2) throw new WhiteboxError('value', 'zScoreDetector: needs at least two rows')
        const d = rows[0]?.length ?? 0
        const cols = Array.from({ length: d }, (_, j) => rows.map((r) => r[j] ?? 0))
        const median = (v: number[]): number => {
          const s = [...v].sort((a, b) => a - b)
          return s.length % 2 === 1
            ? (s[(s.length - 1) / 2] ?? 0)
            : ((s[s.length / 2 - 1] ?? 0) + (s[s.length / 2] ?? 0)) / 2
        }
        const centre = cols.map((c) => (h.robust === true ? median(c) : c.reduce((a, v) => a + v, 0) / c.length))
        const scale = cols.map((c, j) =>
          h.robust === true
            ? 1.4826 * median(c.map((v) => Math.abs(v - (centre[j] ?? 0))))
            : Math.sqrt(c.reduce((a, v) => a + (v - (centre[j] ?? 0)) ** 2, 0) / c.length),
        )
        const params = { centre, scale, threshold: h.threshold ?? 3 }
        const z = zs(params, data.x)
        const train = z.map((r) => Math.max(...r))
        const fitted = {
          estimator: 'z-score-detector',
          kind: 'outlier' as const,
          params,
          ...outlierVerbs(scores(params), params.threshold, d),
        }
        return {
          params,
          exposition: outlierExposition(
            fitted,
            data.x,
            'zScore',
            params.threshold,
            { name: 'flaggingFeature', values: z.map((r) => r.indexOf(Math.max(...r))) },
            train,
          ),
        }
      },
      methods: (p) => outlierVerbs(scores(p), p.threshold, p.centre.length),
    },
    hyper,
  )
}
