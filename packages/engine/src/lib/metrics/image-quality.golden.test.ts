import { goldenSuite } from '../../testing/suite.ts'
import type { TensorLike } from '../tensor/index.ts'
import * as M from './index.ts'

const features = (p: Readonly<Record<string, unknown>>, key: string): TensorLike[] => p[key] as TensorLike[]

goldenSuite('image quality against scikit-image, sewar and torchmetrics', 'metrics/image_quality', {
  imageMeanSquaredError: ([a, b]) => M.imageMeanSquaredError(a, b).estimate.value,
  peakSignalNoiseRatio: ([a, b], o) => M.peakSignalNoiseRatio(a, b, o).estimate.value,
  structuralSimilarity: ([a, b], o) => M.structuralSimilarity(a, b, o).estimate.value,
  multiScaleSsim: ([a, b], o) => M.multiScaleSsim(a, b, o).estimate.value,
  visualInformationFidelity: ([a, b]) => M.visualInformationFidelity(a, b).estimate.value,
  universalImageQualityIndex: ([a, b]) => M.universalImageQualityIndex(a, b).estimate.value,
  spectralAngleMapper: ([a, b]) => M.spectralAngleMapper(a, b).estimate.value,
  ergas: ([a, b]) => M.ergas(a, b).estimate.value,
  relativeAverageSpectralError: ([a, b], o) => M.relativeAverageSpectralError(a, b, o).estimate.value,
  lpipsFromFeatures: (_i, p) =>
    M.lpipsFromFeatures(features(p, 'fx'), features(p, 'fy'), features(p, 'w')).estimate.value,
  distsFromFeatures: (_i, p) =>
    M.distsFromFeatures(features(p, 'fx'), features(p, 'fy'), features(p, 'alpha'), features(p, 'beta')).estimate.value,
})
