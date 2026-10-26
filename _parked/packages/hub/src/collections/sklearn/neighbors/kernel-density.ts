/**
 * Hub entry for kernel density estimation, a smooth density from a bump on every point with its bandwidth chosen from
 * the data, on two-dimensional densities and clustering datasets.
 */

import { kernelDensity as kernelDensityMethod } from 'mdml-engine/methods/mixture'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Kernel density estimation: a smooth density from a bump on every point, its bandwidth chosen from the data. */
export const kernelDensity: HubEntry = defineEntry({
  id: 'sklearn/neighbors/kernel-density',
  title: 'Kernel density estimation',
  summary: 'A smooth density from a bump on every point, its bandwidth chosen from the data.',
  collection: 'sklearn/neighbors',
  tags: ['density estimation', 'nonparametric'],
  datasets: ['density2d', 'clustering2d'],
  ...fitEstimator(kernelDensityMethod({ bandwidth: 'cv' }), { unsupervised: true }),
  views: [{ record: 'decisionSurface', title: 'Density' }],
  methods: [{ file: 'methods/mixture/density.ts', name: 'kernelDensity' }],
})
