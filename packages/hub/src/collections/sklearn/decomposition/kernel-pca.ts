import { kernelPca as kernelPcaMethod } from 'whitebox-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Kernel PCA: pCA in a kernel’s feature space: non-linear components. */
export const kernelPca: HubEntry = defineEntry({
  id: 'sklearn/decomposition/kernel-pca',
  title: 'Kernel PCA',
  summary: 'PCA in a kernel’s feature space: non-linear components.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'kernels', 'non-linear'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(kernelPcaMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/kernel-pca.ts', name: 'kernelPca' }],
})
