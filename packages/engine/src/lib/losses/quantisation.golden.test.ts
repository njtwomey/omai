import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

const ema = (inputs: readonly (Tensor | undefined)[], p: { decay?: number; eps?: number }): L.VqEmaState => {
  const [counts, sums, z, idx] = inputs as [Tensor, Tensor, Tensor, Tensor]
  return L.vqEmaUpdate({ counts, sums, codebook: sums }, z, idx, p)
}

goldenSuite('noise-contrastive estimation and vector quantisation against reference formulas', 'losses/quantisation', {
  noiseContrastive: ([sd, qd, sn, qn], p) => L.noiseContrastive(sd, qd, sn as Tensor, qn as Tensor, p),
  vectorQuantiseLoss: ([z, book], p) => L.vectorQuantise(z, book, p).loss,
  vectorQuantiseIndices: ([z, book]) => L.vectorQuantise(z, book).indices,
  vqEmaCounts: (inputs, p) => ema(inputs, p).counts,
  vqEmaSums: (inputs, p) => ema(inputs, p).sums,
  vqEmaCodebook: (inputs, p) => ema(inputs, p).codebook,
})
