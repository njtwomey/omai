import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite(
  'contrastive losses against torch and optax, and the reference code of Barlow Twins, VICReg and alignment-uniformity',
  'losses/contrastive',
  {
    infoNce: ([q, k, n], p) => L.infoNce(q, k, { ...p, ...(n === undefined ? {} : { negatives: n }) }),
    ntXent: ([e, y], p) => L.ntXent(e, y, p),
    tripletMargin: ([a, pos, neg], p) => L.tripletMargin(a, pos, neg as Tensor, p),
    contrastive: ([a, b, y], p) => L.contrastive(a, b, y as Tensor, p),
    cosineEmbedding: ([a, b, y], p) => L.cosineEmbedding(a, b, y as Tensor, p),
    hingeEmbedding: ([d, y], p) => L.hingeEmbedding(d, y, p),
    supervisedContrastive: ([f, y], p) => L.supervisedContrastive(f, y, p),
    clipLoss: ([i, t], p) => L.clipLoss(i, t, p),
    arcFaceLogits: ([x, w, y], p) => L.arcFaceLogits(x, w, y as Tensor, p),
    arcFace: ([x, w, y], p) => L.arcFace(x, w, y as Tensor, p),
    cosFace: ([x, w, y], p) => L.cosFace(x, w, y as Tensor, p),
    barlowTwins: ([a, b], p) => {
      const { lambda_, ...rest } = p as { lambda_?: number }
      return L.barlowTwins(a, b, { ...rest, ...(lambda_ === undefined ? {} : { lambda: lambda_ }) })
    },
    vicreg: ([a, b], p) => L.vicreg(a, b, p),
    alignment: ([a, b], p) => L.alignment(a, b, p),
    uniformity: ([a], p) => L.uniformity(a, p),
  },
)
