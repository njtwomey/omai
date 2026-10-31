import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite('preference objectives against the TRL and paper formulas', 'losses/preference', {
  dpo: ([pc, pr, rc, rr], p) =>
    L.dpo({ policyChosen: pc, policyRejected: pr, referenceChosen: rc as Tensor, referenceRejected: rr as Tensor }, p),
  ipo: ([pc, pr, rc, rr], p) =>
    L.ipo({ policyChosen: pc, policyRejected: pr, referenceChosen: rc as Tensor, referenceRejected: rr as Tensor }, p),
  kto: ([pc, pr, rc, rr], p) => {
    // The pairs read as unpaired answers: the chosen are desirable and the rejected undesirable.
    const n = pc.shape[0] ?? 0
    const policy = [...(pc.data as Float64Array), ...(pr.data as Float64Array)]
    const reference = [...((rc as Tensor).data as Float64Array), ...((rr as Tensor).data as Float64Array)]
    const desirable = policy.map((_, i) => (i < n ? 1 : 0))
    return L.kto(policy, reference, desirable, p)
  },
  orpo: ([c, r], p) => {
    const { lambda_, ...rest } = p as { lambda_?: number }
    return L.orpo(c, r, { ...rest, ...(lambda_ === undefined ? {} : { lambda: lambda_ }) })
  },
  simpo: ([c, r], p) => L.simpo(c, r, p),
  bradleyTerry: ([c, r], p) => L.bradleyTerry(c, r, p),
})
