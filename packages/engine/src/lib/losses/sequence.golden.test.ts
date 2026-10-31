import { goldenSuite } from '../../testing/suite.ts'
import * as L from './index.ts'

goldenSuite('CTC against torch.nn.functional.ctc_loss', 'losses/sequence', {
  ctc: ([lp, targets], p) => L.ctc(lp, targets, p as L.CtcOptions),
})
