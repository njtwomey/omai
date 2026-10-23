import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import { linear } from './linear.ts'
import {
  batchNorm,
  groupNorm,
  instanceNorm,
  layerNorm,
  localResponseNorm,
  normalize,
  rmsNorm,
  spectralNorm,
  weightNorm,
} from './normalisation.ts'

describe('normalisation functions against torch', () => {
  for (const c of loadFixture('nn/functional/normalisation').cases) {
    it(c.name, () => {
      const xs = c.inputs.map((v) => decode(v) as Tensor)
      const [x, a, b, w, bias] = xs as [Tensor, Tensor, Tensor, Tensor, Tensor]
      const p = c.params
      const eps = p['eps'] as number | undefined
      let y: Value
      switch (c.fn) {
        case 'batchNormTrain':
        case 'batchNormRunningMean':
        case 'batchNormRunningVar': {
          const r = batchNorm(x, {
            runningMean: a,
            runningVar: b,
            weight: w,
            bias,
            training: true,
            momentum: p['momentum'] as number,
            eps: eps as number,
          })
          y = (
            c.fn === 'batchNormTrain' ? r.output : c.fn === 'batchNormRunningMean' ? r.runningMean : r.runningVar
          ) as Value
          break
        }
        case 'batchNormEval':
          y = batchNorm(x, {
            runningMean: a,
            runningVar: b,
            weight: w,
            bias,
            training: false,
            eps: eps as number,
          }).output
          break
        case 'layerNorm':
          y = layerNorm(x, p['shape'] as number[], { ...(a === undefined ? {} : { weight: a, bias: b }) })
          break
        case 'rmsNorm':
          y = rmsNorm(x, p['shape'] as number[], {
            ...(a === undefined ? {} : { weight: a }),
            ...(eps === undefined ? {} : { eps }),
          })
          break
        case 'groupNorm':
          y = groupNorm(x, p['groups'] as number, { weight: a, bias: b })
          break
        case 'instanceNorm':
          y = instanceNorm(x, { weight: a, bias: b }).output
          break
        case 'instanceNormRunning': {
          const r = instanceNorm(x, { runningMean: a, runningVar: b, momentum: 0.3 })
          const part = p['part'] as string
          y = (part === 'output' ? r.output : part === 'runningMean' ? r.runningMean : r.runningVar) as Value
          break
        }
        case 'localResponseNorm':
          y = localResponseNorm(x, p['size'] as number, p['alpha'] as number, p['beta'] as number, p['k'] as number)
          break
        case 'normalize':
          y = normalize(x, p['p'] as number, p['axis'] as number)
          break
        case 'weightNormLinear':
          // inputs: x, g, v, bias
          y = linear(x, weightNorm(b, a), w)
          break
        default: {
          // spectralNormLinear(Eval): inputs x, weight_orig, bias, u, v
          const r = spectralNorm(a, w, bias, { training: c.fn === 'spectralNormLinear' })
          const part = (p['part'] as string | undefined) ?? 'output'
          y = part === 'u' ? r.u : part === 'v' ? r.v : linear(x, r.weight, b)
        }
      }
      expectMatches(y, c.output, 1e-11, 1e-12, c.name)
    })
  }
})
