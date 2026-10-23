import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import type { Module } from './module.ts'
import {
  BatchNorm1d,
  BatchNorm2d,
  BatchNorm3d,
  type BatchNormModuleOptions,
  GroupNorm,
  InstanceNorm2d,
  LayerNorm,
  RMSNorm,
} from './normalisation.ts'

type Params = Readonly<Record<string, unknown>>

const stateOf = (p: Params): Record<string, Tensor> =>
  Object.fromEntries(Object.entries(p['state'] as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))

function build(fn: string, p: Params): Module {
  const options: BatchNormModuleOptions = {
    ...(p['momentum'] === undefined ? {} : { momentum: p['momentum'] as number | null }),
    ...(p['eps'] === undefined ? {} : { eps: p['eps'] as number }),
    ...(p['affine'] === undefined ? {} : { affine: p['affine'] as boolean }),
    ...(p['track'] === undefined ? {} : { trackRunningStats: p['track'] as boolean }),
  }
  const features = p['features'] as number
  switch (fn) {
    case 'BatchNorm1d':
      return new BatchNorm1d(features, options)
    case 'BatchNorm2d':
      return new BatchNorm2d(features, options)
    case 'BatchNorm3d':
      return new BatchNorm3d(features, options)
    case 'InstanceNorm2d':
      return new InstanceNorm2d(features, options)
    case 'LayerNorm':
      return new LayerNorm(p['shape'] as number[])
    case 'RMSNorm':
      return new RMSNorm(p['shape'] as number[], { eps: p['eps'] as number })
    default:
      return new GroupNorm(p['groups'] as number, p['channels'] as number)
  }
}

describe('normalisation modules against torch', () => {
  for (const c of loadFixture('nn/module/normalisation').cases) {
    it(c.name, () => {
      const xs = c.inputs.map((v) => decode(v) as Tensor)
      const m = build(c.fn, c.params['config'] as Params).loadStateDict(stateOf(c.params))
      const part = c.params['part'] as string | undefined
      if (part === undefined) {
        expectMatches(m.forward(xs[0] as Tensor), c.output, 1e-11, 1e-12, c.name)
        return
      }
      m.train()
      const y1 = m.forward(xs[0] as Tensor)
      const y2 = m.forward(xs[1] as Tensor)
      m.eval()
      const y3 = m.forward(xs[2] as Tensor)
      const got: Value =
        part === 'train1'
          ? y1
          : part === 'train2'
            ? y2
            : part === 'eval'
              ? y3
              : (m.stateDict()[part.slice('state:'.length)] as Tensor)
      expectMatches(got, c.output, 1e-11, 1e-12, c.name)
    })
  }
})
