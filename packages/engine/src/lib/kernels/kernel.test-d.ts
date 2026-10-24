import { describe, expectTypeOf, it } from 'vitest'
import type { Tensor, Value } from '../tensor/index.ts'
import {
  type FeatureMap,
  type Kernel,
  type KernelParams,
  nystroem,
  randomFourierFeatures,
  rbfKernel,
  sumKernel,
  theta,
  withTheta,
} from './index.ts'

describe('kernel types', () => {
  it('kernels evaluate to values and rebuild from hyperparameter pytrees', () => {
    const k = sumKernel(rbfKernel(), rbfKernel({ lengthScale: [1, 2] }))
    expectTypeOf(k).toEqualTypeOf<Kernel>()
    expectTypeOf(k.k([[1, 2]], [[3, 4]])).toEqualTypeOf<Value>()
    expectTypeOf(k.params).toEqualTypeOf<KernelParams>()
    expectTypeOf(k.withParams).parameter(0).toEqualTypeOf<KernelParams>()
    expectTypeOf(theta(k)).toEqualTypeOf<Tensor<'float64'>>()
    expectTypeOf(withTheta(k, [0, 0, 0])).toEqualTypeOf<Kernel>()
  })
  it('feature maps are feature maps', () => {
    expectTypeOf(randomFourierFeatures({ weights: [[1]], offset: [0] })).toExtend<FeatureMap>()
    expectTypeOf(nystroem(rbfKernel(), [[0]])).toExtend<FeatureMap>()
    // @ts-expect-error a randomised map needs its input dimension (or its draws)
    randomFourierFeatures({ nComponents: 3 })
  })
})
