import { describe, expect, it } from 'vitest'
import { dict } from '../space/index.ts'
import { kernelMatrix, kernelSpace, resolveKernel } from './named.ts'
import { rbfKernel } from './stationary.ts'

describe('named kernels', () => {
  const x = [
    [0, 0],
    [2, 0],
  ]
  it('resolves gamma against the data as scikit-learn does', () => {
    const base = { kernel: 'rbf' as const, degree: 3, coef0: 1 }
    expect(resolveKernel({ ...base, gamma: 'auto' }, x).gamma).toBe(0.5)
    // Var(X) over all entries is 0.75, so 1 / (d Var) = 2/3.
    expect(resolveKernel({ ...base, gamma: 'scale' }, x).gamma).toBeCloseTo(2 / 3, 12)
    expect(resolveKernel({ ...base, gamma: 'scale' }, [[1, 1]]).gamma).toBe(1)
    expect(resolveKernel({ ...base, gamma: 0.3 }, x).gamma).toBe(0.3)
  })

  it('evaluates metrics by name and kernel objects directly', () => {
    const resolved = { kernel: 'rbf' as const, gamma: 0.5, degree: 3, coef0: 1 }
    const k = kernelMatrix({ kernel: 'rbf' }, resolved, x)
    expect(Array.from(k)).toEqual([1, Math.exp(-2), Math.exp(-2), 1])
    const custom = kernelMatrix({ kernel: rbfKernel({ lengthScale: 1 }) }, { ...resolved, kernel: 'custom' }, x)
    expect(Array.from(custom)).toEqual([1, Math.exp(-2), Math.exp(-2), 1])
  })

  it('offers knobs for names and none for a kernel object', () => {
    const space = dict(kernelSpace({ kernel: 'poly', gamma: 'scale', degree: 3, coef0: 1 }))
    expect(Object.keys(space.spaces)).toEqual(['kernel', 'degree', 'coef0'])
    expect(kernelSpace({ kernel: rbfKernel(), gamma: 1, degree: 3, coef0: 1 })).toEqual({})
  })
})
