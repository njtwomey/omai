import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import { conv, convTransposed, fold, type PaddingMode, unfold } from './conv.ts'

type Size = number | number[]

describe('convolutions against torch', () => {
  for (const c of loadFixture('nn/functional/conv').cases) {
    it(c.name, () => {
      const [x, w, b] = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      let y: Value
      if (c.fn.startsWith('convTranspose')) {
        y = convTransposed(x as Tensor, w as Tensor, {
          ...(b === undefined ? {} : { bias: b }),
          stride: p['stride'] as Size,
          padding: p['padding'] as Size,
          outputPadding: p['outputPadding'] as Size,
          groups: p['groups'] as number,
          dilation: p['dilation'] as Size,
        })
      } else if (c.fn === 'unfold') {
        y = unfold(x as Tensor, p['kernelSize'] as Size, { padding: p['padding'] as Size, stride: p['stride'] as Size })
      } else if (c.fn === 'fold') {
        y = fold(x as Tensor, p['outputSize'] as number[], p['kernelSize'] as Size, { stride: p['stride'] as Size })
      } else {
        y = conv(x as Tensor, w as Tensor, {
          ...(b === undefined ? {} : { bias: b }),
          stride: p['stride'] as Size,
          padding: p['padding'] as Size | 'same',
          dilation: p['dilation'] as Size,
          groups: p['groups'] as number,
          paddingMode: p['paddingMode'] as PaddingMode,
        })
      }
      expectMatches(y, c.output, 1e-11, 1e-12, c.name)
    })
  }
})
