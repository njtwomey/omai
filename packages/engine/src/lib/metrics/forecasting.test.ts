import { describe, expect, it } from 'vitest'
import * as M from './index.ts'

describe('forecasting metrics', () => {
  it('aliases the forecasting MAPE to the regression metric', () => {
    expect(M.mape).toBe(M.meanAbsolutePercentageError)
  })
})
