import { describe, expect, it } from 'vitest'
import { filmstripSteps, sampleGrid, sampleStepAt } from './sample-grid.ts'

const text = (steps: number[]) =>
  sampleGrid({ modality: 'text', steps: steps.map((step) => ({ step, samples: [`s${step}`] })) })

describe('the sample grid record', () => {
  it('checks the images against their size and the text against its modality', () => {
    expect(() => sampleGrid({ modality: 'image', steps: [] })).toThrow(/height and width/)
    expect(() =>
      sampleGrid({ modality: 'image', height: 2, width: 2, steps: [{ step: 0, samples: [[0, 1]] }] }),
    ).toThrow(/4 pixel/)
    expect(() =>
      sampleGrid({ modality: 'image', height: 1, width: 1, channels: 3, steps: [{ step: 0, samples: [[0, 1, 0]] }] }),
    ).not.toThrow()
    expect(() => sampleGrid({ modality: 'text', steps: [{ step: 0, samples: [[1]] }] })).toThrow(/not text/)
    expect(() => sampleGrid({ modality: 'text', steps: [{ step: 0, samples: ['a'], labels: [] }] })).toThrow(/labels/)
    expect(() =>
      sampleGrid({
        modality: 'text',
        steps: [
          { step: 3, samples: [] },
          { step: 1, samples: [] },
        ],
      }),
    ).toThrow(/out of order/)
  })

  it('reads the latest recorded step at or before a step, and the first before any', () => {
    const g = text([2, 5, 9])
    expect(sampleStepAt(g, 6)?.samples).toEqual(['s5'])
    expect(sampleStepAt(g, 9)?.samples).toEqual(['s9'])
    expect(sampleStepAt(g, 0)?.samples).toEqual(['s2'])
    expect(sampleStepAt(text([]), 3)).toBeUndefined()
  })

  it('picks the earlier steps of a filmstrip over a log scale, in order, never the viewed one', () => {
    const g = text([0, 1, 2, 4, 8, 16, 32])
    expect(filmstripSteps(g, 32, 3).map((s) => s.step)).toEqual([0, 4, 16])
    expect(filmstripSteps(g, 4, 5).map((s) => s.step)).toEqual([0, 1, 2])
    expect(filmstripSteps(g, 0, 3)).toEqual([])
    expect(filmstripSteps(g, 32, 1).map((s) => s.step)).toEqual([0])
  })
})
