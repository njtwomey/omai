import * as echarts from 'echarts'
import { describe, expect, it } from 'vitest'
import { stateSequence } from 'whitebox-engine/exposition'
import { runsOf, stateSequenceView } from './sequences.ts'

type Opt = { grid: unknown[]; title: { text?: string }[] | { text?: string } }

describe('state sequence views', () => {
  it('split a path into runs', () => {
    expect(runsOf([0, 0, 1, 1, 1, 0])).toEqual([
      [0, 1, 0],
      [2, 4, 1],
      [5, 5, 0],
    ])
  })

  it('stack the posteriors over strips of the Viterbi path and the truth, and render', () => {
    const r = stateSequence({
      method: 'hmm',
      states: ['fair', 'loaded'],
      posteriors: [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
      ],
      decoded: [0, 0, 1, 1],
      truth: [0, 1, 1, 1],
      observations: { values: [0, 3, 5, 5], name: 'roll', symbols: ['1', '2', '3', '4', '5', '6'] },
      stepName: 'roll',
    })
    const opt = stateSequenceView(r) as Opt
    expect(opt.grid).toHaveLength(3)
    expect(JSON.stringify(opt.title)).toContain('75.0%')
    const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 720, height: 360 })
    chart.setOption(opt as echarts.EChartsCoreOption)
    expect(chart.renderToSVGString()).toContain('<svg')
    chart.dispose()
  })
})
