import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { lineChart } from '../charts/lines.ts'
import { figure } from '../figure/figure.ts'
import { Chart } from './Chart.tsx'
import { loadEcharts } from './echarts.ts'
import { ThemeProvider, useColourMode, useTokens } from './theme.tsx'

function ModeProbe(): string {
  return `${useColourMode()}:${useTokens().bg}`
}

describe('react bindings', () => {
  it('render a chart container of the figure size on the server', () => {
    const html = renderToString(
      <Chart option={lineChart([{ y: [1, 2] }])} figure={figure({ width: 300, aspect: 2 })} />,
    )
    expect(html).toContain('width:300px')
    expect(html).toContain('height:150px')
  })
  it('default to dark, and follow an explicit mode', () => {
    expect(renderToString(<ModeProbe />)).toBe('dark:#0d1117')
    expect(
      renderToString(
        <ThemeProvider mode="light">
          <ModeProbe />
        </ThemeProvider>,
      ),
    ).toBe('light:#ffffff')
  })
  it('loads ECharts with the whitebox themes', () => {
    expect(typeof loadEcharts().init).toBe('function')
  })
})
