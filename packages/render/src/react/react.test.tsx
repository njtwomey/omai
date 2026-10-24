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

describe('FigureFrame', () => {
  it('renders a fixed-size frame with expand and export buttons, sizing charts inside', async () => {
    const { FigureFrame } = await import('./FigureFrame.tsx')
    const html = renderToString(
      <FigureFrame width={320} height={200} title="T" caption="c">
        <Chart option={lineChart([{ y: [1, 2] }])} />
      </FigureFrame>,
    )
    expect(html).toContain('aria-label="Expand"')
    expect(html).toContain('aria-label="Export"')
    expect(html).toContain('<figcaption')
    expect(html.match(/width:320px/g)?.length).toBeGreaterThanOrEqual(2)
    const sized = renderToString(
      <FigureFrame exportFormats={[]} expandable={false}>
        {(size) => `${size.width}x${size.height}:${size.expanded}`}
      </FigureFrame>,
    )
    expect(sized).toContain('640x400:false')
    expect(sized).not.toContain('aria-label')
  })
})
