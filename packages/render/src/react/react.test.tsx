import { structure } from 'mdml-engine/scene'
import { type ReactElement, useState } from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { lineChart } from '../charts/lines.ts'
import { figure } from '../figure/figure.ts'
import { inAct, mountElement, resetGlobals } from './_harness.tsx'
import { Chart } from './Chart.tsx'
import { loadEcharts } from './echarts.ts'
import { StructureView } from './StructureView.tsx'
import { ThemeProvider, useColourMode, useTokens } from './theme.tsx'

function ModeProbe(): string {
  return `${useColourMode()}:${useTokens().bg}`
}

afterEach(() => {
  resetGlobals()
  vi.unstubAllGlobals()
})

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
  it('observe the page class with one observer however many figures ask, and disconnect with the last', async () => {
    const made: { disconnected: boolean }[] = []
    vi.stubGlobal(
      'MutationObserver',
      class {
        disconnected = false
        constructor() {
          made.push(this)
        }
        observe(): void {}
        disconnect(): void {
          this.disconnected = true
        }
      },
    )
    const many = (
      <>
        <ModeProbe />
        <ModeProbe />
        <ModeProbe />
      </>
    )
    const first = await mountElement(many, { browser: true })
    const second = await mountElement(<ModeProbe />, { browser: true })
    expect(made).toHaveLength(1)
    await first.unmount()
    expect(made[0]?.disconnected).toBe(false)
    await second.unmount()
    expect(made[0]?.disconnected).toBe(true)
  })
  it('gives a structure view one stable ref callback, so a render does not set state twice', async () => {
    const s = structure('bayes-net', [{ id: 'a', label: 'a', role: 'latent' }], [])
    let bump = (): void => {}
    const Host = (): ReactElement => {
      const [n, setN] = useState(0)
      bump = () => setN(n + 1)
      return <StructureView structure={s} width={300 + n} />
    }
    const mounted = await mountElement(<Host />, { browser: true })
    const refOf = (): unknown => mounted.elements().find((e) => e.props['onPointerDown'] !== undefined)?.props['ref']
    const before = refOf()
    expect(typeof before).toBe('function')
    await inAct(bump)
    expect(refOf()).toBe(before)
    await mounted.unmount()
  })
  it('loads ECharts with the mdml themes', () => {
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

describe('FigureGrid and maths', () => {
  it('overlays text and TeX cells on a subplot chart', async () => {
    const { FigureGrid } = await import('./FigureGrid.tsx')
    const { Equation, Derivation, MathProvider, Tex } = await import('./MathView.tsx')
    const { gridSpec } = await import('../figure/grid.ts')
    const { lines } = await import('../charts/lines.ts')
    const grid = gridSpec(1, 3, {
      cells: [
        { id: 'a', row: 0, col: 0 },
        { id: 't', row: 0, col: 1, kind: 'tex', content: '\\xvec' },
        { id: 'n', row: 0, col: 2, kind: 'text', content: 'note' },
      ],
    })
    const html = renderToString(<FigureGrid grid={grid} panels={{ a: { layers: [lines([{ y: [1, 2] }])] } }} />)
    expect(html).toContain('<math')
    expect(html).toContain('note')
    const maths = renderToString(
      <MathProvider macros={{ '\\R': '\\mathbb{R}' }} output="html">
        <Tex>{'\\R'}</Tex>
        <Equation>{'x'}</Equation>
        <Derivation steps={[{ tex: 'a &= b' }]} />
      </MathProvider>,
    )
    expect(maths).toContain('katex')
  })
})
