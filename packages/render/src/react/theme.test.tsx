import { circle, rect, scene } from 'omai-engine/scene'
import type { ReactElement } from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, describe, expect, it } from 'vitest'
import { echartsTheme, themeName } from '../theme/echarts.ts'
import { tokens } from '../theme/tokens.ts'
import { mountElement, resetGlobals } from './_harness.tsx'
import { NumberField } from './controls/NumberField.tsx'
import { SceneView } from './SceneView.tsx'
import { ThemeProvider, useChartTheme, useColourMode, useThemeVariables, useTokens } from './theme.tsx'

afterEach(() => {
  resetGlobals()
})

function Probe(): ReactElement {
  const t = useTokens()
  return (
    <b>
      {useColourMode()}:{t.fg}:{t.accent}
    </b>
  )
}
const probe = (el: ReactElement): string =>
  renderToString(el)
    .replace(/<[^>]+>/g, '')
    .replace(/<!-- -->/g, '')

// What the hooks returned in the latest render of a `Grab`.
const got: {
  tokens?: unknown
  chart?: ReturnType<typeof useChartTheme>
  vars?: ReturnType<typeof useThemeVariables>
} = {}
function GrabTokens(props: { readonly mode?: 'light' | 'dark' }): null {
  got.tokens = useTokens(props.mode)
  return null
}
function GrabChart(): null {
  got.chart = useChartTheme()
  return null
}
function GrabVars(): null {
  got.vars = useThemeVariables()
  return null
}

/** The wrapper div of a provider mounted in the stand-in DOM: its props. */
async function wrapper(el: ReactElement): Promise<Readonly<Record<string, unknown>>> {
  const m = await mountElement(el)
  const first = m.elements()[0]?.props ?? {}
  await m.unmount()
  return first
}

describe('ThemeProvider modes', () => {
  it('nest: an inner mode wins, one without a mode inherits, and the page mode is the default', () => {
    expect(probe(<Probe />)).toBe(`dark:${tokens('dark').fg}:${tokens('dark').accent}`)
    const light = `light:${tokens('light').fg}:${tokens('light').accent}`
    expect(
      probe(
        <ThemeProvider mode="dark">
          <ThemeProvider mode="light">
            <Probe />
          </ThemeProvider>
        </ThemeProvider>,
      ),
    ).toBe(light)
    expect(
      probe(
        <ThemeProvider mode="light">
          <ThemeProvider>
            <Probe />
          </ThemeProvider>
        </ThemeProvider>,
      ),
    ).toBe(light)
    // Back out of the inner one, the outer mode again.
    expect(
      probe(
        <ThemeProvider mode="dark">
          <ThemeProvider mode="light">
            <Probe />
          </ThemeProvider>
          <Probe />
        </ThemeProvider>,
      ),
    ).toBe(`${light}dark:${tokens('dark').fg}:${tokens('dark').accent}`)
  })

  it('renders one wrapper that does not disturb the layout, carries data-theme, and takes className and style', async () => {
    const props = await wrapper(
      <ThemeProvider mode="light" className="deck" style={{ color: 'red' }}>
        <i />
      </ThemeProvider>,
    )
    expect(props['data-theme']).toBe('light')
    expect(props['className']).toBe('deck')
    expect(props['style']).toMatchObject({ display: 'contents', color: 'red' })
  })

  it('writes every --wb variable of a pinned mode that differs from the one around it, and none otherwise', async () => {
    // The page is dark: pinning light changes the mode, so the controls need the light variables.
    const pinned = (await wrapper(<ThemeProvider mode="light" />))['style'] as Record<string, string>
    expect(pinned['--wb-fg']).toBe(tokens('light').fg)
    expect(pinned['--wb-bg']).toBe(tokens('light').bg)
    expect(pinned['--wb-tab3']).toBe(tokens('light').tab3)
    // Pinning the mode already in effect leaves the host's own variables alone.
    const same = (await wrapper(<ThemeProvider mode="dark" />))['style'] as Record<string, string>
    expect(Object.keys(same)).toEqual(['display'])
    expect(Object.keys((await wrapper(<ThemeProvider />))['style'] as object)).toEqual(['display'])
  })
})

describe('ThemeProvider tokens', () => {
  it('merge over the mode tokens for everything below: an object, one per mode, or a function of the mode', () => {
    const accent = (t: ReactElement): string => probe(t).split(':')[2] as string
    expect(
      accent(
        <ThemeProvider tokens={{ accent: '#7c3aed' }}>
          <Probe />
        </ThemeProvider>,
      ),
    ).toBe('#7c3aed')
    const perMode = { light: { accent: '#111111' }, dark: { accent: '#eeeeee' } }
    expect(
      accent(
        <ThemeProvider mode="dark" tokens={perMode}>
          <Probe />
        </ThemeProvider>,
      ),
    ).toBe('#eeeeee')
    expect(
      accent(
        <ThemeProvider mode="light" tokens={perMode}>
          <Probe />
        </ThemeProvider>,
      ),
    ).toBe('#111111')
    expect(
      accent(
        <ThemeProvider mode="light" tokens={(mode) => ({ accent: mode === 'light' ? '#abc' : '#def' })}>
          <Probe />
        </ThemeProvider>,
      ),
    ).toBe('#abc')
    // The tokens not overridden are the mode's.
    expect(
      probe(
        <ThemeProvider mode="light" tokens={{ accent: '#7c3aed' }}>
          <Probe />
        </ThemeProvider>,
      ),
    ).toContain(`:${tokens('light').fg}:`)
  })

  it('nest: the inner over the outer, and an outer override survives a pinned mode', () => {
    expect(
      probe(
        <ThemeProvider mode="light" tokens={{ accent: '#111111', fg: '#222222' }}>
          <ThemeProvider tokens={{ accent: '#333333' }}>
            <Probe />
          </ThemeProvider>
        </ThemeProvider>,
      ),
    ).toBe('light:#222222:#333333')
    expect(
      probe(
        <ThemeProvider mode="dark" tokens={{ accent: '#111111' }}>
          <ThemeProvider mode="light">
            <Probe />
          </ThemeProvider>
        </ThemeProvider>,
      ),
    ).toBe(`light:${tokens('light').fg}:#111111`)
  })

  it('is the same object as the mode tokens when nothing overrides them, and an explicit mode keeps the overrides', () => {
    renderToString(<GrabTokens />)
    expect(got.tokens).toBe(tokens('dark'))
    renderToString(
      <ThemeProvider tokens={{ accent: '#7c3aed' }}>
        <GrabTokens mode="light" />
      </ThemeProvider>,
    )
    expect(got.tokens).toMatchObject({ mode: 'light', accent: '#7c3aed', fg: tokens('light').fg })
  })

  it('write the overridden tokens, and only those, as variables on the wrapper', async () => {
    const style = (await wrapper(<ThemeProvider tokens={{ accent: '#7c3aed', danger: '#ff0000' }} />))[
      'style'
    ] as Record<string, string>
    expect(style).toEqual({ display: 'contents', '--wb-accent': '#7c3aed', '--wb-danger': '#ff0000' })
  })

  it('reach a chart: the ECharts theme is made from them and named by them, the standard one stays', () => {
    renderToString(<GrabChart />)
    expect(got.chart?.name).toBe(themeName('dark'))
    renderToString(
      <ThemeProvider mode="light" tokens={{ fg: '#123456', tab0: '#fe0000' }}>
        <GrabChart />
      </ThemeProvider>,
    )
    const made = got.chart
    expect(made?.name).toMatch(/^omai-light-[0-9a-z]+$/)
    const theme = echartsTheme('light', made?.tokens)
    expect((theme['textStyle'] as { color: string }).color).toBe('#123456')
    expect(theme.color[0]).toBe('#fe0000')
    expect(theme.color[1]).toBe(tokens('light').tab1)
    // The same override names the same theme; another one another.
    renderToString(
      <ThemeProvider mode="light" tokens={{ tab0: '#fe0000', fg: '#123456' }}>
        <GrabChart />
      </ThemeProvider>,
    )
    expect(got.chart?.name).toBe(made?.name)
    renderToString(
      <ThemeProvider mode="light" tokens={{ fg: '#654321' }}>
        <GrabChart />
      </ThemeProvider>,
    )
    expect(got.chart?.name).not.toBe(made?.name)
  })

  it('reach a scene, a nested mode included', () => {
    const s = scene(100, 50, [rect(0, 0, 10, 10, { fill: 'accent' }), circle(20, 20, 5, { fill: 'bg' })])
    const html = renderToString(
      <ThemeProvider mode="dark" tokens={{ accent: '#7c3aed' }}>
        <SceneView scene={s} />
        <ThemeProvider mode="light">
          <SceneView scene={s} />
        </ThemeProvider>
      </ThemeProvider>,
    )
    const [first, second] = html.split('</svg>')
    expect(first).toContain('#7c3aed')
    expect(first).toContain(`fill="${tokens('dark').bg}"`)
    expect(second).toContain('#7c3aed')
    expect(second).toContain(`fill="${tokens('light').bg}"`)
  })

  it('reach a control through the variables its classes read', async () => {
    const m = await mountElement(
      <ThemeProvider tokens={{ accent: '#7c3aed' }}>
        <NumberField label="k" value={3} onChange={() => undefined} min={1} max={9} />
      </ThemeProvider>,
    )
    const els = m.elements()
    expect(els[0]?.props['style']).toMatchObject({ '--wb-accent': '#7c3aed' })
    expect(els.some((e) => String(e.props['className'] ?? '').includes('(--wb-accent)'))).toBe(true)
    await m.unmount()
  })
})

describe('ThemeProvider cssVariables', () => {
  it('maps token names to the host’s variables as var(...), and takes any other value as written', async () => {
    const style = (
      await wrapper(
        <ThemeProvider
          cssVariables={{ fg: '--foreground', muted: '--muted-foreground', accent: 'hsl(var(--primary))' }}
        />,
      )
    )['style'] as Record<string, string>
    expect(style).toEqual({
      display: 'contents',
      '--wb-fg': 'var(--foreground)',
      '--wb-muted': 'var(--muted-foreground)',
      '--wb-accent': 'hsl(var(--primary))',
    })
  })

  it('puts the mapping over a token override of the same name, and a nested pinned mode keeps it', async () => {
    const both = (await wrapper(<ThemeProvider tokens={{ fg: '#fff' }} cssVariables={{ fg: '--foreground' }} />))[
      'style'
    ] as Record<string, string>
    expect(both['--wb-fg']).toBe('var(--foreground)')
    // An inner provider that pins another mode writes the whole set, and keeps the outer mapping.
    const m = await mountElement(
      <ThemeProvider cssVariables={{ fg: '--foreground' }}>
        <ThemeProvider mode="light" />
      </ThemeProvider>,
    )
    const inner = m.elements().filter((e) => e.props['data-theme'] !== undefined)[1]
    await m.unmount()
    const style = inner?.props['style'] as Record<string, string>
    expect(style['--wb-bg']).toBe(tokens('light').bg)
    // The mapped one is the outer's, which the inner wrapper inherits, so it does not write it again.
    expect(style['--wb-fg']).toBeUndefined()
  })

  it('gives popups the variables of every provider above, inner over outer', () => {
    renderToString(<GrabVars />)
    expect(got.vars).toEqual({ vars: {} })
    renderToString(
      <ThemeProvider tokens={{ accent: '#111' }} cssVariables={{ fg: '--foreground' }}>
        <ThemeProvider tokens={{ accent: '#222' }}>
          <GrabVars />
        </ThemeProvider>
      </ThemeProvider>,
    )
    expect(got.vars?.vars).toEqual({ '--wb-accent': '#222', '--wb-fg': 'var(--foreground)' })
    expect(got.vars?.mode).toBe('dark')
  })
})
