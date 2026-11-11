import { afterEach, describe, expect, it, vi } from 'vitest'
import { invoke, mountElement, resetGlobals } from '../_harness.tsx'
import { useProbe } from '../session/RowsPanel.tsx'
import { FigurePanel, PanelsStrip, usePanels } from './FigurePanel.tsx'

afterEach(() => {
  resetGlobals()
  vi.unstubAllGlobals()
})

const labels = { surface: 'Surface', pairs: 'Pair plot' }

function Page(props: { readonly live: never; readonly id?: string }): React.ReactElement {
  const panels = usePanels(props.id ?? 'test-page', labels)
  const value = useProbe(props.live, 1, panels.active('surface') ? 'surface' : undefined).value
  return (
    <div>
      <PanelsStrip panels={panels} />
      <FigurePanel panels={panels} name="surface">
        <p>{JSON.stringify(value)}</p>
      </FigurePanel>
      <FigurePanel panels={panels} name="pairs">
        <p>pairs body</p>
      </FigurePanel>
    </div>
  )
}

describe('FigurePanel', () => {
  it('folds and hides from its bar, lists every panel in the strip, and asks for no probe while folded or hidden', async () => {
    const probe = vi.fn((_t: number, path: string) => ({ path }))
    const live = { session: { t: 5, probe }, version: 0 } as never
    const m = await mountElement(<Page live={live} />)
    const find = (pred: (p: Record<string, unknown>) => boolean): ReturnType<typeof m.elements>[number] | undefined =>
      m.elements().find((e) => pred(e.props))
    const bar = (): ReturnType<typeof m.elements>[number] | undefined =>
      find((p) => p['aria-expanded'] !== undefined && p['aria-controls'] !== undefined)
    const strip = (label: string): ReturnType<typeof m.elements>[number] | undefined =>
      m.elements().find((e) => e.props['aria-pressed'] !== undefined && e.text === label)

    // All open at first: both bodies are drawn, the probe was asked for once, the strip lists both as on show.
    expect(bar()?.props['aria-expanded']).toBe(true)
    expect(m.elements().some((e) => e.text === 'pairs body')).toBe(true)
    expect(probe).toHaveBeenCalledTimes(1)
    expect(strip('Surface')?.props['aria-pressed']).toBe(true)

    // Folded: the body goes, the bar stays with aria-expanded false, and the probe is not asked for again.
    await invoke(bar(), 'onClick')
    expect(bar()?.props['aria-expanded']).toBe(false)
    expect(m.elements().some((e) => e.text.startsWith('{'))).toBe(false)
    // Open again: the answer already held comes back (no new computation).
    await invoke(bar(), 'onClick')
    expect(m.elements().some((e) => e.text.startsWith('{'))).toBe(true)
    expect(probe).toHaveBeenCalledTimes(1)

    // Hidden from the strip: the panel is not drawn at all, the strip still lists it, and no probe is requested.
    probe.mockClear()
    await invoke(strip('Pair plot'), 'onClick')
    expect(m.elements().some((e) => e.text === 'pairs body')).toBe(false)
    expect(strip('Pair plot')?.props['aria-pressed']).toBe(false)
    await invoke(
      find((p) => p['aria-label'] === 'Hide Surface'),
      'onClick',
    )
    expect(m.elements().some((e) => e.text.startsWith('{'))).toBe(false)
    expect(probe).not.toHaveBeenCalled()

    // Brought back from the strip.
    await invoke(strip('Pair plot'), 'onClick')
    expect(m.elements().some((e) => e.text === 'pairs body')).toBe(true)
    await m.unmount()
  })

  it('remembers the choice per page and starts open without storage', async () => {
    const store = new Map<string, string>()
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    })
    const first = await mountElement(
      <Page live={{ session: { t: 5, probe: () => null }, version: 0 } as never} id="remembered" />,
    )
    await invoke(
      first.elements().find((e) => e.props['aria-expanded'] !== undefined),
      'onClick',
    )
    await first.unmount()
    expect(JSON.parse(store.get('omai-panels:remembered') ?? '{}')).toEqual({ surface: 'folded' })
    const second = await mountElement(
      <Page live={{ session: { t: 5, probe: () => null }, version: 0 } as never} id="remembered" />,
    )
    expect(second.elements().find((e) => e.props['aria-expanded'] !== undefined)?.props['aria-expanded']).toBe(false)
    await second.unmount()

    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('blocked')
      },
    })
    const third = await mountElement(
      <Page live={{ session: { t: 5, probe: () => null }, version: 0 } as never} id="remembered" />,
    )
    expect(third.elements().find((e) => e.props['aria-expanded'] !== undefined)?.props['aria-expanded']).toBe(true)
    await invoke(
      third.elements().find((e) => e.props['aria-expanded'] !== undefined),
      'onClick',
    )
    expect(third.elements().find((e) => e.props['aria-expanded'] !== undefined)?.props['aria-expanded']).toBe(false)
    await third.unmount()
  })
})
