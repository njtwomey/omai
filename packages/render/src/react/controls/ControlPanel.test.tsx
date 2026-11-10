import { afterEach, describe, expect, it } from 'vitest'
import { mountElement, resetGlobals } from '../_harness.tsx'
import { ControlPanel, PanelArea } from './ControlPanel.tsx'

afterEach(resetGlobals)

// The grid the controls sit in: the div that has a grid template.
async function grid(element: React.ReactElement): Promise<Record<string, unknown>> {
  const m = await mountElement(element)
  const found = m.elements().find((e) => {
    const style = e.props['style'] as Record<string, unknown> | undefined
    return style?.['gridTemplateColumns'] !== undefined || style?.['gridTemplateRows'] !== undefined
  })
  await m.unmount()
  return (found?.props['style'] ?? {}) as Record<string, unknown>
}

describe('ControlPanel columns', () => {
  it('one column is a plain full-width track', async () => {
    expect((await grid(<ControlPanel>x</ControlPanel>))['gridTemplateColumns']).toBe('minmax(0, 1fr)')
    expect((await grid(<ControlPanel columns={1}>x</ControlPanel>))['gridTemplateColumns']).toBe('minmax(0, 1fr)')
  })

  it('several columns are at most that many, each at least the controls own minimum wide, so a narrow screen has fewer', async () => {
    const tracks = String((await grid(<ControlPanel columns={3}>x</ControlPanel>))['gridTemplateColumns'])
    expect(tracks).toBe(
      'repeat(auto-fill, minmax(min(100%, max(calc(9rem*var(--omai-control-scale,1)), calc((100% - 2rem) / 3))), 1fr))',
    )
  })

  it('minColumnWidth is pixels or any CSS length', async () => {
    expect(
      String(
        (
          await grid(
            <ControlPanel columns={2} minColumnWidth={200}>
              x
            </ControlPanel>,
          )
        )['gridTemplateColumns'],
      ),
    ).toContain('max(200px, calc((100% - 1rem) / 2))')
    expect(
      String(
        (
          await grid(
            <ControlPanel columns={2} minColumnWidth="12rem">
              x
            </ControlPanel>,
          )
        )['gridTemplateColumns'],
      ),
    ).toContain('max(12rem, ')
  })

  it('rows and areas keep their own grids', async () => {
    const rows = await grid(
      <ControlPanel rows={2} columns={3}>
        x
      </ControlPanel>,
    )
    expect(rows['gridTemplateRows']).toBe('repeat(2, auto)')
    expect(rows['gridTemplateColumns']).toBeUndefined()
    const areas = await grid(
      <ControlPanel areas={['a a b']}>
        <PanelArea name="a">x</PanelArea>
      </ControlPanel>,
    )
    expect(areas['gridTemplateColumns']).toBe('repeat(3, minmax(0, 1fr))')
  })
})
