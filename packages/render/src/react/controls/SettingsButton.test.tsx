import { dict, int } from 'omai-engine/space'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { FigureFrame } from '../FigureFrame.tsx'
import { SettingsButton, SpaceSettings } from './SettingsButton.tsx'

describe('SettingsButton', () => {
  it('draws only a labelled gear button while closed, with no dialog in the page', () => {
    const html = renderToString(
      <SettingsButton value={1} onApply={() => undefined} title="Figure settings">
        {(draft) => <span>{draft}</span>}
      </SettingsButton>,
    )
    expect(html).toContain('aria-label="Figure settings"')
    expect(html).toContain('aria-haspopup="dialog"')
    expect(html).toContain('aria-expanded="false"')
    expect(html).not.toContain('role="dialog"')
    expect(html).toContain('<svg')
  })

  it('can be disabled', () => {
    const html = renderToString(
      <SettingsButton value={1} onApply={() => undefined} disabled>
        {() => null}
      </SettingsButton>,
    )
    expect(html).toContain('disabled')
  })

  it('draws the small toolbar variant for a figure', () => {
    const control = renderToString(
      <SettingsButton value={1} onApply={() => undefined}>
        {() => null}
      </SettingsButton>,
    )
    const toolbar = renderToString(
      <SettingsButton value={1} onApply={() => undefined} appearance="toolbar">
        {() => null}
      </SettingsButton>,
    )
    expect(control).toContain('size-[calc(2rem*')
    expect(toolbar).toContain('size-7')
  })
})

describe('SpaceSettings', () => {
  it('is a settings button for a space', () => {
    const html = renderToString(
      <SpaceSettings space={dict({ k: int(1, 9, { default: 3 }) })} value={{ k: 3 }} onApply={() => undefined} />,
    )
    expect(html).toContain('aria-label="Settings"')
  })
})

describe('FigureFrame settings slot', () => {
  it('puts the settings control in the toolbar beside the export and expand buttons', () => {
    const html = renderToString(
      <FigureFrame
        width={200}
        height={100}
        settings={
          <SettingsButton value={1} onApply={() => undefined} appearance="toolbar" title="Figure settings">
            {() => null}
          </SettingsButton>
        }
      >
        <div />
      </FigureFrame>,
    )
    const settings = html.indexOf('aria-label="Figure settings"')
    expect(settings).toBeGreaterThan(-1)
    expect(settings).toBeLessThan(html.indexOf('aria-label="Export"'))
    expect(html.indexOf('aria-label="Export"')).toBeLessThan(html.indexOf('aria-label="Expand"'))
  })

  it('draws a toolbar even when the figure has no export or expand button but has settings', () => {
    const html = renderToString(
      <FigureFrame
        width={200}
        height={100}
        exportFormats={[]}
        expandable={false}
        settings={
          <SettingsButton value={1} onApply={() => undefined} appearance="toolbar" title="Only settings">
            {() => null}
          </SettingsButton>
        }
      >
        <div />
      </FigureFrame>,
    )
    expect(html).toContain('aria-label="Only settings"')
  })
})
