import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { bool, choice, dict, int, real } from 'whitebox-engine/space'
import { Choice } from './controls/Choice.tsx'
import { ControlLayout, ControlPanel, PanelArea } from './controls/ControlPanel.tsx'
import { DatasetPicker } from './controls/DatasetPicker.tsx'
import { selectEntry } from './controls/datasets.ts'
import { MethodControls } from './controls/MethodControls.tsx'
import { NumberField } from './controls/NumberField.tsx'
import { SeedInput } from './controls/SeedInput.tsx'
import { Select } from './controls/Select.tsx'
import { SpaceControls } from './controls/SpaceControls.tsx'
import { Toggle } from './controls/Toggle.tsx'
import { Brush } from './interaction/Brush.tsx'
import { DraggablePoints } from './interaction/DraggablePoints.tsx'
import { InteractionOverlay } from './interaction/InteractionOverlay.tsx'
import { NearestHover } from './interaction/NearestHover.tsx'
import { linearProjection } from './interaction/projection.ts'
import { Player } from './Player.tsx'
import type { PlaybackController } from './usePlayback.ts'

const noop = (): void => {}

describe('React bindings render on the server', () => {
  it('draws every control', () => {
    const html = renderToString(
      <ControlLayout
        placement="left"
        controls={
          <ControlPanel title="controls" areas={['a b', 'c c']}>
            <PanelArea name="a">
              <NumberField label="rate" value={0.01} min={1e-4} max={1} log onChange={noop} />
            </PanelArea>
            <PanelArea name="b">
              <Select label="kernel" value="rbf" options={['rbf', 'linear']} onChange={noop} />
            </PanelArea>
            <PanelArea name="c">
              <Choice label="side" value={1} options={[1, 2]} onChange={noop} cycle />
              <Toggle label="log" value onChange={noop} />
              <SeedInput value={3} onChange={noop} />
            </PanelArea>
          </ControlPanel>
        }
      >
        figure
      </ControlLayout>,
    )
    expect(html).toContain('rate')
    expect(html).toContain('role="switch"')
    expect(html).toContain('grid-template-areas')
  })

  it('draws controls from a space and a dataset picker', () => {
    const space = dict({ k: int(1, 9), lr: real(1e-3, 1, { log: true }), act: choice(['relu', 'tanh']), b: bool() })
    const html = renderToString(
      <>
        <SpaceControls space={space} value={{ k: 3, lr: 0.1, act: 'relu', b: false }} onChange={noop} />
        <DatasetPicker collection="classification2d" value={selectEntry('classification2d')} onChange={noop} />
      </>,
    )
    expect(html).toContain('act')
    expect(html).toContain('Two moons')
    expect(html).toContain('noise')
  })

  it('draws the player and overlay handles', () => {
    const controller: PlaybackController = {
      frame: 3,
      count: 10,
      playing: false,
      fps: 10,
      open: false,
      play: noop,
      pause: noop,
      toggle: noop,
      step: noop,
      seek: noop,
      setFps: noop,
      reset: noop,
    }
    expect(renderToString(<Player controller={controller} stamp />)).toContain('0:00.30')
    const projection = linearProjection({ x: [0, 1], y: [0, 1], width: 100, height: 100 })
    const svg = renderToString(
      <InteractionOverlay projection={projection} width={100} height={100}>
        <DraggablePoints points={[[0.5, 0.5]]} onChange={noop} add />
      </InteractionOverlay>,
    )
    expect(svg).toContain('cx="50"')
  })

  it('draws method controls from a hyperSpace, and the brush and hover tools', () => {
    const estimator = {
      name: 'toy',
      kind: 'classifier',
      hyper: { depth: 3 },
      hyperSpace: dict({ depth: int(1, 9, { default: 3, label: 'depth' }) }),
    } as unknown as Parameters<typeof MethodControls>[0]['estimator']
    expect(renderToString(<MethodControls estimator={estimator} onChange={noop} />)).toContain('depth')
    const projection = linearProjection({ x: [0, 1], y: [0, 1], width: 100, height: 100 })
    const svg = renderToString(
      <InteractionOverlay projection={projection} width={100} height={100}>
        <Brush value={{ x: [0, 0.5], y: [0, 0.5] }} onChange={noop} />
        <NearestHover points={[[0.5, 0.5]]} onHover={noop} />
      </InteractionOverlay>,
    )
    expect(svg).toContain('width="50"')
  })
})
