import { datasetRegistry } from 'mdml-engine/datasets'
import { environmentRegistry } from 'mdml-engine/environments'
import { thresholdConfusion } from 'mdml-engine/metrics'
import { structure } from 'mdml-engine/scene'
import { bool, choice, dict, int, real } from 'mdml-engine/space'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ContingencyTable } from './ContingencyTable.tsx'
import { Choice } from './controls/Choice.tsx'
import { ColumnOrder, moveItem } from './controls/ColumnOrder.tsx'
import { ControlLayout, ControlPanel, PanelArea } from './controls/ControlPanel.tsx'
import { DatasetPicker } from './controls/DatasetPicker.tsx'
import { DatasetSelector } from './controls/DatasetSelector.tsx'
import { selectEntry } from './controls/datasets.ts'
import { EntryList } from './controls/EntryList.tsx'
import { EnvironmentSelector } from './controls/EnvironmentSelector.tsx'
import { selectEnvironment } from './controls/environments.ts'
import { NumberField } from './controls/NumberField.tsx'
import { SeedInput } from './controls/SeedInput.tsx'
import { Select } from './controls/Select.tsx'
import { SpaceControls } from './controls/SpaceControls.tsx'
import { Toggle } from './controls/Toggle.tsx'
import { Brush } from './interaction/Brush.tsx'
import { DraggableLine } from './interaction/DraggableLine.tsx'
import { DraggablePoints } from './interaction/DraggablePoints.tsx'
import { InteractionOverlay } from './interaction/InteractionOverlay.tsx'
import { NearestHover } from './interaction/NearestHover.tsx'
import { linearProjection } from './interaction/projection.ts'
import { Player } from './Player.tsx'
import { StructureView } from './StructureView.tsx'
import { ThresholdExplorer } from './ThresholdExplorer.tsx'
import type { PlaybackController } from './usePlayback.ts'

const noop = (): void => {}

describe('React bindings render on the server', () => {
  it('draws column chips to reorder and flip', () => {
    const html = renderToString(<ColumnOrder order={['a', 'b']} onOrder={noop} inverted={['b']} onInvert={noop} />)
    expect(html).toContain('draggable="true"')
    expect(html).toContain('aria-pressed="true"')
    expect(moveItem(['a', 'b', 'c'], 2, 0)).toEqual(['c', 'a', 'b'])
  })

  it('draws draggable lines bound to a value', () => {
    const projection = linearProjection({ x: [0, 10], y: [0, 1], width: 200, height: 100 })
    const html = renderToString(
      <InteractionOverlay projection={projection} width={200} height={100}>
        <DraggableLine axis="x" value={4} onChange={noop} span={[0, 1]} min={0} max={10} step={0.5} label="threshold" />
        <DraggableLine axis="y" value={0.5} onChange={noop} />
      </InteractionOverlay>,
    )
    expect(html).toContain('role="slider"')
    expect(html).toContain('aria-valuenow="4"')
    expect(html).toContain('threshold')
  })

  it('draws a contingency table with its named cells and rates', () => {
    const table = thresholdConfusion([0, 0, 1, 1, 1], [0.1, 0.7, 0.4, 0.8, 0.9], 0.5)
    const two = renderToString(<ContingencyTable table={table} positive={1} classNames={['neg', 'pos']} />)
    for (const name of [
      'TN',
      'FP',
      'FN',
      'TP',
      'sensitivity',
      'specificity',
      'precision',
      'TPR',
      'accuracy',
      'prevalence',
    ])
      expect(two).toContain(name)
    const three = renderToString(
      <ContingencyTable
        table={{
          labels: ['a', 'b', 'c'],
          matrix: [
            [3, 1, 0],
            [0, 2, 1],
            [1, 0, 4],
          ],
        }}
        share="row"
      />,
    )
    expect(three).toContain('recall')
    const costed = renderToString(
      <ContingencyTable
        table={table}
        positive={1}
        costs={[
          [0, 1],
          [10, 0],
        ]}
      />,
    )
    expect(costed).toContain('total cost')
    expect(costed).toContain('cost per case')
    expect(three).not.toContain('TPR')
  })

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
    const plain = renderToString(<Player controller={controller} stamp />)
    expect(plain).toContain('0:00.30')
    expect(plain).toContain('aria-label="Go to the end"')
    expect(plain).not.toContain('aria-label="Eager"')
    const eager = renderToString(<Player controller={{ ...controller, eager: true, setEager: noop }} />)
    expect(eager).toContain('aria-label="Eager"')
    expect(eager).toContain('aria-pressed="true"')
    expect(plain).not.toContain('aria-label="Reset to defaults"')
    expect(renderToString(<Player controller={{ ...controller, resetForm: noop }} />)).toContain(
      'aria-label="Reset to defaults"',
    )
    const projection = linearProjection({ x: [0, 1], y: [0, 1], width: 100, height: 100 })
    const svg = renderToString(
      <InteractionOverlay projection={projection} width={100} height={100}>
        <DraggablePoints points={[[0.5, 0.5]]} onChange={noop} add />
      </InteractionOverlay>,
    )
    expect(svg).toContain('cx="50"')
  })

  it('draws the brush and hover tools', () => {
    const projection = linearProjection({ x: [0, 1], y: [0, 1], width: 100, height: 100 })
    const svg = renderToString(
      <InteractionOverlay projection={projection} width={100} height={100}>
        <Brush value={{ x: [0, 0.5], y: [0, 0.5] }} onChange={noop} />
        <NearestHover points={[[0.5, 0.5]]} onHover={noop} />
      </InteractionOverlay>,
    )
    expect(svg).toContain('width="50"')
  })

  it('draws a structure view', () => {
    const s = structure(
      'tree',
      [
        { id: 'r', label: 'root', role: 'internal' },
        { id: 'l', label: 'leaf', role: 'leaf' },
      ],
      [{ from: 'r', to: 'l' }],
    )
    const html = renderToString(<StructureView structure={s} />)
    expect(html).toContain('data-node-id="r"')
    expect(html).toContain('Zoom in')
  })

  it('draws the threshold explorer: figures, controls and the contingency table at one threshold', () => {
    const y = [0, 0, 1, 0, 1, 1]
    const scores = [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]
    const html = renderToString(
      <ThresholdExplorer
        yTrue={y}
        scores={scores}
        costs={[
          [0, 1],
          [4, 0],
        ]}
        followCheapest
        classNames={['no', 'yes']}
      />,
    )
    expect(html).toContain('follow the cheapest threshold')
    expect(html).toContain('total cost')
    expect(html).toContain('yes')
    const plain = renderToString(<ThresholdExplorer yTrue={y} scores={scores} threshold={0.5} controls={false} />)
    expect(plain).not.toContain('follow the cheapest')
    expect(plain).toContain('TP')
  })
})

describe('registry selectors render on the server', () => {
  it('draws an entry list with a search box, titles and tag chips, the chosen row selected', () => {
    const html = renderToString(
      <EntryList
        label="things"
        rows={[
          { name: 'a', title: 'Thing A', tags: ['real', '10 rows'] },
          { name: 'b', title: 'Thing B', tags: ['synthetic'] },
        ]}
        value="b"
        onChange={noop}
      />,
    )
    expect(html).toContain('role="listbox"')
    expect(html).toContain('type="search"')
    expect(html).toContain('aria-selected="true"')
    expect(html).toContain('Thing A')
    expect(html).toContain('10 rows')
    expect(html.match(/role="option"/g)?.length).toBe(2)
    expect(renderToString(<EntryList label="t" rows={[]} value="" onChange={noop} search={false} />)).not.toContain(
      'type="search"',
    )
  })

  it('draws the dataset selector filtered by its props, with the chosen entry’s tags, provenance and knobs', () => {
    const real = renderToString(
      <DatasetSelector kind="classification" real value={selectEntry(datasetRegistry, 'iris')} onChange={noop} />,
    )
    expect(real).toContain('Iris')
    expect(real).toContain('Wine')
    expect(real).not.toContain('Two moons')
    expect(real).toContain('scikit-learn load_iris')
    expect(real).not.toContain('aria-label="New seed"')
    const synthetic = renderToString(
      <DatasetSelector
        modality={['tabular']}
        real={false}
        value={selectEntry(datasetRegistry, 'moons')}
        onChange={noop}
      />,
    )
    expect(synthetic).toContain('Two moons')
    expect(synthetic).not.toContain('Iris')
    expect(synthetic).toContain('aria-label="New seed"')
    expect(synthetic).toContain('noise')
    expect(
      renderToString(<DatasetSelector form="remote" value={{ name: 'x', seed: 0, knobs: {} }} onChange={noop} />),
    ).toContain('remote')
  })

  it('draws the environment selector with the chosen entry’s spaces and options', () => {
    const html = renderToString(
      <EnvironmentSelector
        family={['classicControl', 'gridWorld']}
        actions="discrete"
        value={selectEnvironment(environmentRegistry, 'frozenLake')}
        onChange={noop}
      />,
    )
    expect(html).toContain('Frozen lake (4 by 4)')
    expect(html).toContain('Cart-pole')
    expect(html).not.toContain('Pendulum')
    expect(html).toContain('FrozenLake-v1')
    expect(html).toContain('Discrete(16)')
    expect(html).toContain('Discrete(4)')
    expect(html).toContain('isSlippery')
    const game = renderToString(
      <EnvironmentSelector multiAgent value={selectEnvironment(environmentRegistry, 'ticTacToe')} onChange={noop} />,
    )
    expect(game).toContain('2 agents')
    expect(game).toContain('observations (player_0)')
  })
})
