import { isExposition } from 'mdml-engine/exposition'
import type { Tensor } from 'mdml-engine/tensor'
import { cartesian } from 'mdml-render/charts'
import { Chart, ExpositionView, FigureFrame, useColourMode } from 'mdml-render/react'
import { metricCurves } from 'mdml-render/trace'
import { type ReactNode, useMemo } from 'react'
import { TapeChannels } from '../session/TapeView.tsx'
import type { LiveSession } from '../session/useSession.ts'
import type { HubEntry, Panel } from './entries.ts'
import { FIG, Note, RowsPanel, useProbe } from './RowsPanel.tsx'

/** What a panel needs of the page: the entry, the session, the scrubbed step and the training data. */
export interface PanelContext {
  readonly entry: HubEntry
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly at: number | null
  readonly x: Tensor
  readonly y?: Tensor
}

const ROWS_TITLES = {
  assignments: 'assignments and centres',
  components: 'components (1σ solid, 2σ dashed)',
  fit: 'the fit',
  surface: 'decision surface',
} as const

const slug = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]+/g, '-')

/** The panels of a layout, in a responsive two-column grid; nothing else is drawn. */
export function Panels(props: PanelContext & { readonly layout: readonly Panel[] }): ReactNode {
  const { layout, ...ctx } = props
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2" data-testid="panels">
      {layout.map((panel) => (
        <div key={JSON.stringify(panel)} className="min-w-0">
          <PanelView panel={panel} {...ctx} />
        </div>
      ))}
    </div>
  )
}

function PanelView(props: PanelContext & { readonly panel: Panel }): ReactNode {
  const { panel, entry, live, at } = props
  const t = at === null ? live.session.t : Math.min(at, live.session.t)
  const file = (title: string): string => `${entry.id.replace('/', '-')}-${slug(title)}`
  switch (panel.kind) {
    case 'rows': {
      const title = panel.title ?? ROWS_TITLES[panel.overlay ?? 'assignments']
      return (
        <RowsPanel
          overlay={panel.overlay}
          title={`${title} at step ${t}`}
          filename={file(title)}
          live={live}
          t={t}
          x={props.x}
          {...(props.y === undefined ? {} : { y: props.y })}
        />
      )
    }
    case 'probe':
      return <ProbeView {...props} panel={panel} t={t} file={file} />
    case 'curves': {
      const title = panel.title ?? panel.paths.join(', ')
      return <CurvesView live={live} paths={panel.paths} title={title} filename={file(title)} />
    }
    case 'projection':
      return <ProjectionView live={live} name={panel.name} />
    case 'tape':
      return <TapeChannels live={live} at={at} />
  }
}

function CurvesView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly paths: readonly string[]
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { live, paths } = props
  const mode = useColourMode()
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const run = useMemo(() => live.session.run(), [live.session, live.version])
  const option = useMemo(
    () => cartesian([metricCurves(run, paths, { mode })], { mode, figure: FIG, x: { label: 'step' }, legend: true }),
    [run, paths, mode],
  )
  return (
    <FigureFrame figure={FIG} title={props.title} filename={props.filename}>
      <Chart figure={FIG} option={option} />
    </FigureFrame>
  )
}

function ProbeView(
  props: PanelContext & {
    readonly panel: Extract<Panel, { kind: 'probe' }>
    readonly t: number
    readonly file: (title: string) => string
  },
): ReactNode {
  const { panel, live, t } = props
  const probed = useProbe(live, t, panel.path, panel.args)
  const title = `${panel.title ?? panel.path} at step ${t}`
  const filename = props.file(panel.title ?? panel.path)
  if (probed.value !== undefined && isExposition(probed.value))
    return <ExpositionView record={probed.value} figure={FIG} />
  const text = probed.value === undefined ? undefined : JSON.stringify(probed.value, null, 1)
  return (
    <FigureFrame
      figure={FIG}
      title={title}
      filename={filename}
      exportFormats={['json']}
      onExport={() => {
        const a = document.createElement('a')
        a.href = URL.createObjectURL(new Blob([text ?? ''], { type: 'application/json' }))
        a.download = `${filename}.json`
        a.click()
      }}
      {...(probed.error === undefined ? {} : { caption: <Note>unavailable: {probed.error}</Note> })}
    >
      <pre className="h-full overflow-auto p-2 font-mono text-xs">{text}</pre>
    </FigureFrame>
  )
}

// A projection of the tape so far; hidden while it has nothing to project.
function ProjectionView(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly name: string
}): ReactNode {
  const { live, name } = props
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const record = useMemo((): unknown => {
    try {
      return live.session.project(name)
    } catch {
      return undefined
    }
  }, [live.session, live.version, name])
  return record !== undefined && isExposition(record) ? <ExpositionView record={record} figure={FIG} /> : null
}
