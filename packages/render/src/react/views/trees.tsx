import { type ReactElement, useEffect, useMemo, useRef, useState } from 'react'
import type { TreeRecord } from 'whitebox-engine/exposition'
import { type Figure, scaleFigure } from '../../figure/figure.ts'
import { growthSteps, splitCandidatesView, treeNodeOf, treeScene } from '../../views/trees.ts'
import { Chart } from '../Chart.tsx'
import { FigureFrame, useFrameSize } from '../FigureFrame.tsx'
import { Player } from '../Player.tsx'
import { SceneView } from '../SceneView.tsx'
import { usePlayback } from '../usePlayback.ts'
import type { RecordViewProps } from './_frame.tsx'
import { useViewFigure } from './_frame.tsx'

const TREE_FIGURE = { width: 720, aspect: 1.15 } as const

/** Props of {@link TreeView}. */
export interface TreeViewProps extends RecordViewProps<TreeRecord> {
  /** The deepest level drawn (default all). */
  readonly maxDepth?: number
  /** The node whose candidate splits are shown (default the root; clicking a node selects it). */
  readonly selected?: number
  readonly onSelect?: (node: number) => void
  /** Offer a player over the order in which the tree grew (default true). */
  readonly growth?: boolean
}

// The tree above and the selected node's candidate splits below, at the frame's size.
function TreeFigure(props: {
  readonly record: TreeRecord
  readonly figure: Figure
  readonly selected: number
  readonly onSelect: (node: number) => void
  readonly grownTo: number
  readonly maxDepth?: number
}): ReactElement {
  const frame = useFrameSize()
  const fig =
    frame?.expanded === true ? { ...scaleFigure(props.figure, frame.width), mode: props.figure.mode } : props.figure
  const chartHeight = Math.round(fig.height * 0.34)
  const treeHeight = fig.height - chartHeight
  const { record, selected, grownTo, maxDepth } = props
  const scene = useMemo(
    () => treeScene(record, { selected, grownTo, ...(maxDepth === undefined ? {} : { maxDepth }) }),
    [record, selected, grownTo, maxDepth],
  )
  const chartFigure = useMemo(
    () => ({ ...fig, height: chartHeight, aspect: fig.width / chartHeight }),
    [fig, chartHeight],
  )
  const option = useMemo(
    () => splitCandidatesView(record, selected, { figure: chartFigure }),
    [record, selected, chartFigure],
  )
  return (
    <div style={{ width: fig.width, height: fig.height, display: 'flex', flexDirection: 'column' }}>
      <div
        data-whitebox-svg=""
        style={{ height: treeHeight, display: 'flex', justifyContent: 'center', cursor: 'pointer' }}
      >
        <SceneView
          scene={scene}
          width={fig.width}
          height={treeHeight}
          mode={fig.mode}
          onNodeClick={(id) => {
            const node = treeNodeOf(id)
            if (node !== undefined) props.onSelect(node)
          }}
        />
      </div>
      <Chart option={option} width={fig.width} height={chartHeight} mode={fig.mode} />
    </div>
  )
}

/**
 * A grown tree (CART, ID3, C4.5, a tree inside an ensemble): each node with its split, class proportions, row count and
 * impurity; click a node to see every candidate split it evaluated (impurity decrease against threshold, per feature),
 * and play the order in which the tree grew.
 *
 * @param props - The record, depth limit, selection and growth player, and the figure and frame settings.
 * @returns The figure with its player.
 */
export function TreeView(props: TreeViewProps): ReactElement {
  const figure = useViewFigure(props.figure, TREE_FIGURE)
  const { record } = props
  const root = record.nodes.find((n) => n.parent === null)?.id ?? 0
  const [own, setOwn] = useState(root)
  const selected = props.selected ?? own
  const select = (node: number): void => {
    setOwn(node)
    props.onSelect?.(node)
  }
  const steps = growthSteps(record)
  const playback = usePlayback(steps, { fps: Math.min(12, Math.max(2, Math.round(steps / 8))) })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // A new tree opens fully grown, with its root selected.
  useEffect(() => {
    seek.current(steps - 1)
    setOwn(root)
  }, [steps, root])
  const growth = props.growth ?? true
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: figure.width, maxWidth: '100%' }}>
      <FigureFrame
        figure={figure}
        filename={props.filename ?? 'tree'}
        {...(props.title === undefined ? {} : { title: props.title })}
        {...(props.caption === undefined ? {} : { caption: props.caption })}
      >
        <TreeFigure
          record={record}
          figure={figure}
          selected={selected}
          onSelect={select}
          grownTo={growth ? playback.frame : steps}
          {...(props.maxDepth === undefined ? {} : { maxDepth: props.maxDepth })}
        />
      </FigureFrame>
      {growth && steps > 1 && <Player controller={playback} stamp={(f) => `grown ${f + 1}/${steps} nodes`} />}
    </div>
  )
}
