import { type ReactElement, type RefObject, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { TreeRecord } from 'whitebox-engine/exposition'
import { type Figure, scaleFigure } from '../../figure/figure.ts'
import { tokens } from '../../theme/tokens.ts'
import { growthSteps, splitCandidatesView, treeNodeOf, treeScene } from '../../views/trees.ts'
import { Chart } from '../Chart.tsx'
import { FigureFrame, useFrameSize } from '../FigureFrame.tsx'
import { Player } from '../Player.tsx'
import { SceneView } from '../SceneView.tsx'
import { usePlayback } from '../usePlayback.ts'
import type { RecordViewProps } from './_frame.tsx'
import { useViewFigure } from './_frame.tsx'

const TREE_FIGURE = { width: 720, aspect: 1.6 } as const

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

// Cards are never drawn smaller than this fraction of their natural size: a wide tree in a narrow slot scrolls (drag to
// pan) instead of shrinking past legibility.
const MIN_SCALE = 0.7

// The width an element is given by its parent, followed as it changes.
function useMeasuredWidth(fallback: number): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null)
  const [width, setWidth] = useState(fallback)
  useLayoutEffect(() => {
    const el = ref.current
    if (el === null) return
    const measure = (): void =>
      setWidth((w) => (Math.abs(w - el.clientWidth) < 1 || el.clientWidth === 0 ? w : el.clientWidth))
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return [ref, width]
}

// The tree above and the selected node's candidate splits below, at the frame's size (or the narrower width the page
// gives it).
function TreeFigure(props: {
  readonly record: TreeRecord
  readonly figure: Figure
  readonly selected: number
  readonly onSelect: (node: number) => void
  readonly grownTo: number
  readonly maxDepth?: number
  /** The share of the figure's height the tree takes (the candidate splits take the rest). */
  readonly treeShare: number
}): ReactElement {
  const frame = useFrameSize()
  const expandedWidth = frame?.expanded === true ? frame.width : undefined
  const base = useMemo(
    () =>
      expandedWidth === undefined
        ? props.figure
        : { ...scaleFigure(props.figure, expandedWidth), mode: props.figure.mode },
    [props.figure, expandedWidth],
  )
  const [outer, measured] = useMeasuredWidth(base.width)
  const width = Math.min(base.width, measured)
  const treeHeight = Math.round(base.height * props.treeShare)
  const chartHeight = base.height - treeHeight
  const { record, selected, grownTo, maxDepth } = props
  const scene = useMemo(
    () => treeScene(record, { selected, grownTo, ...(maxDepth === undefined ? {} : { maxDepth }) }),
    [record, selected, grownTo, maxDepth],
  )
  // Fit the tree to its area, but no smaller than MIN_SCALE (and no larger than 1.4): beyond that it scrolls.
  const fit = Math.min(width / scene.width, treeHeight / scene.height)
  const scale = Math.min(1.4, Math.max(fit, MIN_SCALE))
  const drawW = Math.round(scene.width * scale)
  const drawH = Math.round(scene.height * scale)
  const scroller = useRef<HTMLDivElement | null>(null)
  // A new tree (or a new width) starts scrolled to its root, which a tidy tree centres.
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new record re-centres the view
  useEffect(() => {
    const el = scroller.current
    if (el !== null) el.scrollLeft = Math.max(0, (drawW - el.clientWidth) / 2)
  }, [record, drawW])
  const drag = useRef<{ x: number; y: number; left: number; top: number; moved: boolean } | null>(null)
  const chartFigure = useMemo(
    () => ({ ...base, width, height: chartHeight, aspect: width / chartHeight }),
    [base, width, chartHeight],
  )
  const option = useMemo(
    () => splitCandidatesView(record, selected, { figure: chartFigure }),
    [record, selected, chartFigure],
  )
  const overflows = drawW > width + 1 || drawH > treeHeight + 1
  return (
    <div
      ref={outer}
      style={{ width: '100%', maxWidth: base.width, height: base.height, display: 'flex', flexDirection: 'column' }}
    >
      <div
        ref={scroller}
        data-whitebox-svg=""
        style={{
          height: treeHeight,
          width: '100%',
          overflow: overflows ? 'auto' : 'hidden',
          cursor: overflows ? 'grab' : 'pointer',
          display: 'flex',
          alignItems: overflows ? 'flex-start' : 'center',
        }}
        onPointerDown={(e) => {
          const el = scroller.current
          if (!overflows || el === null) return
          drag.current = { x: e.clientX, y: e.clientY, left: el.scrollLeft, top: el.scrollTop, moved: false }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          const el = scroller.current
          if (d === null || el === null) return
          const dx = e.clientX - d.x
          const dy = e.clientY - d.y
          if (Math.abs(dx) + Math.abs(dy) > 4) d.moved = true
          el.scrollLeft = d.left - dx
          el.scrollTop = d.top - dy
        }}
        onPointerUp={() => {
          // Keep the flag until the click that follows a drag has been seen.
          setTimeout(() => {
            drag.current = null
          }, 0)
        }}
        onPointerLeave={() => {
          drag.current = null
        }}
      >
        {overflows && (
          <span
            style={{
              // Sticky and zero-width: it stays in the corner as the tree scrolls and takes no room.
              position: 'sticky',
              left: 8,
              top: 4,
              width: 0,
              whiteSpace: 'nowrap',
              alignSelf: 'flex-start',
              font: `11px ${tokens(base.mode).font}`,
              color: tokens(base.mode).muted,
              pointerEvents: 'none',
              zIndex: 1,
            }}
          >
            drag to pan
          </span>
        )}
        <div style={{ width: drawW, height: drawH, margin: '0 auto', flexShrink: 0 }}>
          <SceneView
            scene={scene}
            width={drawW}
            height={drawH}
            mode={base.mode}
            onNodeClick={(id) => {
              if (drag.current?.moved === true) return
              const node = treeNodeOf(id)
              if (node !== undefined) props.onSelect(node)
            }}
          />
        </div>
      </div>
      <Chart option={option} width={width} height={chartHeight} mode={base.mode} />
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
  const playback = usePlayback(steps, { fps: 10 })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // A new tree opens fully grown, with its root selected.
  useEffect(() => {
    seek.current(steps - 1)
    setOwn(root)
  }, [steps, root])
  const growth = props.growth ?? true
  // The figure keeps its width but grows tall enough for the whole tree at a legible scale (fit to the width, never
  // below MIN_SCALE), with room for the candidate splits beneath; a tree too wide for that scrolls sideways.
  const full = useMemo(
    () => treeScene(record, props.maxDepth === undefined ? {} : { maxDepth: props.maxDepth }),
    [record, props.maxDepth],
  )
  const { sized, treeShare } = useMemo(() => {
    const chartH = Math.max(180, Math.round(figure.height * 0.34))
    const scale = Math.min(1.2, Math.max(MIN_SCALE, figure.width / full.width))
    const treeH = Math.max(figure.height - chartH, Math.round(full.height * scale) + 8)
    const height = treeH + chartH
    return { sized: { ...figure, height, aspect: figure.width / height }, treeShare: treeH / height }
  }, [figure, full])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: figure.width, maxWidth: '100%' }}>
      <FigureFrame
        figure={sized}
        filename={props.filename ?? 'tree'}
        {...(props.title === undefined ? {} : { title: props.title })}
        {...(props.caption === undefined ? {} : { caption: props.caption })}
      >
        <TreeFigure
          record={record}
          figure={sized}
          treeShare={treeShare}
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
