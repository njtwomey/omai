import { anomalyScores } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals, uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Anomaly scores',
  question:
    'How do I show which points a detector finds unusual, and where the line between normal and anomalous falls?',
  explain:
    'An `anomalyScores` record is a score per row (higher is more unusual), the threshold above which a row is flagged and, optionally, a second value per row as evidence (here the distance from the centre). `AnomalyView` draws the scores as a histogram with the threshold marked and, given the rows, the plane with the flagged rows highlighted. The data is a cloud with a share of scattered outliers; the score is a robust distance from the cloud’s centre, and the threshold the quantile you choose, so contamination and the quantile together decide how many true outliers are caught.',
  shows: ['anomalyScores', 'AnomalyView'],
}

export default function AnomalyScoresRecipe(): ReactNode {
  const [contamination, setContamination] = useState(0.06)
  const [quantile, setQuantile] = useState(0.94)
  const [seed, setSeed] = useState(1)
  const { record, points } = useMemo(() => {
    const n = 400
    const z = normals(seed, 2 * n)
    const u = uniforms(seed + 1, 3 * n)
    const out = Array.from({ length: n }, (_, i) => (u[i] ?? 0) < contamination)
    const pts = out.map((o, i) =>
      o ? [(u[n + i] ?? 0) * 12 - 6, (u[2 * n + i] ?? 0) * 12 - 6] : [z[2 * i] ?? 0, 0.7 * (z[2 * i + 1] ?? 0)],
    )
    const med = (v: number[]): number => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] ?? 0
    const cx = med(pts.map((p) => p[0] ?? 0))
    const cy = med(pts.map((p) => p[1] ?? 0))
    const dist = pts.map((p) => Math.hypot(((p[0] ?? 0) - cx) / 1, ((p[1] ?? 0) - cy) / 0.7))
    const sorted = [...dist].sort((a, b) => a - b)
    const threshold = sorted[Math.min(n - 1, Math.floor(quantile * n))] ?? 1
    return {
      points: pts,
      record: anomalyScores({
        method: 'robust distance',
        scores: dist,
        threshold,
        evidence: { name: 'distance from the centre', values: dist },
      }),
    }
  }, [contamination, quantile, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField
          label="share of outliers"
          value={contamination}
          onChange={setContamination}
          min={0}
          max={0.3}
          step={0.01}
        />
        <NumberField
          label="threshold quantile"
          value={quantile}
          onChange={setQuantile}
          min={0.5}
          max={0.999}
          step={0.01}
        />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} points={points} title="Scores and the flagged rows" />
    </div>
  )
}
