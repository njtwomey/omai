import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { tensor, toArray, type Value } from '../../tensor/index.ts'
import * as L from './index.ts'

// Matrices at the awkward places of the factorisations (rank deficiency, ill conditioning, clustered or repeated
// eigenvalues, the edges of the double range), checked against identities that need no reference (A = QR, orthogonality,
// residuals, the Penrose conditions) and against numpy / scipy values (numpy 2.5, scipy 1.18, one-off runs).

type Mat = number[][]
const mat = (v: Value): Mat => toArray(v) as Mat
const vec = (v: Value): number[] => toArray(v) as number[]
const num = (v: Value): number => toArray(v) as unknown as number
const t = (a: Mat): Mat => (a[0] ?? []).map((_, j) => a.map((row) => row[j] ?? 0))
const mm = (a: Mat, b: Mat): Mat =>
  a.map((row) => (b[0] ?? []).map((_, j) => row.reduce((s, x, k) => s + x * (b[k]?.[j] ?? 0), 0)))
const sub = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((x, j) => x - (b[i]?.[j] ?? 0)))
const scale = (a: Mat, s: number): Mat => a.map((row) => row.map((x) => x * s))
const eye = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
const upper = (a: Mat): Mat => a.map((row, i) => row.map((x, j) => (j < i ? 0 : x)))
const lower = (a: Mat): Mat => a.map((row, i) => row.map((x, j) => (j > i ? 0 : x)))
/** Frobenius norm that neither overflows nor underflows (the entries are divided by the largest first). */
const fro = (a: Mat): number => {
  const flat = a.flat()
  const big = Math.max(0, ...flat.map(Math.abs))
  return big === 0 ? 0 : big * Math.sqrt(flat.reduce((s, x) => s + (x / big) ** 2, 0))
}
const hilbert = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => 1 / (i + j + 1)))
const rank1 = [
  [1, -1, 2],
  [2, -2, 4],
  [3, -3, 6],
]
const seq = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
]
const base4 = [
  [3, 1, -2, 0.5],
  [1, 4, 0.3, -1],
  [-2, 0.3, 5, 1],
  [0.5, -1, 1, 2.5],
]
const tall = [
  [1, 2],
  [2, 4],
  [3, 6],
  [1, 2],
]
const wide = [
  [1, 0, 1, 0],
  [0, 1, 0, 1],
]
const spd3 = [
  [4, 1, 2],
  [1, 3, 0],
  [2, 0, 5],
]

// [name, matrix, rank by numpy.linalg.matrix_rank]
const shapes: readonly (readonly [string, Mat, number])[] = [
  ['1 x 1', [[-2.5]], 1],
  [
    'tall, full rank',
    [
      [1, 2],
      [0.3, -1],
      [2, 0.5],
      [-0.7, 1.4],
      [1, 1],
    ],
    2,
  ],
  ['wide, full rank', wide, 2],
  ['tall, rank 1', tall, 1],
  ['3 x 3, rank 1', rank1, 1],
  ['3 x 3, rank 2 (1 to 9)', seq, 2],
  [
    'zero 3 x 3',
    [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    0,
  ],
  ['Hilbert 6 (cond 1.5e7)', hilbert(6), 6],
  ['Hilbert 12 (numpy rank 11)', hilbert(12), 11],
  ['identity 4', eye(4), 4],
  ['4 x 4 at scale 1e150', scale(base4, 1e150), 4],
  ['4 x 4 at scale 1e-150', scale(base4, 1e-150), 4],
  ['4 x 4 at scale 1e-300', scale(base4, 1e-300), 4],
  ['4 x 4 at scale 1e300', scale(base4, 1e300), 4],
]

describe('factorisations on rank-deficient, ill-conditioned and extreme-scale matrices', () => {
  it('QR, SVD, pinv, nullSpace and matrixRank satisfy their identities and numpy on 14 awkward matrices', () => {
    for (const [name, a, rank] of shapes) {
      const size = fro(a)
      const m = a.length
      const n = a[0]?.length ?? 0
      const k = Math.min(m, n)
      // A = QR with orthonormal Q and upper-triangular R.
      const { q, r } = L.qr(a)
      expect(fro(sub(mm(mat(q), mat(r)), a)), `${name}: A - QR`).toBeLessThanOrEqual(1e-14 * size * m)
      expect(fro(sub(mm(t(mat(q)), mat(q)), eye(k))), `${name}: Q'Q - I`).toBeLessThan(1e-13)
      expect(fro(sub(mat(r), upper(mat(r)))), `${name}: below the diagonal of R`).toBe(0)
      // A = U diag(s) Vh with orthonormal U and Vh, s non-negative and descending.
      const { u, s, vh } = L.svd(a)
      const [ua, sa, va] = [mat(u), vec(s), mat(vh)]
      const us = ua.map((row) => row.map((x, j) => x * (sa[j] ?? 0)))
      expect(fro(sub(mm(us, va), a)), `${name}: A - U S Vh`).toBeLessThanOrEqual(1e-14 * size * m)
      expect(fro(sub(mm(t(ua), ua), eye(k))), `${name}: U'U - I`).toBeLessThan(1e-13)
      expect(fro(sub(mm(va, t(va)), eye(k))), `${name}: Vh Vh' - I`).toBeLessThan(1e-13)
      expect(
        sa.every((x, i) => x >= 0 && (i === 0 || x <= (sa[i - 1] ?? 0))),
        `${name}: s sorted`,
      ).toBe(true)
      expect(L.matrixRank(a), `${name}: rank`).toBe(rank)
      // The four Penrose conditions of the pseudo-inverse (A+ has the scale of 1 / A, so they are relative).
      const p = mat(L.pinv(a))
      const [ap, pa] = [mm(a, p), mm(p, a)]
      // Hilbert 12 keeps singular values down to 3e-14, so its pseudo-inverse carries errors of cond * eps (numpy's too).
      const loose = name.startsWith('Hilbert 12') ? 1e7 : 1
      expect(fro(sub(mm(ap, a), a)), `${name}: A A+ A = A`).toBeLessThanOrEqual(1e-9 * size * loose)
      expect(fro(sub(mm(pa, p), p)), `${name}: A+ A A+ = A+`).toBeLessThanOrEqual(1e-9 * fro(p) * loose)
      expect(fro(sub(ap, t(ap))), `${name}: A A+ symmetric`).toBeLessThanOrEqual(1e-7 * loose)
      expect(fro(sub(pa, t(pa))), `${name}: A+ A symmetric`).toBeLessThanOrEqual(1e-7 * loose)
      // The null space: n - rank orthonormal columns that A maps to zero.
      const nul = mat(L.nullSpace(a))
      const cols = nul[0]?.length ?? 0
      expect(cols, `${name}: nullity`).toBe(n - rank)
      if (cols > 0) {
        expect(fro(mm(a, nul)), `${name}: A N = 0`).toBeLessThanOrEqual(1e-12 * size)
        expect(fro(sub(mm(t(nul), nul), eye(cols))), `${name}: N'N - I`).toBeLessThan(1e-12)
      }
    }
    {
      // [matrix, numpy.linalg.pinv]: 3 x 3 of rank 1 and rank 2, a tall one of rank 1, a wide one.
      const rows: readonly (readonly [string, Mat, Mat])[] = [
        [
          'rank 1, 3 x 3',
          rank1,
          [
            [0.011904761904761904, 0.023809523809523815, 0.035714285714285726],
            [-0.011904761904761899, -0.023809523809523805, -0.03571428571428571],
            [0.023809523809523805, 0.04761904761904762, 0.07142857142857144],
          ],
        ],
        [
          'rank 2, 3 x 3',
          seq,
          [
            [-0.638888888888889, -0.16666666666666632, 0.3055555555555554],
            [-0.05555555555555514, 0, 0.055555555555555365],
            [0.5277777777777772, 0.16666666666666624, -0.1944444444444441],
          ],
        ],
        [
          'rank 1, tall',
          tall,
          [
            [0.01333333333333334, 0.026666666666666658, 0.04, 0.013333333333333329],
            [0.026666666666666686, 0.05333333333333333, 0.08, 0.026666666666666665],
          ],
        ],
        [
          'wide',
          wide,
          [
            [0.5, 0],
            [0, 0.5],
            [0.5, 0],
            [0, 0.5],
          ],
        ],
      ]
      for (const [name, a, expected] of rows) {
        expect(mat(L.pinv(a)).flat(), name).toEqual(expected.flat().map((e) => expect.closeTo(e, 12)))
      }
    }
  })
})

describe('linear systems, determinants and the other factorisations at their limits', () => {
  // The exact inverse of the Hilbert matrix of order 6 is integral (scipy.linalg.invhilbert(6, exact=True)).
  const hilbertInverse6: Mat = [
    [36, -630, 3360, -7560, 7560, -2772],
    [-630, 14700, -88200, 211680, -220500, 83160],
    [3360, -88200, 564480, -1411200, 1512000, -582120],
    [-7560, 211680, -1411200, 3628800, -3969000, 1552320],
    [7560, -220500, 1512000, -3969000, 4410000, -1746360],
    [-2772, 83160, -582120, 1552320, -1746360, 698544],
  ]

  it('solve, inv, det, cond, norm, LU and Cholesky hold at ill-conditioning and at the edges of the double range', () => {
    {
      const h = hilbert(6)
      const hi = mat(L.inv(h))
      // The error of the inverse is bounded by cond * eps = 1.5e7 * 2.2e-16 = 3e-9 relative to its norm.
      expect(fro(sub(hi, hilbertInverse6)) / fro(hilbertInverse6)).toBeLessThan(1e-8)
      expect(fro(sub(mm(h, hi), eye(6)))).toBeLessThan(1e-6)
      const b = [[1], [2], [3], [4], [5], [6]]
      const x = mat(L.solve(h, b))
      const exact = mm(hilbertInverse6, b)
      expect(fro(sub(x, exact)) / fro(exact)).toBeLessThan(1e-8)
      expect(fro(sub(mm(h, x), b))).toBeLessThan(1e-9)
      const { sign, logabsdet } = L.slogdet(h)
      expect([num(sign), num(logabsdet)]).toEqual([1, expect.closeTo(-39.766206706136984, 6)])
      // numpy.linalg.cond(hilbert(6), p) for p = 2, 1, inf, 'fro'.
      for (const [p, expected] of [
        [2, 14951058.640168244],
        [1, 29070279.006218337],
        [Number.POSITIVE_INFINITY, 29070279.006220795],
        ['fro', 15118987.129620317],
      ] as const) {
        expect(Math.abs(num(L.cond(h, p)) / expected - 1), `cond ${p}`).toBeLessThan(1e-6)
      }
      // det(c I_3) = c^3 overflows and underflows as it should, and slogdet keeps the logarithm.
      for (const [c, det] of [
        [1e100, 1e300],
        [1e-100, 1e-300],
        [1e200, Number.POSITIVE_INFINITY],
        [1e-200, 0],
      ] as const) {
        const d = num(L.det(scale(eye(3), c)))
        if (det > 0 && Number.isFinite(det)) expect(Math.abs(d / det - 1), `det(${c} I)`).toBeLessThan(1e-12)
        else expect(d, `det(${c} I)`).toBe(det)
        expect(num(L.slogdet(scale(eye(3), c)).logabsdet), `log |det(${c} I)|`).toBeCloseTo(3 * Math.log(c), 9)
      }
      // A system at scale s has the solution b / s, and norm and cond follow the scale (numpy values of the unit matrix).
      const unit = mat(L.solve(spd3, [[1], [2], [3]]))
      expect(unit.flat()).toEqual(
        [-0.30232558139534893, 0.7674418604651164, 0.7209302325581396].map((e) => expect.closeTo(e, 12)),
      )
      for (const s of [1e-300, 1e-150, 1e150, 1e300]) {
        const xs = mat(L.solve(scale(spd3, s), [[1], [2], [3]]))
        expect(
          xs.flat().map((v) => v * s),
          `solve at scale ${s}`,
        ).toEqual(unit.flat().map((e) => expect.closeTo(e, 12)))
        const a = scale(spd3, s)
        // numpy.linalg.norm(A0, ord): 1 and inf 7, 2 6.669079088282289, 'fro' 7.745966692414834, 'nuc' 12.
        for (const [ord, expected] of [
          [1, 7],
          [Number.POSITIVE_INFINITY, 7],
          [2, 6.669079088282289],
          ['fro', 7.745966692414834],
          ['nuc', 12],
        ] as const) {
          expect(num(L.norm(a, ord)) / s / expected - 1, `norm ${ord} at scale ${s}`).toBeCloseTo(0, 12)
        }
        expect(num(L.cond(a)) / num(L.cond(spd3)) - 1, `cond at scale ${s}`).toBeCloseTo(0, 9)
        // The Euclidean norm of a vector: 3-4-5 at scale s (squares leave the double range beyond about 1e154).
        expect(num(L.norm([3 * s, 4 * s])) / s, `vector norm at scale ${s}`).toBeCloseTo(5, 12)
        // Its gradient is the unit vector, scaled path included.
        expect(vec(grad((x: Value) => L.norm(x))(tensor([3 * s, 4 * s]))), `grad of the norm at scale ${s}`).toEqual([
          expect.closeTo(0.6, 12),
          expect.closeTo(0.8, 12),
        ])
      }
    }
    {
      for (const [name, a] of [
        ['SPD 3 x 3', spd3],
        ['Hilbert 6', hilbert(6)],
        ['SPD at scale 1e-150', scale(spd3, 1e-150)],
        ['SPD at scale 1e150', scale(spd3, 1e150)],
        ['SPD at scale 1e300', scale(spd3, 1e300)],
      ] as const) {
        const size = fro(a)
        const l = mat(L.cholesky(a))
        expect(fro(sub(mm(l, t(l)), a)), `${name}: L L' - A`).toBeLessThanOrEqual(1e-14 * size)
        expect(fro(sub(l, lower(l))), `${name}: L lower`).toBe(0)
        expect(
          l.every((row, i) => (row[i] ?? 0) > 0),
          `${name}: positive diagonal`,
        ).toBe(true)
        const { p, l: lo, u } = L.lu(a)
        expect(fro(sub(mm(mm(mat(p), mat(lo)), mat(u)), a)), `${name}: P L U - A`).toBeLessThanOrEqual(1e-14 * size)
        expect(
          mat(lo).every((row, i) =>
            row.every((x, j) => (j > i ? x === 0 : j === i ? x === 1 : Math.abs(x) <= 1 + 1e-14)),
          ),
          `${name}: L is unit lower with |l| <= 1 (partial pivoting)`,
        ).toBe(true)
      }
      // numpy.linalg.cholesky of Hilbert 6, last row: the pivots shrink to 1.2e-3.
      const lh = mat(L.cholesky(hilbert(6)))
      expect([lh[5]?.[4], lh[5]?.[5]]).toEqual([
        expect.closeTo(0.0119047619047679, 10),
        expect.closeTo(0.0011964735895672073, 10),
      ])
      // numpy.linalg.cholesky of [[4, 1, 2], [1, 3, 0], [2, 0, 5]] and the LU of the same.
      expect(mat(L.cholesky(spd3)).flat()).toEqual(
        [2, 0, 0, 0.5, 1.6583123951777, 0, 1, -0.30151134457776363, 1.9771421064483223].map((e) =>
          expect.closeTo(e, 12),
        ),
      )
      for (const a of [
        [
          [1, 2],
          [2, 1],
        ],
        [
          [0, 0],
          [0, 1],
        ],
        [[-1]],
        [
          [1, 2],
          [3, 4],
        ],
      ]) {
        expect(() => L.cholesky(a), JSON.stringify(a)).toThrow(expect.objectContaining({ code: 'value' }))
      }
    }
  })
})

describe('eigh, expm, sqrtm and logm against scipy on clustered, defective and large-norm inputs', () => {
  it('eigh, expm, sqrtm and logm match scipy, NaN input is never answered with a finite result, and singular input is reported', () => {
    {
      const wilkinson = Array.from({ length: 21 }, (_, i) =>
        Array.from({ length: 21 }, (_, j) => (i === j ? Math.abs(i - 10) : Math.abs(i - j) === 1 ? 1 : 0)),
      )
      // [name, matrix, numpy.linalg.eigvalsh or undefined]
      const rows: readonly (readonly [string, Mat, readonly number[] | undefined])[] = [
        ['1 x 1', [[-3]], [-3]],
        ['identity 5', eye(5), [1, 1, 1, 1, 1]],
        [
          'all ones 4 x 4 (eigenvalue 0 three times)',
          [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
          ],
          [0, 0, 0, 4],
        ],
        [
          'diagonal spread 1e-10 to 1e10',
          [
            [1e10, 0, 0],
            [0, 1, 0],
            [0, 0, 1e-10],
          ],
          [1e-10, 1, 1e10],
        ],
        [
          'Wilkinson W21+ (pairs of eigenvalues equal to 1e-13)',
          wilkinson,
          [
            -1.1254415221199832, 0.2538058170966795, 0.9475343675292947, 1.7893213526950813, 2.130209219362507,
            2.961058884185728, 3.0430992925788245, 3.9960482013836267, 4.0043540234408574, 4.999782477742902,
            5.000244425001914, 6.000217522257097, 6.000234031584165, 7.003951798616375, 7.003952209528674,
            8.038941115814275, 8.038941122829023, 9.21067864730492, 9.210678647361332, 10.746194182903322,
            10.746194182903395,
          ],
        ],
        [
          'SPD 4 x 4 at scale 1e150',
          scale(
            base4.map((r, i) => r.map((x, j) => (x + (base4[j]?.[i] ?? 0)) / 2)),
            1e150,
          ),
          undefined,
        ],
        [
          'SPD 4 x 4 at scale 1e-150',
          scale(
            base4.map((r, i) => r.map((x, j) => (x + (base4[j]?.[i] ?? 0)) / 2)),
            1e-150,
          ),
          undefined,
        ],
        ['Hilbert 8', hilbert(8), undefined],
      ]
      for (const [name, a, expected] of rows) {
        const size = fro(a)
        const { eigenvalues, eigenvectors } = L.eigh(a)
        const [w, v] = [vec(eigenvalues), mat(eigenvectors)]
        const n = a.length
        const vw = v.map((row) => row.map((x, j) => x * (w[j] ?? 0)))
        expect(fro(sub(mm(a, v), vw)), `${name}: A V - V W`).toBeLessThanOrEqual(1e-13 * size * n)
        expect(fro(sub(mm(t(v), v), eye(n))), `${name}: V'V - I`).toBeLessThan(1e-13)
        expect(
          w.every((x, i) => i === 0 || x >= (w[i - 1] ?? 0)),
          `${name}: ascending`,
        ).toBe(true)
        // The spectrum of the symmetric matrix: absolute accuracy eps times its largest eigenvalue.
        const close = (got: readonly number[], want: readonly number[]): void => {
          for (const [i, g] of got.entries()) {
            expect(Math.abs(g - (want[i] ?? 0)), `${name}: w[${i}]`).toBeLessThanOrEqual(1e-14 * size * n)
          }
        }
        if (expected) close(w, expected)
        close(vec(L.eigvalsh(a)), w)
      }
    }
    {
      const big4 = [
        [30, -20, 10, 5],
        [-25, 12, 8, -10],
        [4, -35, 18, 20],
        [9, 15, -22, 31],
      ]
      // [name, matrix, scipy.linalg.expm]; "big4" has norm 1 about 100, so scaling and squaring takes 5 squarings.
      const exps: readonly (readonly [string, Mat, Mat])[] = [
        [
          '[[1, 2], [3, 4]]',
          [
            [1, 2],
            [3, 4],
          ],
          [
            [51.96895619870756, 74.73656456700694],
            [112.10484685051044, 164.07380304921801],
          ],
        ],
        [
          'rotation by 1000 (eight squarings)',
          [
            [0, 1000],
            [-1000, 0],
          ],
          [
            [Math.cos(1000), Math.sin(1000)],
            [-Math.sin(1000), Math.cos(1000)],
          ],
        ],
        [
          'diag(-30, 30)',
          [
            [-30, 0],
            [0, 30],
          ],
          [
            [9.357622968840175e-14, 0],
            [0, 10686474581524.463],
          ],
        ],
        [
          'nilpotent of norm 1e6',
          [
            [0, 1e6],
            [0, 0],
          ],
          [
            [1, 1e6],
            [0, 1],
          ],
        ],
        [
          '[[1e-20, 0], [0, 0]]',
          [
            [1e-20, 0],
            [0, 0],
          ],
          [
            [1, 0],
            [0, 1],
          ],
        ],
        [
          '[[-50, 50], [0, -50]]',
          [
            [-50, 50],
            [0, -50],
          ],
          [
            [1.9287498479639178e-22, 9.643749239819589e-21],
            [0, 1.9287498479639178e-22],
          ],
        ],
        [
          '[[1, 1e5], [0, 1]] (non-normal)',
          [
            [1, 1e5],
            [0, 1],
          ],
          [
            [Math.E, 1e5 * Math.E],
            [0, Math.E],
          ],
        ],
        [
          '4 x 4 of norm 100',
          big4,
          [
            [5.5488999043737e17, -1.4587897924623552e17, -7.83576575414034e16, 2.7945758380530074e17],
            [-2.914002979891992e17, 7.660829852702085e16, 4.114949837101197e16, -1.4675705923720202e17],
            [1.9989048875751382e17, -5.2550633371699304e16, -2.8227127412779956e16, 1.0067024812837573e17],
            [-3.956497484242271e17, 1.040151785941763e17, 5.58708717464155e16, -1.9925989768027133e17],
          ],
        ],
      ]
      for (const [name, a, expected] of exps) {
        const got = mat(L.expm(a))
        expect(fro(sub(got, expected)) / fro(expected), `expm ${name}`).toBeLessThan(1e-9)
      }
      // [matrix, scipy.linalg.sqrtm]
      const roots: readonly (readonly [string, Mat, Mat])[] = [
        [
          'SPD',
          [
            [4, 1],
            [1, 3],
          ],
          [
            [1.9815776331353154, 0.2708322060941793],
            [0.2708322060941793, 1.7107454270411362],
          ],
        ],
        [
          'Jordan block',
          [
            [1, 1],
            [0, 1],
          ],
          [
            [1, 0.5],
            [0, 1],
          ],
        ],
        [
          'eigenvalues 2 and 1e-8',
          [
            [2, 0],
            [0, 1e-8],
          ],
          [
            [Math.SQRT2, 0],
            [0, 1e-4],
          ],
        ],
        ['1 x 1', [[4]], [[2]]],
        [
          'non-symmetric 3 x 3',
          [
            [5, 2, 1],
            [0, 3, 1],
            [1, 0, 2],
          ],
          [
            [2.22429884524271, 0.5071789203742318, 0.23127933432154874],
            [-0.022310125865567527, 1.7394300507340468, 0.3205198377838193],
            [0.27589958605268367, -0.04462025173113419, 1.3966000870846602],
          ],
        ],
      ]
      for (const [name, a, expected] of roots) {
        const got = mat(L.sqrtm(a))
        expect(fro(sub(got, expected)) / fro(expected), `sqrtm ${name}`).toBeLessThan(1e-9)
        expect(fro(sub(mm(got, got), a)) / fro(a), `sqrtm ${name} squared`).toBeLessThan(1e-9)
      }
      // scipy.linalg.logm([[4, 1], [1, 3]]) and logm of a unipotent Jordan block (log(I + N) = N).
      expect(
        mat(
          L.logm([
            [4, 1],
            [1, 3],
          ]),
        ).flat(),
      ).toEqual(
        [1.346984922338319, 0.29607457187826797, 0.29607457187826786, 1.0509103504600512].map((e) =>
          expect.closeTo(e, 10),
        ),
      )
      expect(
        mat(
          L.logm([
            [1, 1],
            [0, 1],
          ]),
        ).flat(),
      ).toEqual([0, 1, 0, 0].map((e) => expect.closeTo(e, 8)))
    }
    {
      const nan = [
        [Number.NaN, 1],
        [1, 1],
      ]
      const nonFinite = (call: () => Value | number[] | number): boolean | 'throws' => {
        try {
          const r = call()
          const flat = (typeof r === 'number' ? [r] : Array.isArray(r) ? r : toArray(r as Value)) as unknown
          return [flat].flat(Number.POSITIVE_INFINITY as 1).some((x) => typeof x === 'number' && !Number.isFinite(x))
        } catch {
          return 'throws'
        }
      }
      // The NaN rule (ND-11): a routine that decides or summarises rejects a non-finite entry at its boundary with a
      // `value` error naming itself, for NaN and for either infinity; before, NaN gave garbage (`nullSpace` of a NaN
      // matrix was the identity, `matrixRank` 0, `eigh` threw `convergence`, `cholesky` blamed positive definiteness).
      const inf = [
        [Number.NEGATIVE_INFINITY, 1],
        [1, 1],
      ]
      const rejects: readonly (readonly [string, (m: number[][]) => unknown])[] = [
        ['svd', (m) => L.svd(m)],
        ['svdvals', (m) => L.svdvals(m)],
        ['qr', (m) => L.qr(m)],
        ['lu', (m) => L.lu(m)],
        ['luFactor', (m) => L.luFactor(m)],
        ['cholesky', (m) => L.cholesky(m)],
        ['choleskyJitter', (m) => L.choleskyJitter(m)],
        ['eigh', (m) => L.eigh(m)],
        ['eigh', (m) => L.eigh(m, { method: 'jacobi' })],
        ['eigvalsh', (m) => L.eigvalsh(m)],
        ['solve', (m) => L.solve(m, [1, 2])],
        [
          'solve',
          (m) =>
            L.solve(
              [
                [2, 0],
                [0, 2],
              ],
              m[0] as number[],
            ),
        ],
        ['inv', (m) => L.inv(m)],
        ['pinv', (m) => L.pinv(m)],
        ['lstsq', (m) => L.lstsq(m, [1, 2])],
        [
          'lstsq',
          (m) =>
            L.lstsq(
              [
                [1, 0],
                [0, 1],
              ],
              m[0] as number[],
            ),
        ],
        ['nullSpace', (m) => L.nullSpace(m)],
        ['matrixRank', (m) => L.matrixRank(m)],
        ['cond', (m) => L.cond(m)],
        ['polar', (m) => L.polar(m)],
        ['lowRankApproximation', (m) => L.lowRankApproximation(m, 1)],
      ]
      for (const [name, call] of rejects) {
        for (const m of [nan, inf]) {
          expect(() => call(m), `${name} of ${m[0]?.[0]}`).toThrow(
            expect.objectContaining({
              code: 'value',
              message: expect.stringMatching(new RegExp(`^${name}: .*non-finite`)),
            }),
          )
        }
      }
      // What only propagates (elementwise, or a summary with a defined non-finite value): not rejected.
      expect(L.isPositiveDefinite(nan)).toBe(false)
      // The norm of infinite entries is Infinity (the scaled sum made it NaN, which hid an overflow from fsolve's line search)
      // and a NaN entry still gives NaN.
      expect([
        L.dnrm2([1, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]),
        L.dnrm2([Number.POSITIVE_INFINITY]),
      ]).toEqual([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY])
      expect(L.dnrm2([Number.POSITIVE_INFINITY, Number.NaN])).toBeNaN()
      const calls: readonly (readonly [string, () => Value | number[] | number])[] = [
        ['det', () => L.det(nan)],
        ['slogdet', () => L.slogdet(nan).logabsdet],
        ['expm', () => L.expm(nan)],
        [
          'expm of an infinite matrix (the squaring count was infinite)',
          () =>
            L.expm([
              [Number.POSITIVE_INFINITY, 1],
              [1, 1],
            ]),
        ],
        ['sqrtm', () => L.sqrtm(nan)],
        ['norm', () => L.norm(nan)],
      ]
      for (const [name, call] of calls) expect(nonFinite(call), `${name} of a NaN matrix`).not.toBe(false)
      // numpy raises LinAlgError for these; so do we, with a value error naming the cause.
      const singular = [
        [1, 2],
        [2, 4],
      ]
      for (const [name, call] of [
        ['solve', () => L.solve(singular, [1, 2])],
        ['inv', () => L.inv(singular)],
        [
          'cholesky of an indefinite matrix',
          () =>
            L.cholesky([
              [1, 2],
              [2, 1],
            ]),
        ],
      ] as const) {
        expect(call, name).toThrow(expect.objectContaining({ code: 'value' }))
      }
      // det of a singular matrix is exactly 0, slogdet has sign 0 and log -inf, and the LU of it has a zero pivot.
      expect(num(L.det(singular))).toBe(0)
      const sl = L.slogdet(singular)
      expect([num(sl.sign), num(sl.logabsdet)]).toEqual([0, Number.NEGATIVE_INFINITY])
      expect(mat(L.lu(singular).u)[1]?.[1]).toBe(0)
    }
    {
      // The smaller helpers: the Householder reflector H = I - tau v v' maps x to alpha e1, a Givens rotation zeroes b,
      // the symmetry and definiteness tests, choFactor, invTriangular, the Jacobi sweeps and the jittered Cholesky.
      const x = [3, 4, 12]
      const { v, tau, alpha } = L.householderReflector(x)
      const reflected = x.map(
        (xi, i) => xi - tau * (v[i] ?? 0) * v.reduce((acc: number, vj: number, j: number) => acc + vj * (x[j] ?? 0), 0),
      )
      expect([alpha, ...reflected.slice(1)].map((e) => Math.abs(e))).toEqual(
        [13, 0, 0].map((e) => expect.closeTo(e, 12)),
      )
      expect(reflected[0]).toBeCloseTo(alpha, 12)
      expect(L.householderReflector([5, 0, 0]).tau).toBe(0)
      const g = L.givensRotation(3, 4)
      expect([g.c, g.s, g.r]).toEqual([0.6, 0.8, 5].map((e) => expect.closeTo(e, 14)))
      expect(g.c * 3 + g.s * 4).toBeCloseTo(g.r, 14)
      expect(-g.s * 3 + g.c * 4).toBeCloseTo(0, 14)
      expect([
        L.isSymmetric(spd3),
        L.isSymmetric([
          [1, 2],
          [2.1, 1],
        ]),
        L.isSymmetric(
          [
            [1, 2],
            [2.1, 1],
          ],
          0.2,
        ),
      ]).toEqual([true, false, true])
      expect([
        L.isPositiveDefinite(spd3),
        L.isPositiveDefinite([
          [1, 2],
          [2, 1],
        ]),
        L.isPositiveDefinite(hilbert(6)),
      ]).toEqual([true, false, true])
      const factor = L.choFactor(spd3)
      expect(factor.lower).toBe(true)
      expect(fro(sub(mm(mat(factor.c), t(mat(factor.c))), spd3))).toBeLessThan(1e-14)
      const lowerTri = [
        [2, 0, 0],
        [1, 4, 0],
        [-3, 0.5, 5],
      ]
      expect(fro(sub(mm(mat(L.invTriangular(lowerTri, { lower: true })), lowerTri), eye(3)))).toBeLessThan(1e-14)
      const sweeps = L.jacobiEigenSteps(spd3)
      expect(Array.from(sweeps.eigenvalues as ArrayLike<number>)).toEqual(
        vec(L.eigvalsh(spd3)).map((e) => expect.closeTo(e, 12)),
      )
      const norms = sweeps.sweeps.map((sweep: { offDiagonalNorm: number }) => sweep.offDiagonalNorm)
      expect(norms.every((nrm: number, i: number) => i === 0 || nrm < (norms[i - 1] ?? 0))).toBe(true)
      // [[1, 1], [1, 1]] is singular: the ladder adds a small multiple of the identity and reports it, L L' = A + jitter I.
      const jittered = L.choleskyJitter([
        [1, 1],
        [1, 1],
      ])
      const lj = mat(jittered.l)
      expect(jittered.jitter).toBeGreaterThan(0)
      expect(jittered.jitter).toBeLessThanOrEqual(1e-4)
      expect(
        fro(
          sub(mm(lj, t(lj)), [
            [1 + jittered.jitter, 1],
            [1, 1 + jittered.jitter],
          ]),
        ),
      ).toBeLessThan(1e-12)
      expect(L.choleskyJitter(spd3).jitter).toBe(0)
    }
  })
})
