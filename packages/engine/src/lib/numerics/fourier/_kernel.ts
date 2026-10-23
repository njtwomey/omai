// FFT kernels on Float64Array pairs: iterative radix-2, and Bluestein's chirp-z algorithm for other lengths.

const twiddleCache = new Map<number, { cos: Float64Array; sin: Float64Array }>()

function twiddles(n: number): { cos: Float64Array; sin: Float64Array } {
  let t = twiddleCache.get(n)
  if (t === undefined) {
    const c = new Float64Array(n / 2)
    const s = new Float64Array(n / 2)
    for (let k = 0; k < n / 2; k++) {
      // Exact values at multiples of pi / 2 avoid rounding residue.
      const angle = (2 * Math.PI * k) / n
      c[k] = (4 * k) % n === 0 ? ([1, 0, -1, 0][((4 * k) / n) % 4] ?? Math.cos(angle)) : Math.cos(angle)
      s[k] = (4 * k) % n === 0 ? ([0, 1, 0, -1][((4 * k) / n) % 4] ?? Math.sin(angle)) : Math.sin(angle)
    }
    t = { cos: c, sin: s }
    if (twiddleCache.size > 64) twiddleCache.clear()
    twiddleCache.set(n, t)
  }
  return t
}

const isPowerOfTwo = (n: number): boolean => n > 0 && (n & (n - 1)) === 0

/** In-place radix-2 transform with exponent sign `sign` (-1 forward, +1 inverse), unscaled. */
function radix2(re: Float64Array, im: Float64Array, sign: number): void {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      const tr = re[i] ?? 0
      re[i] = re[j] ?? 0
      re[j] = tr
      const ti = im[i] ?? 0
      im[i] = im[j] ?? 0
      im[j] = ti
    }
  }
  const { cos, sin } = twiddles(n)
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1
    const step = n / size
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step] ?? 1
        const wi = sign * (sin[k * step] ?? 0)
        const a = start + k
        const b = a + half
        const xr = (re[b] ?? 0) * wr - (im[b] ?? 0) * wi
        const xi = (re[b] ?? 0) * wi + (im[b] ?? 0) * wr
        re[b] = (re[a] ?? 0) - xr
        im[b] = (im[a] ?? 0) - xi
        re[a] = (re[a] ?? 0) + xr
        im[a] = (im[a] ?? 0) + xi
      }
    }
  }
}

/** Bluestein: X_k = conj(w_k) sum_n (x_n conj(w_n)) w_{k-n}, w_m = exp(i pi m^2 / n), as a power-of-two convolution. */
function bluestein(re: Float64Array, im: Float64Array, sign: number): void {
  const n = re.length
  let m = 1
  while (m < 2 * n - 1) m <<= 1
  const wr = new Float64Array(n)
  const wi = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    // k^2 mod 2n keeps the angle small and exact.
    const angle = (Math.PI * ((k * k) % (2 * n))) / n
    wr[k] = Math.cos(angle)
    wi[k] = -sign * Math.sin(angle)
  }
  const ar = new Float64Array(m)
  const ai = new Float64Array(m)
  for (let k = 0; k < n; k++) {
    // x_k * conj-chirp (the chirp with the transform's sign).
    ar[k] = (re[k] ?? 0) * (wr[k] ?? 0) + (im[k] ?? 0) * (wi[k] ?? 0)
    ai[k] = (im[k] ?? 0) * (wr[k] ?? 0) - (re[k] ?? 0) * (wi[k] ?? 0)
  }
  const br = new Float64Array(m)
  const bi = new Float64Array(m)
  br[0] = wr[0] ?? 1
  bi[0] = wi[0] ?? 0
  for (let k = 1; k < n; k++) {
    br[k] = br[m - k] = wr[k] ?? 0
    bi[k] = bi[m - k] = wi[k] ?? 0
  }
  radix2(ar, ai, -1)
  radix2(br, bi, -1)
  for (let k = 0; k < m; k++) {
    const r = (ar[k] ?? 0) * (br[k] ?? 0) - (ai[k] ?? 0) * (bi[k] ?? 0)
    const i = (ar[k] ?? 0) * (bi[k] ?? 0) + (ai[k] ?? 0) * (br[k] ?? 0)
    ar[k] = r
    ai[k] = i
  }
  radix2(ar, ai, 1)
  for (let k = 0; k < n; k++) {
    const r = (ar[k] ?? 0) / m
    const i = (ai[k] ?? 0) / m
    re[k] = r * (wr[k] ?? 0) + i * (wi[k] ?? 0)
    im[k] = i * (wr[k] ?? 0) - r * (wi[k] ?? 0)
  }
}

/** In-place DFT of one sequence: sum_n x_n exp(sign 2 pi i k n / N), times scale. */
export function transform(re: Float64Array, im: Float64Array, sign: number, scale: number): void {
  const n = re.length
  if (n <= 1) {
    if (n === 1) {
      re[0] = (re[0] ?? 0) * scale
      im[0] = (im[0] ?? 0) * scale
    }
    return
  }
  if (isPowerOfTwo(n)) radix2(re, im, sign)
  else bluestein(re, im, sign)
  if (scale !== 1) {
    for (let k = 0; k < n; k++) {
      re[k] = (re[k] ?? 0) * scale
      im[k] = (im[k] ?? 0) * scale
    }
  }
}

/** Apply {@link transform} along one axis of row-major data with the given shape. */
export function transformAxis(
  re: Float64Array,
  im: Float64Array,
  shape: readonly number[],
  axis: number,
  sign: number,
  scale: number,
): { re: Float64Array; im: Float64Array } {
  const n = shape[axis] ?? 1
  const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1)
  const outer = shape.slice(0, axis).reduce((a, b) => a * b, 1)
  const outRe = Float64Array.from(re)
  const outIm = Float64Array.from(im)
  const lineRe = new Float64Array(n)
  const lineIm = new Float64Array(n)
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      const base = o * n * inner + i
      for (let k = 0; k < n; k++) {
        lineRe[k] = outRe[base + k * inner] ?? 0
        lineIm[k] = outIm[base + k * inner] ?? 0
      }
      transform(lineRe, lineIm, sign, scale)
      for (let k = 0; k < n; k++) {
        outRe[base + k * inner] = lineRe[k] ?? 0
        outIm[base + k * inner] = lineIm[k] ?? 0
      }
    }
  }
  return { re: outRe, im: outIm }
}
