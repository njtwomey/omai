"""Shading terms against closed forms and scipy quadrature: the exact Fresnel equations (s and p reflectances from
Snell's law) and Schlick's approximation, the GGX distribution, Smith masking through its Lambda form, Phong and
Blinn-Phong terms, the Cook-Torrance and metallic-roughness BRDFs written out in numpy, and directional albedos (the
integral of BRDF times cosine over the hemisphere) by scipy.integrate.dblquad."""

from collections.abc import Callable

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.integrate import dblquad

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def unit_rows(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fresnel(cos_i: FloatArray, n1: float, n2: float) -> FloatArray:
    sin_t = n1 / n2 * np.sqrt(np.clip(1 - cos_i**2, 0, None))
    out = np.ones_like(cos_i)
    ok = sin_t < 1
    ct = np.sqrt(1 - sin_t[ok] ** 2)
    ci = cos_i[ok]
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
    out[ok] = (rs**2 + rp**2) / 2
    return out


def schlick(cos: FloatArray, f0: FloatArray | float) -> FloatArray:
    return f0 + (1 - np.asarray(f0)) * (1 - cos) ** 5


def schlick_dielectric(cos_i: FloatArray, n1: float, n2: float) -> FloatArray:
    r0 = ((n1 - n2) / (n1 + n2)) ** 2
    sin2t = (n1 / n2) ** 2 * (1 - cos_i**2)
    c = np.sqrt(np.clip(1 - sin2t, 0, None)) if n1 > n2 else cos_i
    return np.where(sin2t > 1, 1.0, schlick(c, r0))


def ggx(cos_h: FloatArray, alpha: FloatArray | float) -> FloatArray:
    a2 = np.asarray(alpha) ** 2
    return np.where(cos_h > 0, a2 / (np.pi * (cos_h**2 * (a2 - 1) + 1) ** 2), 0.0)


def smith_g1(cos: FloatArray, alpha: FloatArray | float) -> FloatArray:
    tan2 = (1 - cos**2) / cos**2
    lam = (-1 + np.sqrt(1 + np.asarray(alpha) ** 2 * tan2)) / 2
    return 1 / (1 + lam)


def cook_torrance(n: FloatArray, ld: FloatArray, v: FloatArray, rough: float, f0: FloatArray) -> FloatArray:
    alpha = rough**2
    h = unit_rows(ld + v)
    nl = np.sum(n * ld, axis=-1)
    nv = np.sum(n * v, axis=-1)
    d = ggx(np.sum(n * h, axis=-1), alpha)
    g = smith_g1(nl, alpha) * smith_g1(nv, alpha)
    f = schlick(np.clip(np.sum(h * v, axis=-1), 0, None)[..., None], f0)
    spec = (d * g / (4 * nl * nv))[..., None] * f
    return np.where(((nl > 0) & (nv > 0))[..., None], spec, 0.0)


def metallic_roughness(
    n: FloatArray,
    ld: FloatArray,
    v: FloatArray,
    albedo: FloatArray,
    rough: float,
    metal: float,
) -> FloatArray:
    f0 = 0.04 * (1 - metal) + metal * albedo
    h = unit_rows(ld + v)
    f = schlick(np.clip(np.sum(h * v, axis=-1), 0, None)[..., None], f0)
    lit = (np.sum(n * ld, axis=-1) > 0) & (np.sum(n * v, axis=-1) > 0)
    diffuse = (1 - f) * (1 - metal) * albedo / np.pi
    return cook_torrance(n, ld, v, rough, f0) + np.where(lit[..., None], diffuse, 0.0)


def directional_albedo(f: Callable[[FloatArray], float]) -> float:
    """The integral over the upper hemisphere (z up) of f(direction) cos(theta)."""

    def integrand(phi: float, theta: float) -> float:
        d = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        return f(d) * np.cos(theta) * np.sin(theta)

    value, _ = dblquad(integrand, 0, np.pi / 2, 0, 2 * np.pi, epsabs=1e-13, epsrel=1e-11)
    return float(value)


def cases() -> list[Case]:
    g = rng(9090)
    out: list[Case] = []
    cos = np.linspace(0.02, 1, 12)
    for n1, n2 in ((1.0, 1.5), (1.5, 1.0), (1.0, 2.4)):
        out.append(case(f"fresnel/{n1}-{n2}", "fresnel", [cos], fresnel(cos, n1, n2), n1=n1, n2=n2))
        out.append(
            case(
                f"schlickDielectric/{n1}-{n2}",
                "schlickDielectric",
                [cos],
                schlick_dielectric(cos, n1, n2),
                n1=n1,
                n2=n2,
            )
        )
    f0 = g.uniform(0, 1, size=12)
    out.append(case("schlick", "schlick", [cos, f0], schlick(cos, f0)))
    alpha = g.uniform(0.05, 1, size=12)
    out.append(case("ggx", "ggx", [cos, alpha], ggx(cos, alpha), rtol=1e-12))
    out.append(case("smith", "smith", [cos, alpha], smith_g1(cos, alpha), rtol=1e-12))
    nz = np.array([0.0, 0.0, 1.0])
    k = 10
    n = unit_rows(nz + g.uniform(-0.3, 0.3, size=(k, 3)))
    ld = unit_rows(nz + g.uniform(-0.8, 0.8, size=(k, 3)))
    v = unit_rows(nz + g.uniform(-0.8, 0.8, size=(k, 3)))
    s = 9.0
    r = 2 * np.sum(n * ld, axis=-1, keepdims=True) * n - ld
    rv = np.sum(r * v, -1)
    out.append(case("phong", "phong", [n, ld, v], np.where(rv > 0, rv, 0) ** s, rtol=1e-11, shininess=s))
    h = unit_rows(ld + v)
    out.append(
        case("blinnPhong", "blinnPhong", [n, ld, v], np.clip(np.sum(n * h, -1), 0, None) ** s, rtol=1e-11, shininess=s)
    )
    f0c = np.array([0.9, 0.6, 0.2])
    out.append(
        case(
            "cookTorrance",
            "cookTorrance",
            [n, ld, v, f0c],
            cook_torrance(n, ld, v, 0.45, f0c),
            rtol=1e-11,
            roughness=0.45,
        )
    )
    albedo = np.array([0.8, 0.3, 0.1])
    mr = metallic_roughness(n, ld, v, albedo, 0.6, 0.3)
    out.append(
        case("metallicRoughness", "metallicRoughness", [n, ld, v, albedo], mr, rtol=1e-11, roughness=0.6, metallic=0.3)
    )
    # Directional albedos by adaptive quadrature: light at angle theta in the xz-plane, viewer integrated over.
    for shininess, theta in ((1.0, 0.0), (10.0, 0.0), (10.0, 0.5), (40.0, 0.3)):
        light = np.array([np.sin(theta), 0, np.cos(theta)])
        refl = 2 * light[2] * nz - light

        def phong_lobe(d: FloatArray, s: float = shininess, r_: FloatArray = refl) -> float:
            return (s + 2) / (2 * np.pi) * max(0.0, float(r_ @ d)) ** s

        out.append(
            case(
                f"albedo/phong/{shininess}/{theta}",
                "albedoPhong",
                [light],
                np.array(directional_albedo(phong_lobe)),
                rtol=1e-6,
                atol=1e-8,
                shininess=shininess,
            )
        )
    for rough, theta in ((0.5, 0.0), (0.7, 0.8), (1.0, 1.2)):
        light = np.array([np.sin(theta), 0, np.cos(theta)])

        def ct(d: FloatArray, r_: float = rough, l_: FloatArray = light) -> float:
            return float(cook_torrance(nz[None], l_[None], d[None], r_, np.array([1.0]))[0, 0])

        out.append(
            case(
                f"albedo/cookTorrance/{rough}/{theta}",
                "albedoCookTorrance",
                [light],
                np.array(directional_albedo(ct)),
                rtol=1e-6,
                atol=1e-8,
                roughness=rough,
            )
        )
    for a in (0.3, 0.6, 1.0):

        def ndf(d: FloatArray, a_: float = a) -> float:
            return float(ggx(np.array([d[2]]), a_)[0])

        out.append(
            case(
                f"normalisation/ggx/{a}",
                "ggxNormalisation",
                [],
                np.array(directional_albedo(ndf)),
                rtol=1e-6,
                atol=1e-8,
                alpha=a,
            )
        )
    return out
