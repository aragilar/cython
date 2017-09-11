# NumPy math library
#
# This exports the functionality of the NumPy core math library, aka npymath,
# which provides implementations of C99 math functions and macros for system
# with a C89 library (such as MSVC). npymath is available with NumPy >=1.3,
# although some functions will require later versions. The spacing function is
# not in C99, but comes from Fortran.
#
# On the Cython side, the npymath functions are available without the "npy_"
# prefix that they have in C, to make this is a drop-in replacement for
# libc.math. The same is true for the constants, where possible.
#
# See the NumPy documentation for linking instructions.
#
# Complex number support and NumPy 2.0 half-precision functions are currently
# not exported.
#
# Author: Lars Buitinck

cdef extern from "numpy/npy_math.h" nogil:
    # Floating-point classification
    long double NAN "NPY_NAN"
    long double INFINITY "NPY_INFINITY"
    long double PZERO "NPY_PZERO"        # positive zero
    long double NZERO "NPY_NZERO"        # negative zero

    # These four are actually macros and work on any floating-point type.
    int isinf "npy_isinf"(long double)  # -1 / 0 / 1
    bint isfinite "npy_isfinite"(long double)
    bint isnan "npy_isnan"(long double)
    bint signbit "npy_signbit"(long double)

    # Math constants
    long double E "NPY_E"
    long double LOG2E "NPY_LOG2E"       # ln(e) / ln(2)
    long double LOG10E "NPY_LOG10E"     # ln(e) / ln(10)
    long double LOGE2 "NPY_LOGE2"       # ln(2)
    long double LOGE10 "NPY_LOGE10"     # ln(10)
    long double PI "NPY_PI"
    long double PI_2 "NPY_PI_2"         # pi / 2
    long double PI_4 "NPY_PI_4"         # pi / 4
    long double NPY_1_PI                # 1 / pi; NPY_ because of ident syntax
    long double NPY_2_PI                # 2 / pi
    long double EULER "NPY_EULER"       # Euler constant (gamma, 0.57721)

    # Low-level floating point manipulation (NumPy >=1.4)
    float copysignf "npy_copysignf"(float, float)
    float nextafterf "npy_nextafterf"(float x, float y)
    float spacingf "npy_spacingf"(float x)
    double copysign "npy_copysign"(double, double)
    double nextafter "npy_nextafter"(double x, double y)
    double spacing "npy_spacing"(double x)
    long double copysignl "npy_copysignl"(long double, long double)
    long double nextafterl "npy_nextafterl"(long double x, long double y)
    long double spacingl "npy_spacingl"(long double x)

    # Float C99 functions
    float sinf "npy_sinf"(float x)
    float cosf "npy_cosf"(float x)
    float tanf "npy_tanf"(float x)
    float sinhf "npy_sinhf"(float x)
    float coshf "npy_coshf"(float x)
    float tanhf "npy_tanhf"(float x)
    float fabsf "npy_fabsf"(float x)
    float floorf "npy_floorf"(float x)
    float ceilf "npy_ceilf"(float x)
    float rintf "npy_rintf"(float x)
    float sqrtf "npy_sqrtf"(float x)
    float log10f "npy_log10f"(float x)
    float logf "npy_logf"(float x)
    float expf "npy_expf"(float x)
    float expm1f "npy_expm1f"(float x)
    float asinf "npy_asinf"(float x)
    float acosf "npy_acosf"(float x)
    float atanf "npy_atanf"(float x)
    float asinhf "npy_asinhf"(float x)
    float acoshf "npy_acoshf"(float x)
    float atanhf "npy_atanhf"(float x)
    float log1pf "npy_log1pf"(float x)
    float exp2f "npy_exp2f"(float x)
    float log2f "npy_log2f"(float x)
    float atan2f "npy_atan2f"(float x, float y)
    float hypotf "npy_hypotf"(float x, float y)
    float powf "npy_powf"(float x, float y)
    float fmodf "npy_fmodf"(float x, float y)
    float modff "npy_modff"(float x, float* y)

    # Long double C99 functions
    long double sinl "npy_sinl"(long double x)
    long double cosl "npy_cosl"(long double x)
    long double tanl "npy_tanl"(long double x)
    long double sinhl "npy_sinhl"(long double x)
    long double coshl "npy_coshl"(long double x)
    long double tanhl "npy_tanhl"(long double x)
    long double fabsl "npy_fabsl"(long double x)
    long double floorl "npy_floorl"(long double x)
    long double ceill "npy_ceill"(long double x)
    long double rintl "npy_rintl"(long double x)
    long double sqrtl "npy_sqrtl"(long double x)
    long double log10l "npy_log10l"(long double x)
    long double logl "npy_logl"(long double x)
    long double expl "npy_expl"(long double x)
    long double expm1l "npy_expm1l"(long double x)
    long double asinl "npy_asinl"(long double x)
    long double acosl "npy_acosl"(long double x)
    long double atanl "npy_atanl"(long double x)
    long double asinhl "npy_asinhl"(long double x)
    long double acoshl "npy_acoshl"(long double x)
    long double atanhl "npy_atanhl"(long double x)
    long double log1pl "npy_log1pl"(long double x)
    long double exp2l "npy_exp2l"(long double x)
    long double log2l "npy_log2l"(long double x)
    long double atan2l "npy_atan2l"(long double x, long double y)
    long double hypotl "npy_hypotl"(long double x, long double y)
    long double powl "npy_powl"(long double x, long double y)
    long double fmodl "npy_fmodl"(long double x, long double y)
    long double modfl "npy_modfl"(long double x, long double* y)

    # NumPy extensions
    float deg2radf "npy_deg2radf"(float x)
    float rad2degf "npy_rad2degf"(float x)
    float logaddexpf "npy_logaddexpf"(float x, float y)
    float logaddexp2f "npy_logaddexp2f"(float x, float y)

    double deg2rad "npy_deg2rad"(double x)
    double rad2deg "npy_rad2deg"(double x)
    double logaddexp "npy_logaddexp"(double x, double y)
    double logaddexp2 "npy_logaddexp2"(double x, double y)

    long double deg2radl "npy_deg2radl"(long double x)
    long double rad2degl "npy_rad2degl"(long double x)
    long double logaddexpl "npy_logaddexpl"(long double x, long double y)
    long double logaddexp2l "npy_logaddexp2l"(long double x, long double y)

    # Additional float types
    _Float32 sinf32 "npy_sinf32"(_Float32 x)
    _Float32 cosf32 "npy_cosf32"(_Float32 x)
    _Float32 tanf32 "npy_tanf32"(_Float32 x)
    _Float32 sinhf32 "npy_sinhf32"(_Float32 x)
    _Float32 coshf32 "npy_coshf32"(_Float32 x)
    _Float32 tanhf32 "npy_tanhf32"(_Float32 x)
    _Float32 fabsf32 "npy_fabsf32"(_Float32 x)
    _Float32 floorf32 "npy_floorf32"(_Float32 x)
    _Float32 ceilf32 "npy_ceilf32"(_Float32 x)
    _Float32 rintf32 "npy_rintf32"(_Float32 x)
    _Float32 sqrtf32 "npy_sqrtf32"(_Float32 x)
    _Float32 log10f32 "npy_log10f32"(_Float32 x)
    _Float32 logf32 "npy_logf32"(_Float32 x)
    _Float32 expf32 "npy_expf32"(_Float32 x)
    _Float32 expm1f32 "npy_expm1f32"(_Float32 x)
    _Float32 asinf32 "npy_asinf32"(_Float32 x)
    _Float32 acosf32 "npy_acosf32"(_Float32 x)
    _Float32 atanf32 "npy_atanf32"(_Float32 x)
    _Float32 asinhf32 "npy_asinhf32"(_Float32 x)
    _Float32 acoshf32 "npy_acoshf32"(_Float32 x)
    _Float32 atanhf32 "npy_atanhf32"(_Float32 x)
    _Float32 log1pf32 "npy_log1pf32"(_Float32 x)
    _Float32 exp2f32 "npy_exp2f32"(_Float32 x)
    _Float32 log2f32 "npy_log2f32"(_Float32 x)
    _Float32 atan2f32 "npy_atan2f32"(_Float32 x, _Float32 y)
    _Float32 hypotf32 "npy_hypotf32"(_Float32 x, _Float32 y)
    _Float32 powf32 "npy_powf32"(_Float32 x, _Float32 y)
    _Float32 fmodf32 "npy_fmodf32"(_Float32 x, _Float32 y)
    _Float32 modff32 "npy_modff32"(_Float32 x, _Float32* y)
    _Float32 deg2radf32 "npy_deg2radf32"(_Float32 x)
    _Float32 rad2degf32 "npy_rad2degf32"(_Float32 x)
    _Float32 logaddexpf32 "npy_logaddexpf32"(_Float32 x, _Float32 y)
    _Float32 logaddexp2f32 "npy_logaddexp2f32"(_Float32 x, _Float32 y)

    _Float64 sinf64 "npy_sinf64"(_Float64 x)
    _Float64 cosf64 "npy_cosf64"(_Float64 x)
    _Float64 tanf64 "npy_tanf64"(_Float64 x)
    _Float64 sinhf64 "npy_sinhf64"(_Float64 x)
    _Float64 coshf64 "npy_coshf64"(_Float64 x)
    _Float64 tanhf64 "npy_tanhf64"(_Float64 x)
    _Float64 fabsf64 "npy_fabsf64"(_Float64 x)
    _Float64 floorf64 "npy_floorf64"(_Float64 x)
    _Float64 ceilf64 "npy_ceilf64"(_Float64 x)
    _Float64 rintf64 "npy_rintf64"(_Float64 x)
    _Float64 sqrtf64 "npy_sqrtf64"(_Float64 x)
    _Float64 log10f64 "npy_log10f64"(_Float64 x)
    _Float64 logf64 "npy_logf64"(_Float64 x)
    _Float64 expf64 "npy_expf64"(_Float64 x)
    _Float64 expm1f64 "npy_expm1f64"(_Float64 x)
    _Float64 asinf64 "npy_asinf64"(_Float64 x)
    _Float64 acosf64 "npy_acosf64"(_Float64 x)
    _Float64 atanf64 "npy_atanf64"(_Float64 x)
    _Float64 asinhf64 "npy_asinhf64"(_Float64 x)
    _Float64 acoshf64 "npy_acoshf64"(_Float64 x)
    _Float64 atanhf64 "npy_atanhf64"(_Float64 x)
    _Float64 log1pf64 "npy_log1pf64"(_Float64 x)
    _Float64 exp2f64 "npy_exp2f64"(_Float64 x)
    _Float64 log2f64 "npy_log2f64"(_Float64 x)
    _Float64 atan2f64 "npy_atan2f64"(_Float64 x, _Float64 y)
    _Float64 hypotf64 "npy_hypotf64"(_Float64 x, _Float64 y)
    _Float64 powf64 "npy_powf64"(_Float64 x, _Float64 y)
    _Float64 fmodf64 "npy_fmodf64"(_Float64 x, _Float64 y)
    _Float64 modff64 "npy_modff64"(_Float64 x, _Float64* y)
    _Float64 deg2radf64 "npy_deg2radf64"(_Float64 x)
    _Float64 rad2degf64 "npy_rad2degf64"(_Float64 x)
    _Float64 logaddexpf64 "npy_logaddexpf64"(_Float64 x, _Float64 y)
    _Float64 logaddexp2f64 "npy_logaddexp2f64"(_Float64 x, _Float64 y)

    _Float128 sinf128 "npy_sinf128"(_Float128 x)
    _Float128 cosf128 "npy_cosf128"(_Float128 x)
    _Float128 tanf128 "npy_tanf128"(_Float128 x)
    _Float128 sinhf128 "npy_sinhf128"(_Float128 x)
    _Float128 coshf128 "npy_coshf128"(_Float128 x)
    _Float128 tanhf128 "npy_tanhf128"(_Float128 x)
    _Float128 fabsf128 "npy_fabsf128"(_Float128 x)
    _Float128 floorf128 "npy_floorf128"(_Float128 x)
    _Float128 ceilf128 "npy_ceilf128"(_Float128 x)
    _Float128 rintf128 "npy_rintf128"(_Float128 x)
    _Float128 sqrtf128 "npy_sqrtf128"(_Float128 x)
    _Float128 log10f128 "npy_log10f128"(_Float128 x)
    _Float128 logf128 "npy_logf128"(_Float128 x)
    _Float128 expf128 "npy_expf128"(_Float128 x)
    _Float128 expm1f128 "npy_expm1f128"(_Float128 x)
    _Float128 asinf128 "npy_asinf128"(_Float128 x)
    _Float128 acosf128 "npy_acosf128"(_Float128 x)
    _Float128 atanf128 "npy_atanf128"(_Float128 x)
    _Float128 asinhf128 "npy_asinhf128"(_Float128 x)
    _Float128 acoshf128 "npy_acoshf128"(_Float128 x)
    _Float128 atanhf128 "npy_atanhf128"(_Float128 x)
    _Float128 log1pf128 "npy_log1pf128"(_Float128 x)
    _Float128 exp2f128 "npy_exp2f128"(_Float128 x)
    _Float128 log2f128 "npy_log2f128"(_Float128 x)
    _Float128 atan2f128 "npy_atan2f128"(_Float128 x, _Float128 y)
    _Float128 hypotf128 "npy_hypotf128"(_Float128 x, _Float128 y)
    _Float128 powf128 "npy_powf128"(_Float128 x, _Float128 y)
    _Float128 fmodf128 "npy_fmodf128"(_Float128 x, _Float128 y)
    _Float128 modff128 "npy_modff128"(_Float128 x, _Float128* y)
    _Float128 deg2radf128 "npy_deg2radf128"(_Float128 x)
    _Float128 rad2degf128 "npy_rad2degf128"(_Float128 x)
    _Float128 logaddexpf128 "npy_logaddexpf128"(_Float128 x, _Float128 y)
    _Float128 logaddexp2f128 "npy_logaddexp2f128"(_Float128 x, _Float128 y)
