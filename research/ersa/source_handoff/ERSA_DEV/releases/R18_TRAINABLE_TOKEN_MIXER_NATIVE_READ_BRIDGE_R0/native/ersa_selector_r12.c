/* ERSA R12 experimental frozen linear neural scorer. No admission, state, or
 * training authority. The host supplies exactly eight validated weights.
 * Two-phase evaluation prevents partial output on any malformed candidate.
 */
#include <stdint.h>
#include <math.h>
#include <stddef.h>
int ersa12_score(int64_t n, const double *x, const double *w, double *out) {
    if (n<0 || n>8192 || !x || !w || !out) return -1;
    for (int j=0;j<8;j++) if (!isfinite(w[j]) || fabs(w[j])>1.e6) return -2;
    for (int64_t i=0;i<n;i++) {
        long double v=0.;
        for (int j=0;j<8;j++) {
            const double f=x[8*i+j];
            if (!isfinite(f) || fabs(f)>1.e7) return -3;
            v+=(long double)f*(long double)w[j];
        }
        if (!isfinite((double)v)) return -4;
    }
    for (int64_t i=0;i<n;i++) {
        long double v=0.;
        for (int j=0;j<8;j++) v+=(long double)x[8*i+j]*(long double)w[j];
        out[i]=(double)v;
    }
    return 0;
}
