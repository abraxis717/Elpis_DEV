/* R13 offline-trained bounded halting head. State-free, bounded and fail-closed.
   Action: 0=CONTINUE, 1=STOP, 2=ABSTAIN; negative return=invalid.
   The caller owns its admission permissions and exact native state. */
#include <math.h>
#include <stddef.h>
int ersa13_halt(const double *features, const double *weights, int step,
                int max_steps, double *prob, int *action) {
    if (!features || !weights || !prob || !action || step<1 || max_steps<1 ||
        max_steps>4 || step>max_steps) return -1;
    long double z=weights[5];
    for (int j=0; j<6; j++) if (!isfinite(weights[j]) || fabs(weights[j])>1.e5) return -2;
    for (int j=0; j<5; j++) {
        if (!isfinite(features[j]) || fabs(features[j])>1.e5) return -3;
        z+=(long double)features[j]*(long double)weights[j];
    }
    if (!isfinite((double)z)) return -4;
    double zz=(double)z;
    double p=zz>=0.?1./(1.+exp(-zz)):exp(zz)/(1.+exp(zz));
    if (!isfinite(p)) return -5;
    int a=p>=.70?1:p<=.30?0:2;
    if (a==0 && step==max_steps) a=2;
    *prob=p; *action=a;
    return 0;
}
