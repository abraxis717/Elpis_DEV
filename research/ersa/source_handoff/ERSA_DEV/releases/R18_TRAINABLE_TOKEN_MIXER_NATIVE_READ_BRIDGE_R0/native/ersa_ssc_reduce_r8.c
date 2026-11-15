/* R8 explicit SSC compatibility reducer. Distinct from ERSA signed L1 rule.
 * No state, no admission rights, no persistent allocation.
 * Inputs must already be authenticated/transported to SSC canonical frame.
 */
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#define MAX_N 8192
int ersa_ssc_softmax_r8(int64_t n,const double *vectors,double temperature,
                         double *salient,double *collective){
    if(n<1 || n>MAX_N || !vectors || !salient || !collective ||
       !isfinite(temperature) || temperature<=0.0) return -1;
    long double maxexp=-HUGE_VALL;
    for(int64_t i=0;i<n;i++){
        long double s=0;
        for(int j=0;j<4;j++){
            double v=vectors[4*i+j];
            if(!isfinite(v))return -2;
            s+=(long double)v*(long double)v;
        }
        long double e=(long double)temperature*sqrtl(s);
        if(!isfinite(e))return -3;
        if(e>maxexp)maxexp=e;
    }
    long double den=0,ws[4]={0,0,0,0},us[4]={0,0,0,0};
    for(int64_t i=0;i<n;i++){
        long double s=0;
        for(int j=0;j<4;j++)s+=(long double)vectors[4*i+j]*vectors[4*i+j];
        long double w=expl((long double)temperature*sqrtl(s)-maxexp);
        den+=w;
        for(int j=0;j<4;j++){
            ws[j]+=w*vectors[4*i+j];
            us[j]+=vectors[4*i+j];
        }
    }
    if(!(den>0) || !isfinite(den))return -4;
    double os[4],oc[4];
    for(int j=0;j<4;j++){
        os[j]=(double)(ws[j]/den);
        oc[j]=(double)(us[j]/(long double)n);
        if(!isfinite(os[j]) || !isfinite(oc[j]))return -5;
    }
    for(int j=0;j<4;j++){
        salient[j]=os[j];collective[j]=oc[j];
    }
    return 0;
}
