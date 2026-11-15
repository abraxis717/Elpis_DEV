/* ERSA R14: a single bounded C-native recurrent read decision.
 * Authority: caller-supplied immutable exact-state snapshot and admission mask.
 * Does not allocate persistent memory, mutate inputs, read/write files, perform
 * admission, or acquire independent state authority.  Max 8192 records / 4 hops.
 * Return 0 for computed OK/HOLD receipt; nonzero for invalid input, with
 * guaranteed unchanged receipt.  No oracle paths or labels are inputs.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <string.h>
#define ERSA14_CAP 8192
#define ERSA14_MAXHOPS 4
#define ERSA14_CONTINUE 0
#define ERSA14_STOP 1
#define ERSA14_ABSTAIN 2
/* Receipt statuses: 0 OK, 1 uncertain stop, 2 exhausted,
 * 3 no admitted candidate, 4 ambiguous, 5 cyclic path. Invalid inputs return <0.
 */
typedef struct {
    int32_t status;
    int32_t count;
    int32_t path[4];
    double answer[4];
} Ersa14Receipt;
static int valid_vec(const double *p,int n) {
    for(int j=0;j<n;j++)if(!isfinite(p[j]) || fabs(p[j])>1.0e5)return 0;
    return 1;
}
static double cos4(const double *a,const double *b) {
    double aa=0.,bb=0.,ab=0.;
    for(int j=0;j<4;j++) {aa+=a[j]*a[j];bb+=b[j]*b[j];ab+=a[j]*b[j];}
    double na=sqrt(aa),nb=sqrt(bb);
    return na<1.0e-14 || nb<1.0e-14 ? 0.0 : ab/(na*nb);
}
static int better(double lhs,int li,double rhs,int ri){
    return lhs>rhs || (lhs==rhs && li<ri);
}
static int action_from_features(const double x[5], const double w[6],int step,int maxsteps) {
    long double z=w[5];
    for(int i=0;i<5;i++)z+=(long double)x[i]*(long double)w[i];
    if(!isfinite((double)z))return -1;
    double d=(double)z;
    double prob=d>=0.?1./(1.+exp(-d)):exp(d)/(1.+exp(d));
    if(!isfinite(prob))return -1;
    int action=prob>=.70?ERSA14_STOP:prob<=.30?ERSA14_CONTINUE:ERSA14_ABSTAIN;
    if(action==ERSA14_CONTINUE && step==maxsteps)action=ERSA14_ABSTAIN;
    return action;
}
int ersa14_recur(int64_t n,const double *keys,const double *goals,const double *values,
    const double *meta,const int64_t *issued,const uint8_t *allowed,
    const double *selector_w,const double *halt_w,const double *qkey,
    const double *qgoal,const double *qdir,int maxsteps,int policy,
    double margin_norm,Ersa14Receipt *output) {
    if(n<1 || n>ERSA14_CAP || maxsteps<1 || maxsteps>ERSA14_MAXHOPS ||
       policy<0 || policy>3 || !keys || !goals || !values || !meta ||
       !issued || !allowed || !selector_w || !halt_w || !qkey || !qgoal ||
       !qdir || !output || !isfinite(margin_norm) || margin_norm<0.0 ||
       margin_norm>1.)return -1;
    if(!valid_vec(selector_w,8) || !valid_vec(halt_w,6) ||
       !valid_vec(qkey,4) || !valid_vec(qgoal,4) || !valid_vec(qdir,4))return -2;
    long double norm=0.;
    for(int j=0;j<8;j++)norm+=(long double)selector_w[j]*selector_w[j];
    double selector_norm=sqrt((double)norm);
    if(!isfinite(selector_norm) || selector_norm<1.0e-15)return -3;
    for(int64_t i=0;i<n;i++) {
        if(allowed[i]>1 || issued[i]<0 || issued[i]>64 ||
           !valid_vec(keys+4*i,4) || !valid_vec(goals+4*i,4) ||
           !valid_vec(values+4*i,4) || !valid_vec(meta+4*i,4) ||
           (meta[4*i+2]!=0. && meta[4*i+2]!=-1.) ||
           (meta[4*i+3]!=0. && meta[4*i+3]!=1.))return -4;
    }
    Ersa14Receipt r={0};
    r.status=2;
    for(int j=0;j<4;j++)r.path[j]=-1;
    double query[4];memcpy(query,qkey,sizeof(query));
    for(int step=1;step<=maxsteps;step++) {
        int best=-1,second=-1;
        double rawbest=-INFINITY,rawsecond=-INFINITY;
        double rankbest=-INFINITY,ranksecond=-INFINITY;
        for(int i=0;i<(int)n;i++)if(allowed[i]) {
            double ff[8]={cos4(keys+4*i,query),meta[4*i],
                exp(-((double)(64-issued[i]))/64.0),meta[4*i+1],
                meta[4*i+2],cos4(goals+4*i,qgoal),meta[4*i+3],
                cos4(values+4*i,qdir)};
            long double z=0.;
            for(int j=0;j<8;j++)z+=(long double)ff[j]*selector_w[j];
            if(!isfinite((double)z))return -5;
            double score=(double)z;
            double canonical=nearbyint(score*1.e12)/1.e12;
            if(!isfinite(canonical))return -5;
            if(best<0 || better(canonical,i,rankbest,best)) {
                second=best;ranksecond=rankbest;rawsecond=rawbest;
                best=i;rankbest=canonical;rawbest=score;
            } else if(second<0 || better(canonical,i,ranksecond,second)) {
                second=i;ranksecond=canonical;rawsecond=score;
            }
        }
        if(best<0){r.status=3;break;}
        double gap=second<0?INFINITY:rawbest-rawsecond;
        if(gap<margin_norm*selector_norm){r.status=4;break;}
        r.path[r.count++]=best;
        const double *v=values+4*best;
        double simgoal=cos4(v,qgoal);
        double maxkey=-INFINITY;
        for(int i=0;i<(int)n;i++) {
            double sim=cos4(v,keys+4*i);
            if(sim>maxkey)maxkey=sim;
        }
        double margin=gap/selector_norm;
        if(margin<0.)margin=0.;
        if(margin>2. || !isfinite(margin))margin=2.;
        double hf[5]={simgoal,maxkey,simgoal-maxkey,
            (double)step/(double)ERSA14_MAXHOPS,margin};
        int action;
        if(policy<=1) action=action_from_features(hf,halt_w,step,maxsteps);
        else if(policy==2) action=hf[0]>=.80?ERSA14_STOP:
                        (step==maxsteps?ERSA14_ABSTAIN:ERSA14_CONTINUE);
        else action=step==2?ERSA14_STOP:
                        (step==maxsteps?ERSA14_ABSTAIN:ERSA14_CONTINUE);
        if(action<0)return -6;
        /* An admitted key match is a pointer, not a final answer. */
        if(action==ERSA14_STOP && hf[1]>=.999999)
            action=step<maxsteps?ERSA14_CONTINUE:ERSA14_ABSTAIN;
        if(action==ERSA14_STOP) {
            int repeated=0;
            for(int j=0;j<r.count-1;j++)if(r.path[j]==best)repeated=1;
            if(repeated){r.status=5;break;}
            r.status=0;memcpy(r.answer,v,sizeof(r.answer));break;
        }
        if(action==ERSA14_ABSTAIN){r.status=1;break;}
        memcpy(query,v,sizeof(query));
    }
    memcpy(output,&r,sizeof(r));
    return 0;
}
