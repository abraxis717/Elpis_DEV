/* ERSA R11 experimental typed C SSC state-machine, RAM-only, no I/O.
 * All 14 public SSC transitions mutate bounded C-owned typed structures.
 * Immutable admission registry, transport operators, and input validation are
 * injected by a host boundary; they are NOT independently C-attested.
 * One pthread mutex serializes operations; all error paths which can fail
 * after admission are evaluated before any irreversible update.
 */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include <math.h>

enum { BEGIN, OBSERVE, READ_EXACT, UPSERT, READ_ASSOC, PROPOSE, AGGREGATE,
       APPLY, REINFORCE, DECAY, EVICT, STEP, RESET, HALT };
enum { OK, REJECTED, NOT_FOUND, PENDING };
enum { NO_CODE, INVALID_ADDRESS, DUPLICATE_IDENTITY, CAPACITY_EXCEEDED,
       OP_BUDGET, STEP_BUDGET, UNADMITTED_RELATION, UNKNOWN_PARTICIPANT,
       ROLE_MISMATCH, ORDINAL_MISMATCH, FLAGS_MISMATCH, PROVENANCE_MISMATCH,
       STALE_TICKET, UNBOUND_FRAME, UNAUTHORIZED_WRITE, PROTECTED_RECORD,
       DISCONNECTED_PARTICIPANT, INVALID_VALUE, NONFINITE_VALUE,
       STALE_PROPOSAL, UNKNOWN_RELATION };

typedef struct {
    int entity_cap,relation_cap,control_cap,window_cap,conv_slots,conv_dim;
    int max_ops,max_steps,policy,proposal_cap;
    int64_t address_lo,address_hi;
    double finite_bound,salient_temp;
} Config;
typedef struct {int used,pinned;int64_t addr,born,last_read; double v[4];} Entity;
typedef struct {int used,rid,protected_record;int64_t participant,born,last_read,utility_tick;
    double v[4],utility;} Relation;
typedef struct {uint64_t identity; int committed;} ProposalToken;
typedef struct {int op,status,code;int64_t tick;} Control;
typedef struct {int64_t token,logical;} Window;
typedef struct {int64_t addr,participant,tokens_len; int rid,pre_code,authorized,protected_record;
    uint64_t proposal_id;
    int64_t ticks; double v[4],co,feedback;
    const int64_t *tokens; const double *vectors;const uint64_t *proposal_ids; int n_proposals;
} Request;
typedef struct {int status,code;int64_t number;double v[4],v2[4],scalar;
    int count;uint64_t generation;} Result;
typedef struct {int entities,relations,control,window;int64_t tick,logical,evictions,rejections,reinforcements;
    int ops_this_tick,halted,issued;} Counters;
typedef struct {
    Config cfg;Entity *entities;Relation *relations;Control *control;Window *window;
    double *conv;ProposalToken *proposals;
    Counters x; uint64_t generation;pthread_mutex_t mutex;
} Bank;
static double aged(const Relation *r,int64_t tick){return r->utility*exp(-0.00035*(double)(tick-r->utility_tick));}
static void logctrl(Bank*b,int op,int st,int code){
    if(!b->cfg.control_cap)return;
    int n=b->x.control,cap=b->cfg.control_cap;
    if(n==cap){memmove(b->control,b->control+1,(size_t)(cap-1)*sizeof(Control));n--;}
    b->control[n]=(Control){op,st,code,b->x.tick};b->x.control=n+1;
}
static Result result(int st,int code){Result r={0};r.status=st;r.code=code;return r;}
static int valid_vec(const Config*c,const double*v){
    for(int i=0;i<4;i++)if(!isfinite(v[i])||fabs(v[i])>c->finite_bound)return 0;
    return 1;
}
static int entity_index(Bank*b,int64_t a){for(int i=0;i<b->cfg.entity_cap;i++)if(b->entities[i].used&&b->entities[i].addr==a)return i;return -1;}
static int relation_index(Bank*b,int rid){for(int i=0;i<b->cfg.relation_cap;i++)if(b->relations[i].used&&b->relations[i].rid==rid)return i;return -1;}
static int proposal_index(Bank*b,uint64_t id){for(int i=0;i<b->x.issued;i++)if(b->proposals[i].identity==id)return i;return -1;}
static int free_entity(Bank*b){for(int i=0;i<b->cfg.entity_cap;i++)if(!b->entities[i].used)return i;return -1;}
static int free_relation(Bank*b){for(int i=0;i<b->cfg.relation_cap;i++)if(!b->relations[i].used)return i;return -1;}
static void drop_relation(Bank*b,int i){b->relations[i].used=0;b->x.relations--;b->x.evictions++;}
static void drop_entity(Bank*b,int i){
    int64_t addr=b->entities[i].addr;b->entities[i].used=0;b->x.entities--;b->x.evictions++;
    for(int k=0;k<b->cfg.relation_cap;k++)if(b->relations[k].used&&b->relations[k].participant==addr)drop_relation(b,k);
}
static int is_referenced(Bank*b,int64_t addr){for(int j=0;j<b->cfg.relation_cap;j++)if(b->relations[j].used&&b->relations[j].participant==addr)return 1;return 0;}
static int victim_entity(Bank*b){
    int best=-1,prefer_orphan=0;
    for(int i=0;i<b->cfg.entity_cap;i++){
        Entity*e=&b->entities[i];if(!e->used||e->pinned)continue;
        int orphan=!is_referenced(b,e->addr);
        if(best<0||(orphan&&!prefer_orphan)||(orphan==prefer_orphan&&
            (e->last_read<b->entities[best].last_read||
            (e->last_read==b->entities[best].last_read&&(e->born<b->entities[best].born||
            (e->born==b->entities[best].born&&e->addr<b->entities[best].addr)))))){
            best=i;prefer_orphan=orphan;
        }
    }return best;
}
static int victim_relation(Bank*b){
    int best=-1,has_unprotected=0;
    if(b->cfg.policy==3){for(int i=0;i<b->cfg.relation_cap;i++)if(b->relations[i].used&&!b->relations[i].protected_record)has_unprotected=1;}
    for(int i=0;i<b->cfg.relation_cap;i++){
        Relation*r=&b->relations[i];if(!r->used||(has_unprotected&&r->protected_record))continue;
        if(best<0){best=i;continue;}
        Relation*q=&b->relations[best];int less=0;
        if(b->cfg.policy==0){less=(r->born<q->born||(r->born==q->born&&r->rid<q->rid));}
        else if(b->cfg.policy==1){less=(r->last_read<q->last_read||(r->last_read==q->last_read&&
                (r->born<q->born||(r->born==q->born&&r->rid<q->rid))));}
        else {double a=aged(r,b->x.tick),c=aged(q,b->x.tick);
            less=(a<c||(a==c&&(r->born<q->born||(r->born==q->born&&r->rid<q->rid))));}
        if(less)best=i;
    }return best;
}
Bank*ersa11_new(const Config *c){
    if(!c||c->entity_cap<1||c->entity_cap>8192||c->relation_cap<1||c->relation_cap>8192||
       c->control_cap<0||c->control_cap>65536||c->window_cap<0||c->window_cap>65536||
       c->conv_slots<0||c->conv_dim<0||(int64_t)c->conv_slots*c->conv_dim>65536||
       c->max_ops<0||c->max_steps<0||c->policy<0||c->policy>3||
       c->proposal_cap<1||c->proposal_cap>8192||c->address_lo>c->address_hi||
       !(c->finite_bound>0&&isfinite(c->finite_bound))||
       !(c->salient_temp>0&&isfinite(c->salient_temp)))return NULL;
    Bank*b=calloc(1,sizeof(*b));if(!b)return NULL;
    b->cfg=*c;
    b->entities=calloc((size_t)c->entity_cap,sizeof(Entity));
    b->relations=calloc((size_t)c->relation_cap,sizeof(Relation));
    b->control=calloc((size_t)c->control_cap+1,sizeof(Control));
    b->window=calloc((size_t)c->window_cap+1,sizeof(Window));
    b->conv=calloc((size_t)c->conv_slots*c->conv_dim+1,sizeof(double));
    b->proposals=calloc((size_t)c->proposal_cap,sizeof(ProposalToken));
    if(!b->entities||!b->relations||!b->control||!b->window||!b->conv||!b->proposals||
       pthread_mutex_init(&b->mutex,NULL)){
        free(b->entities);free(b->relations);free(b->control);free(b->window);free(b->conv);free(b->proposals);free(b);return NULL;
    }return b;
}
void ersa11_free(Bank*b){if(!b)return;pthread_mutex_lock(&b->mutex);
    if(b->entities)memset(b->entities,0,(size_t)b->cfg.entity_cap*sizeof(Entity));
    if(b->relations)memset(b->relations,0,(size_t)b->cfg.relation_cap*sizeof(Relation));
    if(b->proposals)memset(b->proposals,0,(size_t)b->cfg.proposal_cap*sizeof(ProposalToken));
    free(b->entities);free(b->relations);free(b->control);free(b->window);free(b->conv);free(b->proposals);
    pthread_mutex_unlock(&b->mutex);pthread_mutex_destroy(&b->mutex);memset(b,0,sizeof(*b));free(b);
}
int ersa11_has_entity(Bank*b,int64_t addr){if(!b)return 0;pthread_mutex_lock(&b->mutex);int v=entity_index(b,addr)>=0;pthread_mutex_unlock(&b->mutex);return v;}
int64_t ersa11_tick(Bank*b){if(!b)return -1;pthread_mutex_lock(&b->mutex);int64_t v=b->x.tick;pthread_mutex_unlock(&b->mutex);return v;}
uint64_t ersa11_generation(Bank*b){if(!b)return 0;pthread_mutex_lock(&b->mutex);uint64_t g=b->generation;pthread_mutex_unlock(&b->mutex);return g;}
int ersa11_count(Bank*b,Counters*out){if(!b||!out)return 0;pthread_mutex_lock(&b->mutex);*out=b->x;pthread_mutex_unlock(&b->mutex);return 1;}
int ersa11_entity(Bank*b,int i,Entity*out){if(!b||!out||i<0||i>=b->cfg.entity_cap)return 0;pthread_mutex_lock(&b->mutex);*out=b->entities[i];pthread_mutex_unlock(&b->mutex);return 1;}
int ersa11_relation(Bank*b,int i,Relation*out){if(!b||!out||i<0||i>=b->cfg.relation_cap)return 0;pthread_mutex_lock(&b->mutex);*out=b->relations[i];pthread_mutex_unlock(&b->mutex);return 1;}
int ersa11_control(Bank*b,int i,Control*out){if(!b||!out||i<0||i>=b->cfg.control_cap)return 0;pthread_mutex_lock(&b->mutex);*out=b->control[i];pthread_mutex_unlock(&b->mutex);return 1;}
int ersa11_window(Bank*b,int i,Window*out){if(!b||!out||i<0||i>=b->cfg.window_cap)return 0;pthread_mutex_lock(&b->mutex);*out=b->window[i];pthread_mutex_unlock(&b->mutex);return 1;}
int ersa11_conv(Bank*b,int i,double*out){if(!b||!out||i<0||i>=b->cfg.conv_slots*b->cfg.conv_dim)return 0;pthread_mutex_lock(&b->mutex);*out=b->conv[i];pthread_mutex_unlock(&b->mutex);return 1;}
/* Return 0 for successful execution, 1 for stale CAS, 2 for invalid input,
 * 3 for generation overflow. The result contains a contract receipt. */
int ersa11_run(Bank*b,uint64_t expected,int method,const Request*q,Result*out){
    if(!b||!q||!out||method<0||method>HALT)return 2;
    pthread_mutex_lock(&b->mutex);
    if(expected!=b->generation){pthread_mutex_unlock(&b->mutex);return 1;}
    if(b->generation==UINT64_MAX){pthread_mutex_unlock(&b->mutex);return 3;}
    Result r=result(OK,0);Config*c=&b->cfg;Counters*s=&b->x;
    if(method==BEGIN||method==RESET){
        memset(b->entities,0,(size_t)c->entity_cap*sizeof(Entity));
        memset(b->relations,0,(size_t)c->relation_cap*sizeof(Relation));
        memset(b->control,0,(size_t)c->control_cap*sizeof(Control));
        memset(b->window,0,(size_t)c->window_cap*sizeof(Window));
        memset(b->conv,0,(size_t)c->conv_slots*c->conv_dim*sizeof(double));
        memset(b->proposals,0,(size_t)c->proposal_cap*sizeof(ProposalToken));
        memset(s,0,sizeof(*s));goto done;
    }
    if(method==HALT){s->halted=1;logctrl(b,method,OK,0);goto done;}
    if((method==STEP||method==DECAY)&&(q->ticks<0||q->ticks>c->max_steps||q->pre_code)){
        r=result(REJECTED,STEP_BUDGET);goto done;
    }
    if(s->ops_this_tick>=c->max_ops){s->rejections++;r=result(REJECTED,OP_BUDGET);goto done;}
    s->ops_this_tick++;
    switch(method){
    case STEP:case DECAY:
        if(q->ticks>INT64_MAX-s->tick){r=result(REJECTED,STEP_BUDGET);break;}
        s->tick+=q->ticks;s->ops_this_tick=0;logctrl(b,method,OK,0);break;
    case OBSERVE:
        if(q->pre_code){r=result(REJECTED,q->pre_code);break;}
        if(q->tokens_len<=0||q->tokens_len>c->max_steps||!q->tokens){r=result(REJECTED,INVALID_VALUE);break;}
        for(int64_t i=0;i<q->tokens_len;i++)if(q->tokens[i]<0){r=result(REJECTED,INVALID_VALUE);goto done;}
        if(q->tokens_len>INT64_MAX-s->tick||q->tokens_len>INT64_MAX-s->logical){r=result(REJECTED,STEP_BUDGET);break;}
        for(int64_t i=0;i<q->tokens_len;i++){
            int64_t tok=q->tokens[i];s->logical++;s->tick++;s->ops_this_tick=0;
            if(c->window_cap){
                int j=s->window;if(j==c->window_cap){memmove(b->window,b->window+1,(size_t)(j-1)*sizeof(Window));j--;}
                b->window[j]=(Window){tok,s->logical};s->window=j+1;
            }
            for(int k=0;k<c->conv_slots;k++){
                double*base=b->conv+(size_t)k*c->conv_dim;
                if(c->conv_dim){memmove(base,base+1,(size_t)(c->conv_dim-1)*sizeof(double));
                    base[c->conv_dim-1]=(double)((tok%13+(7*(int64_t)k)%13)%13)/13.0;}
            }
        }r.number=s->logical;logctrl(b,method,OK,0);break;
    case READ_EXACT:case UPSERT:{
        if(q->pre_code){r=result(REJECTED,q->pre_code);break;}
        if(q->addr<c->address_lo||q->addr>c->address_hi){r=result(REJECTED,INVALID_ADDRESS);break;}
        int i=entity_index(b,q->addr);
        if(method==READ_EXACT){if(i<0){r=result(NOT_FOUND,0);break;}
            memcpy(r.v,b->entities[i].v,sizeof(r.v));b->entities[i].last_read=s->tick;
            logctrl(b,method,OK,0);break;}
        if(!q->authorized){s->rejections++;r=result(REJECTED,UNAUTHORIZED_WRITE);logctrl(b,method,REJECTED,r.code);break;}
        if(!valid_vec(c,q->v)){r=result(REJECTED,NONFINITE_VALUE);break;}
        if(i<0){i=free_entity(b);if(i<0){i=victim_entity(b);
                if(i<0){r=result(REJECTED,CAPACITY_EXCEEDED);break;}drop_entity(b,i);}
            b->entities[i].addr=q->addr;b->entities[i].born=s->tick;
            b->entities[i].pinned=0;b->entities[i].used=1;s->entities++;}
        memcpy(b->entities[i].v,q->v,sizeof(r.v));b->entities[i].last_read=s->tick;
        logctrl(b,method,OK,0);break;
    }
    case READ_ASSOC:{
        if(q->pre_code){r=result(REJECTED,q->pre_code);break;}
        int best=-1,hits=0;double mx=0;
        for(int i=0;i<c->relation_cap;i++){
            Relation*rel=&b->relations[i];if(!rel->used||rel->participant!=q->participant)continue;
            hits++;double v=aged(rel,s->tick);
            if(best<0||v>mx||(v==mx&&rel->rid<b->relations[best].rid)){best=i;mx=v;}
        }
        if(best<0){r=result(NOT_FOUND,0);break;}
        Relation *rel=&b->relations[best];rel->last_read=s->tick;
        memcpy(r.v,rel->v,sizeof(r.v));r.scalar=mx;r.count=hits;logctrl(b,method,OK,0);break;
    }
    case PROPOSE:{
        if(q->pre_code){s->rejections++;r=result(REJECTED,q->pre_code);logctrl(b,method,REJECTED,r.code);break;}
        int at=proposal_index(b,q->proposal_id);
        if(at<0){if(s->issued==c->proposal_cap){r=result(REJECTED,CAPACITY_EXCEEDED);break;}
            b->proposals[s->issued]=(ProposalToken){q->proposal_id,0};s->issued++;}
        r=result(PENDING,0);logctrl(b,method,PENDING,0);break;
    }
    case APPLY:case AGGREGATE:{
        if(method==AGGREGATE && q->pre_code==INVALID_VALUE && q->n_proposals==0){
            r=result(REJECTED,INVALID_VALUE);break;
        }
        if(q->pre_code==STALE_PROPOSAL){r=result(REJECTED,STALE_PROPOSAL);break;}
        int at=proposal_index(b,q->proposal_id);
        if(at<0){r=result(REJECTED,STALE_PROPOSAL);break;}
        if(method==APPLY&&b->proposals[at].committed){s->rejections++;r=result(REJECTED,DUPLICATE_IDENTITY);logctrl(b,method,REJECTED,r.code);break;}
        if(q->pre_code){s->rejections++;r=result(REJECTED,q->pre_code);logctrl(b,method,REJECTED,r.code);break;}
        if(method==APPLY){
            if(q->rid<0||!valid_vec(c,q->v)){r=result(REJECTED,NONFINITE_VALUE);break;}
            int i=relation_index(b,q->rid);
            if(i<0){i=free_relation(b);if(i<0){i=victim_relation(b);
                    if(i<0){r=result(REJECTED,CAPACITY_EXCEEDED);break;}drop_relation(b,i);}
                b->relations[i].used=1;b->relations[i].rid=q->rid;
                b->relations[i].participant=q->participant;b->relations[i].protected_record=q->protected_record;
                b->relations[i].born=s->tick;b->relations[i].utility=0.020;
                b->relations[i].utility_tick=s->tick;s->relations++;}
            memcpy(b->relations[i].v,q->v,sizeof(r.v));b->relations[i].last_read=s->tick;
            b->proposals[at].committed=1;logctrl(b,method,OK,0);break;
        }
        if(q->n_proposals<=0||q->n_proposals>c->proposal_cap||!q->proposal_ids||!q->vectors){r=result(REJECTED,INVALID_VALUE);break;}
        for(int j=0;j<q->n_proposals;j++){
            if(proposal_index(b,q->proposal_ids[j])<0){r=result(REJECTED,STALE_PROPOSAL);goto done;}
            if(!valid_vec(c,q->vectors+4*j)){r=result(REJECTED,NONFINITE_VALUE);goto done;}
        }
        {
            double peak=-INFINITY;
            for(int j=0;j<q->n_proposals;j++){
                const double *v=q->vectors+4*j;
                double score=c->salient_temp*hypot(hypot(v[0],v[1]),hypot(v[2],v[3]));
                if(!isfinite(score)){r=result(REJECTED,NONFINITE_VALUE);goto done;}
                if(score>peak)peak=score;
            }
            double sum=0;
            for(int j=0;j<q->n_proposals;j++){
                const double *v=q->vectors+4*j;
                double score=c->salient_temp*hypot(hypot(v[0],v[1]),hypot(v[2],v[3]));
                double w=exp(score-peak);sum+=w;
                for(int t=0;t<4;t++){r.v[t]+=w*v[t];r.v2[t]+=v[t];}
            }
            for(int t=0;t<4;t++){
                r.v[t]/=sum;r.v2[t]/=q->n_proposals;
                if(!isfinite(r.v[t])||!isfinite(r.v2[t])){
                    r=result(REJECTED,NONFINITE_VALUE);goto done;
                }
            }
        }
        logctrl(b,method,OK,0);break;
    }
    case REINFORCE:{
        if(q->pre_code){r=result(REJECTED,q->pre_code);break;}
        if(!isfinite(q->co)||!isfinite(q->feedback)||fabs(q->co)>1||fabs(q->feedback)>1){r=result(REJECTED,INVALID_VALUE);break;}
        int i=relation_index(b,q->rid);if(i<0){r=result(NOT_FOUND,0);break;}
        Relation *rel=&b->relations[i];s->reinforcements++;rel->last_read=s->tick;
        if(c->policy>=2){double v=aged(rel,s->tick)+.20*fmax(q->co,0.);
            if(c->policy==3){v+=.34*q->feedback;if(q->feedback<0)rel->protected_record=0;}
            rel->utility=fmax(0.,fmin(1.,v));rel->utility_tick=s->tick;}
        logctrl(b,method,OK,0);break;
    }
    case EVICT:{
        int i=relation_index(b,q->rid);if(i<0){r=result(NOT_FOUND,0);break;}
        if(c->policy==3&&b->relations[i].protected_record){r=result(REJECTED,PROTECTED_RECORD);break;}
        drop_relation(b,i);logctrl(b,method,OK,0);break;
    }
    default:r=result(REJECTED,INVALID_VALUE);break;
    }
done:
    b->generation++;r.generation=b->generation;*out=r;
    pthread_mutex_unlock(&b->mutex);return 0;
}
