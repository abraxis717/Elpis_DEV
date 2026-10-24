"""Independent scratch recomputation oracle. Never imports generator or incremental engine."""

def all_prefix_labels(scene):
    qs=scene['queries']
    facts=scene['factors']
    events=scene['events']
    labels=[]
    for prefix in range(len(events)+1):
        status={}
        for ev in events[:prefix]:status[ev['source']]=(ev['status'],ev['epoch'])
        active=[]
        for edge in facts:
            if status.get(edge['source'])==('VALID',edge['version']):active.append(edge)
        outcomes={}
        for q in qs:
            head={e['roles']['to'] for e in active if e['rel']=='R0' and
                  e['roles']['from']==q['from'] and e['roles']['scope']==q['scope'] and
                  e['roles']['context']==q['context']}
            middle={e['roles']['to'] for e in active if e['rel']=='R1' and
                    e['roles']['from'] in head and e['roles']['scope']==q['scope'] and
                    e['roles']['context']==q['context']}
            found=any(e['rel']=='R2' and e['roles']['from'] in middle and
                      e['roles']['to']==q['to'] and e['roles']['scope']==q['scope'] and
                      e['roles']['context']==q['context'] for e in active)
            outcomes[q['id']]=int(found)
        labels.append(outcomes)
    return labels

def dataset_gold(ds):
    return {s['case_id']:all_prefix_labels(s) for w in ds['worlds'] for s in w['cases']}
