from generator import build_split
from engines import HcaNativeSemantics
from hashlib import sha256

tr, tt = build_split("TRAIN", 12345, 4)
c = [x for w in tr["worlds"] for x in w["cases"] if x["case_id"] == "z30bf45debe0d336dcf7735faf62d"][0]
q = [x for x in c["queries"] if x["id"] == "z8b21e69e615aab06aaea0775b7c7"][0]
b2 = HcaNativeSemantics(c, method="B2", rounds=4)
i3 = b2.index[("zf94763159167d71c8f92a19e8c3f", "L3")]
i2 = b2.index[("z2ad33e2568d57d02d9a7804996f5", "L2")]
for n, e in enumerate(c["events"]):
    b2.apply(e)
    sc = b2.values[q["id"]]
    order = sorted(range(len(b2.cells)),
                   key=lambda i: (sha256(b2.SCHEDULE + b2.step.to_bytes(8, "big")
                                        + i.to_bytes(8, "big")).digest(), i))
    print(n, "L2", sc[i2], "L3", sc[i3], "pos2", order.index(i2), "pos3", order.index(i3))
