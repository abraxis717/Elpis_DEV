from dataclasses import replace
import numpy as np
import pytest
from elpis.inference.contracts import Bank,RowIdentity,ContractError
from elpis.inference.rows import RowEngine,RowTable,row_representation,decode_row


def make_table(provider):
    f,path,m,a=provider
    # Existing bytes are finite F32 values. Exact raw bytes, no zero substitution.
    bank=Bank('synthetic','tiny','fixture','DSV41_ENGRAM/v1','1'*64,1,8,4,m.content,
              row_representation(a,0,8,4,'F32_LE'))
    return RowTable(bank,a,0,'F32_LE')


@pytest.mark.parametrize('workers',[1,3])
def test_batch_sorted_deduplicated_restored(provider,workers):
    f,path,m,a=provider; table=make_table(provider)
    engine=RowEngine(f,table,expected_bank=table.bank.digest,workers=workers)
    rows=(7,1,7,0,2,1)
    got=engine.lookup(tuple(RowIdentity(table.bank.digest,r) for r in rows))
    expected=np.frombuffer(path.read_bytes(),dtype='<f4').reshape(8,4)[list(rows)]
    assert got.tobytes()==expected.tobytes()
    assert engine.last_metrics['unique_rows']==4
    assert engine.last_metrics['pread_bytes']==64
    assert engine.last_metrics['semantic_bytes']==96


def test_wrong_row_bank_and_output_budget(provider):
    f,*_=provider; table=make_table(provider)
    engine=RowEngine(f,table,expected_bank=table.bank.digest,max_rows=2)
    for request in [(RowIdentity('0'*64,1),),(RowIdentity(table.bank.digest,8),),
                    (RowIdentity(table.bank.digest,0),)*3]:
        with pytest.raises(ContractError): engine.lookup(request)
    with pytest.raises(ContractError):
        RowEngine(f,replace(table,offset=1),expected_bank=table.bank.digest)


def test_decode_rejects_nonfinite_and_invalid_encodings():
    for x in [float('nan'),float('inf'),-float('inf')]:
        with pytest.raises(ContractError,match='ENCODING'):
            decode_row(np.array([x],dtype='<f4').tobytes(),1,'F32_LE')
    for raw in [bytes([127])*32+b'\x7f',bytes(32)+b'\xff',bytes([126])*32+b'\xfe']:
        with pytest.raises(ContractError,match='ENCODING'):
            decode_row(raw,32,'DS4_E4M3_E8M0_BF16')
    assert np.array_equal(decode_row(bytes([56])*32+b'\x7f',32,'DS4_E4M3_E8M0_BF16'),np.ones(32))
