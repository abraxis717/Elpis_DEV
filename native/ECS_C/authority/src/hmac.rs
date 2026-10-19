//! HMAC-SHA256 (RFC 2104) over the continuity crate's SHA-256; checked against the RFC 4231 vectors.

use elpis_continuity::sha256::{self, Sha256};

const BLOCK: usize = 64;

/// HMAC-SHA256 of the concatenation of `parts` under `key` (a key longer than one block is hashed first).
pub fn hmac(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        k[..32].copy_from_slice(&sha256::digest(&[key]));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(&ipad);
    for part in parts {
        inner.update(part);
    }
    let inner = inner.finish();
    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner);
    wipe(&mut k);
    wipe(&mut ipad);
    wipe(&mut opad);
    outer.finish()
}

/// Equality that does not stop at the first differing byte.
pub fn equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Overwrite key material before it is released.
pub fn wipe(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        // SAFETY: `b` is a valid, aligned, exclusive reference to one byte.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::hmac;

    fn hex(d: [u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn rfc4231_vectors() {
        // Test case 1.
        assert_eq!(
            hex(hmac(&[0x0b; 20], &[b"Hi There"])),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        // Test case 2.
        assert_eq!(
            hex(hmac(b"Jefe", &[b"what do ya want ", b"for nothing?"])),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Test case 3.
        assert_eq!(
            hex(hmac(&[0xaa; 20], &[&[0xdd; 50]])),
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"
        );
        // Test case 6: a key longer than one block.
        assert_eq!(
            hex(hmac(&[0xaa; 131], &[b"Test Using Larger Than Block-Size Key - Hash Key First"])),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
        // Test case 7.
        let data: &[u8] = b"This is a test using a larger than block-size key and a larger than block-size data. \
The key needs to be hashed before being used by the HMAC algorithm.";
        assert_eq!(
            hex(hmac(&[0xaa; 131], &[data])),
            "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
        );
    }
}
