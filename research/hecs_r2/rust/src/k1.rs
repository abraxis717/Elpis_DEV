//! The qualified native ECS K1 primitive, reached only through its existing C ABI
//! (native/ECS/include/elpis/ecsg_k1.h). The library is loaded by explicit path at run time; nothing here
//! re-implements or alters the ECS mathematics. Each hierarchy level owns one K1 state (W, epoch, H, a).

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

type Handle = *mut c_void;
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct Transition {
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub generation_before: u64,
    pub generation_after: u64,
    pub steps: u64,
    pub failed_step: u64,
}
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct Counters {
    pub workspace_bytes: usize,
    pub max_rows: usize,
    pub heap_allocations: u64,
    pub forward_calls: u64,
    pub learn_calls: u64,
    pub steps_executed: u64,
    pub corrected_steps: u64,
    pub consolidations: u64,
    pub commits: u64,
    pub refusals: u64,
    pub txn_begins: u64,
    pub txn_aborts: u64,
    pub stale_refusals: u64,
    pub busy_refusals: u64,
}

struct Api {
    abi_version: unsafe extern "C" fn() -> u32,
    image_bytes: unsafe extern "C" fn(usize, usize) -> usize,
    create: unsafe extern "C" fn(usize, usize, usize, *const f64, *mut Handle) -> c_int,
    destroy: unsafe extern "C" fn(*mut Handle) -> c_int,
    forward: unsafe extern "C" fn(Handle, *const f64, usize, *mut f64) -> c_int,
    learn: unsafe extern "C" fn(Handle, *const f64, *const f64, usize, f64, u64, *mut Transition) -> c_int,
    epoch: unsafe extern "C" fn(Handle) -> u64,
    copy_w: unsafe extern "C" fn(Handle, *mut f64, usize) -> c_int,
    state_digest: unsafe extern "C" fn(Handle, *mut u8) -> c_int,
    stats: unsafe extern "C" fn(Handle, *mut Counters) -> c_int,
}

/// The loaded K1 library.
pub struct K1 {
    api: Api,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct K1Error(pub i32);

impl std::fmt::Display for K1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self.0 {
            -1 => "INVALID",
            -2 => "NONFINITE",
            -3 => "STALE",
            -4 => "BUSY",
            -5 => "CAPACITY",
            -6 => "NOMEM",
            -7 => "CORRUPT",
            _ => "UNKNOWN",
        };
        write!(f, "K1 {name} ({})", self.0)
    }
}

fn check(rc: c_int) -> Result<(), K1Error> {
    if rc == 0 {
        Ok(())
    } else {
        Err(K1Error(rc))
    }
}

impl K1 {
    pub fn load(path: &Path) -> Result<K1, String> {
        let c = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        let lib = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        if lib.is_null() {
            return Err(format!("cannot load K1 library {}", path.display()));
        }
        unsafe fn sym<T>(lib: *mut c_void, name: &str) -> Result<T, String> {
            let c = CString::new(name).unwrap();
            let p = dlsym(lib, c.as_ptr());
            if p.is_null() {
                return Err(format!("missing K1 symbol {name}"));
            }
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&p))
        }
        let api = unsafe {
            Api {
                abi_version: sym(lib, "elpis_ecsg_k1_abi_version")?,
                image_bytes: sym(lib, "elpis_ecsg_k1_image_bytes")?,
                create: sym(lib, "elpis_ecsg_k1_create")?,
                destroy: sym(lib, "elpis_ecsg_k1_destroy")?,
                forward: sym(lib, "elpis_ecsg_k1_forward")?,
                learn: sym(lib, "elpis_ecsg_k1_learn")?,
                epoch: sym(lib, "elpis_ecsg_k1_epoch")?,
                copy_w: sym(lib, "elpis_ecsg_k1_copy_w")?,
                state_digest: sym(lib, "elpis_ecsg_k1_state_digest")?,
                stats: sym(lib, "elpis_ecsg_k1_stats")?,
            }
        };
        Ok(K1 { api })
    }

    pub fn abi_version(&self) -> u32 {
        unsafe { (self.api.abi_version)() }
    }

    /// Bytes of one retained image (header + W + H packed + a) for a shape.
    pub fn image_bytes(&self, dim: usize, width: usize) -> usize {
        unsafe { (self.api.image_bytes)(dim, width) }
    }

    pub fn create<'a>(&'a self, dim: usize, width: usize, max_rows: usize, w: &[f64]) -> Result<K1State<'a>, K1Error> {
        assert_eq!(w.len(), dim * width);
        let mut h: Handle = std::ptr::null_mut();
        check(unsafe { (self.api.create)(dim, width, max_rows, w.as_ptr(), &mut h) })?;
        Ok(K1State { lib: self, handle: h, dim, width })
    }
}

/// One owned K1 state: the authoritative (W, epoch, H, a) of one hierarchy level.
pub struct K1State<'a> {
    lib: &'a K1,
    handle: Handle,
    pub dim: usize,
    pub width: usize,
}

impl Drop for K1State<'_> {
    fn drop(&mut self) {
        unsafe { (self.lib.api.destroy)(&mut self.handle) };
    }
}

impl K1State<'_> {
    /// QUERY: f_W(x_r) for each row (reads W only).
    pub fn forward(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), K1Error> {
        let rows = out.len();
        assert_eq!(x.len(), rows * self.dim);
        check(unsafe { (self.lib.api.forward)(self.handle, x.as_ptr(), rows, out.as_mut_ptr()) })
    }

    /// LEARN: `steps` qualified K1 steps on (x, y) as one committed transition.
    pub fn learn(&mut self, x: &[f64], y: &[f64], rate: f64, steps: u64) -> Result<Transition, K1Error> {
        assert_eq!(x.len(), y.len() * self.dim);
        let mut t = Transition::default();
        check(unsafe { (self.lib.api.learn)(self.handle, x.as_ptr(), y.as_ptr(), y.len(), rate, steps, &mut t) })?;
        Ok(t)
    }

    pub fn epoch(&self) -> u64 {
        unsafe { (self.lib.api.epoch)(self.handle) }
    }

    pub fn w(&mut self) -> Result<Vec<f64>, K1Error> {
        let mut w = vec![0.0; self.dim * self.width];
        check(unsafe { (self.lib.api.copy_w)(self.handle, w.as_mut_ptr(), w.len()) })?;
        Ok(w)
    }

    pub fn digest(&mut self) -> Result<[u8; 32], K1Error> {
        let mut d = [0u8; 32];
        check(unsafe { (self.lib.api.state_digest)(self.handle, d.as_mut_ptr()) })?;
        Ok(d)
    }

    pub fn counters(&mut self) -> Result<Counters, K1Error> {
        let mut c = Counters::default();
        check(unsafe { (self.lib.api.stats)(self.handle, &mut c) })?;
        Ok(c)
    }
}
