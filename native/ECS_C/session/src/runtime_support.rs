//! The C ABI's runtime: a malloc-backed global allocator and an abort-on-panic handler (built only into
//! `libelpis_ecs_c_session.so`; an embedding Rust crate supplies its own std runtime instead).

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;
use core::ptr::null_mut;

#[link(name = "c")]
extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(p: *mut c_void);
    fn posix_memalign(out: *mut *mut c_void, alignment: usize, size: usize) -> i32;
    fn abort() -> !;
}

struct Malloc;

// SAFETY: forwards to the C library allocator; alignments above malloc's 16-byte guarantee use posix_memalign.
unsafe impl GlobalAlloc for Malloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= 16 {
            malloc(layout.size()) as *mut u8
        } else {
            let mut p: *mut c_void = null_mut();
            if posix_memalign(&mut p, layout.align(), layout.size()) == 0 {
                p as *mut u8
            } else {
                null_mut()
            }
        }
    }

    unsafe fn dealloc(&self, p: *mut u8, _layout: Layout) {
        free(p as *mut c_void)
    }
}

#[global_allocator]
static ALLOCATOR: Malloc = Malloc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // SAFETY: abort never returns and touches no Elpis state.
    unsafe { abort() }
}

/// Referenced by the precompiled `core`'s unwind tables. Never called: every panic aborts (panic = "abort").
#[no_mangle]
extern "C" fn rust_eh_personality() {}
