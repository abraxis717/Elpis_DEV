//! The shared library exports the ECS_C authority C ABI only (include/elpis/ecs_c_authority.h).
//!
//! A cdylib re-exports every `#[no_mangle]` function of the crates it embeds, so without this it would export
//! the continuity crate's C ABI (`elpis_continuity_*`) a second time beside `libelpis_continuity.so`. Upstream
//! crates are linked from archives (rlibs), and `--exclude-libs ALL` keeps archive symbols out of the dynamic
//! symbol table; this crate's own `elpis_ecs_c_*` functions are unaffected.
//!
//! Only with the `c-abi` feature, the build of this crate's own library: Cargo also passes a dependency's cdylib link
//! arguments to the dependent's cdylib, where this would hide the dependent's re-exported upstream C ABIs (RuntimeCore,
//! which embeds this crate without `c-abi`, exports continuity's).

fn main() {
    if std::env::var_os("CARGO_FEATURE_C_ABI").is_some() {
        println!("cargo:rustc-cdylib-link-arg=-Wl,--exclude-libs,ALL");
    }
}
