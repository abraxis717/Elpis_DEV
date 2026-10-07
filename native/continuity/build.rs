// The shared library carries a stable SONAME so C callers can link it by name.
fn main() {
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libelpis_continuity.so");
}
