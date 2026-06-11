//! Public-API surface snapshots for the PARENT package (docs/public-api/).
//! The nested `zenwebp-recompress/` workspace is separate and not covered.
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new().workspace_dir("..").run();
}
