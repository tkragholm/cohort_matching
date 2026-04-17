#[test]
fn typed_api_rejects_raw_primitive_arguments() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
