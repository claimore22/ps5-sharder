use ps5_shader::parse_shader;
fn main() {
    let src = r#"
enum class LightType : uint
{
	kDirectional = 0,
	kPoint = 1,
	kSpot = 2
};
struct Foo { int x; };
"#;
    match parse_shader(src) {
        Ok(m) => println!("ok: {} structs", m.structs.len()),
        Err(e) => println!("err: {}", e),
    }
}
