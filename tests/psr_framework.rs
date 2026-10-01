use ps5_shader::parse_shader;

// Regression tests for the `sce::Psr` framework lowering family:
// parsing the Psr namespace (enums/structs/prototypes) instead of
// skipping it, caller-scope evaluation of identifier template args,
// range-typed enumerators, C functional casts, and prototype
// template-value draining. These jointly unblock the PSR `run`-style
// samples (`run<flags>` with `TraversalFlags const flags`).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

// `sce::Psr::TraversalFlags const flags = ...kUseSharedStack` must type
// the local as `int` (enum typedef) and `run<flags>` must monomorphize
// to a concrete clone (caller-scope constant evaluation).
const SCOPED_ENUM_TEMPLATE_ARG: &str = "namespace sce { namespace Psr {\n\
     enum class TraversalFlags : ulong { kNone = 0, kUseSharedStack = 16 };\n\
     } }\n\
     template <uint64_t flags> void run(int2 idx) { }\n\
     [NUM_THREADS(8,8,1)]\n\
     void mainBase(int2 idx : S_DISPATCH_THREAD_ID) {\n\
     sce::Psr::TraversalFlags const flags = sce::Psr::TraversalFlags::kUseSharedStack;\n\
     run<flags>(idx);\n\
     }\n";

#[test]
fn scoped_enum_type_and_value_arg_monomorphize() {
    let hlsl = emit(SCOPED_ENUM_TEMPLATE_ARG);
    assert!(
        hlsl.contains("run__mono"),
        "run<flags> must instantiate, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("run<flags>"),
        "mangled spelling must not leak, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "qualified Psr names must not leak, got:\n{hlsl}"
    );
}

// `ulong`-backed enumerators over int32 range (`kValue = 0xffffffff`)
// must emit as `uint` with a `u` suffix: `static const int` would make
// DXC truncate the value to 0 (silent miscompile).
const BIG_ENUMERATOR: &str = "namespace sce { namespace Psr {\n\
     enum class HitInvalid : unsigned int { kValue = 0xffffffffu };\n\
     } }\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint v = unsigned(sce::Psr::HitInvalid::kValue);\n\
     outBuf[0] = v;\n\
     }\n";

#[test]
fn big_enumerator_emits_uint_with_suffix() {
    let hlsl = emit(BIG_ENUMERATOR);
    assert!(
        hlsl.contains("static const uint kValue = 4294967295u;"),
        "kValue must be uint-suffixed, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint(kValue)") || hlsl.contains("uint(4294967295"),
        "unsigned() cast must lower to uint(), got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("unsigned("),
        "bare unsigned() must not leak, got:\n{hlsl}"
    );
}

// `template <BvhFormat fmt>` on a body-less prototype must not leak its
// value-param into the next parsed body as `uint fmt = 0;`.
const PROTOTYPE_VALUE_DRAIN: &str = "enum class BvhFormat : unsigned int { kBase = 0 };\n\
     template <BvhFormat fmt> uint proto(uint x);\n\
     template <BvhFormat fmt> uint proto2(uint x);\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     outBuf[0] = i;\n\
     }\n";

#[test]
fn prototype_template_values_do_not_leak() {
    let hlsl = emit(PROTOTYPE_VALUE_DRAIN);
    assert!(
        !hlsl.contains("uint fmt"),
        "drained prototype params must not appear, got:\n{hlsl}"
    );
}

// Chained constants (`optimizedFlags = flags | kA | kB`) must fold
// through locals so nested template-ids resolve (`g<19>`).
const CHAINED_FLAG_FOLD: &str = "namespace sce { namespace Psr {\n\
     enum class TraversalFlags : ulong { kA = 1, kB = 2, kBase = 16 };\n\
     } }\n\
     template <uint64_t flags> void g(int2 idx) { }\n\
     [NUM_THREADS(8,8,1)]\n\
     void mainBase(int2 idx : S_DISPATCH_THREAD_ID) {\n\
     sce::Psr::TraversalFlags const flags = sce::Psr::TraversalFlags::kBase;\n\
     sce::Psr::TraversalFlags const both = flags | sce::Psr::TraversalFlags::kA | sce::Psr::TraversalFlags::kB;\n\
     g<both>(idx);\n\
     }\n";

#[test]
fn chained_flag_expression_folds_to_literal() {
    let hlsl = emit(CHAINED_FLAG_FOLD);
    assert!(
        hlsl.contains("g__mono"),
        "g<both> must instantiate via chained fold, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("g<both>"),
        "unresolved spelling must not leak, got:\n{hlsl}"
    );
}

// `static constexpr const bool` (triple qualifier) must parse: the
// declaration-lookahead used to miss token-spelled `const`/`static`
// after `constexpr` and desynced the struct parser (psr `type_equal`).
const TRIPLE_QUALIFIER: &str = "template <typename T, typename U> struct type_equal {\n\
     static constexpr const bool value = false;\n\
     };\n\
     template <typename T> struct type_equal2 {\n\
     static constexpr const bool value = true;\n\
     };\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     outBuf[0] = i;\n\
     }\n";

#[test]
fn triple_qualifier_member_parses() {
    let hlsl = emit(TRIPLE_QUALIFIER);
    assert!(
        hlsl.contains("outBuf[0]"),
        "struct with triple-qualifier member must parse, got:\n{hlsl}"
    );
}

// Adjacent string literals concatenate (`__error(false, "A" "B")`).
const ADJACENT_STRINGS: &str = "void log(string s) {}\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     log(\"alpha \" \"beta\");\n\
     outBuf[0] = i;\n\
     }\n";

#[test]
fn adjacent_string_literals_concatenate() {
    let module = parse_shader(ADJACENT_STRINGS).expect("parser accepted shader");
    let found = module
        .functions
        .iter()
        .flat_map(|f| f.body.iter())
        .any(|s| format!("{s:?}").contains("alpha beta"));
    assert!(found, "adjacent strings must fold to one literal");
}

// Leading `[...]` attributes on struct members skip (`[isolate]`).
const MEMBER_ATTRIBUTE: &str = "struct Q {\n\
     uint m;\n\
     [isolate] void push(uint v) { m = v; }\n\
     };\n\
     Q q;\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     q.push(i);\n\
     outBuf[0] = q.m;\n\
     }\n";

#[test]
fn member_attribute_skips() {
    let hlsl = emit(MEMBER_ATTRIBUTE);
    assert!(
        hlsl.contains("push(i)"),
        "attributed method must survive, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("[isolate]"),
        "attribute must not leak to HLSL, got:\n{hlsl}"
    );
}

// Variadic ergonomics macros: `EMPTY(...)` drops, `PASS(...)`
// splices args through (Sony `IF_PSR_CPU` / `IF_PSR_GPU`).
const VARIADIC_MACROS: &str = "#define EMPTY(...)\n\
     #define PASS(...) __VA_ARGS__\n\
     static bool EMPTY(constexpr) s_flag = true;\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     outBuf[PASS(i)] = i;\n\
     }\n";

#[test]
fn variadic_empty_and_passthrough_expand() {
    // `#define`s live in the preprocessor: expand first, then parse.
    let expanded =
        ps5_shader::preprocess::preprocess(VARIADIC_MACROS, None).expect("preprocess succeeded");
    assert!(
        !expanded.contains("EMPTY(") && !expanded.contains("PASS("),
        "variadic macros must expand away, got:\n{expanded}"
    );
    let hlsl = emit(&expanded);
    assert!(
        !hlsl.contains("EMPTY") && !hlsl.contains("PASS"),
        "no macro residue may leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("outBuf[i]"),
        "passthrough must splice args, got:\n{hlsl}"
    );
}

// Prototyped methods (declared `;` with no body, e.g. psr `BvhNode`
// helpers) must not emit as `{}`: a non-void empty body is a hard DXC
// error, while dropping the uncalled method is exactly right.
const PROTOTYPE_METHOD: &str = "struct Node {\n\
     uint x;\n\
     float3 getVertex(uint a, uint b) const;\n\
     };\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     Node n; n.x = i;\n\
     outBuf[0] = n.x;\n\
     }\n";

#[test]
fn prototype_method_drops() {
    let hlsl = emit(PROTOTYPE_METHOD);
    assert!(
        !hlsl.contains("getVertex"),
        "prototyped method must not emit, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("outBuf[0]"),
        "shader must survive, got:\n{hlsl}"
    );
}

// Mixed explicit template args (`f<T, false>`) bind positionally with
// kind agreement: the type binds the typename param, the literal binds
// the value param (psr `updateAccumBuffer<FastTraceSampleSrt, false>`).
// The bound value also substitutes into `if constexpr` conditions so
// the untaken branch folds away.
const MIXED_TEMPLATE_ARGS: &str = "struct Srt { RWBuffer<uint> buf; uint k; };\n\
     Srt srt : S_SRT_DATA;\n\
     template <typename T, bool flag = true> uint fetch(T const *s, uint i) {\n\
     if constexpr (flag) { return s->buf[i]; }\n\
     return s->k;\n\
     }\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     outBuf[0] = fetch<Srt, false>(srt, i);\n\
     }\n";

#[test]
fn mixed_type_and_value_args_bind() {
    let hlsl = emit(MIXED_TEMPLATE_ARGS);
    assert!(
        hlsl.contains("fetch__mono"),
        "mixed args must instantiate, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("fetch<Srt"),
        "mangled spelling must not leak, got:\n{hlsl}"
    );
}

// Non-call structured bindings (`auto [a, b] = buf[i]`) desugar through
// the buffer element struct when it provably matches the name count.
const NONCALL_DESTRUCTURE: &str = "struct Pair { float x; float y; };\n\
     StructuredBuffer<Pair> buf;\n\
     RWBuffer<float> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     auto const [a, b] = buf[i];\n\
     outBuf[0] = a + b;\n\
     }\n";

#[test]
fn noncall_struct_destructure_desugars() {
    let hlsl = emit(NONCALL_DESTRUCTURE);
    assert!(
        hlsl.contains("__destructure"),
        "buffer-element destructure must materialize, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("outBuf[0]"),
        "uses must resolve, got:\n{hlsl}"
    );
}

// A colliding `typedef` elsewhere must never hijack a qualified
// struct path: `sce::Psr::Hit` is the struct even with a global
// `typedef uint Hit;` in scope (psr three-level BVH headers nest one
// inside a struct).
const TYPEDEF_STRUCT_COLLISION: &str = "namespace sce { namespace Psr {\n\
     struct Hit { float m_depth; uint m_instanceID; };\n\
     } }\n\
     typedef uint Hit;\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     sce::Psr::Hit h;\n\
     h.m_depth = 1.0;\n\
     h.m_instanceID = i;\n\
     outBuf[0] = h.m_instanceID;\n\
     }\n";

#[test]
fn qualified_struct_beats_colliding_typedef() {
    let hlsl = emit(TYPEDEF_STRUCT_COLLISION);
    assert!(
        !hlsl.contains("sce::"),
        "qualified path must fold to the struct, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("h.m_instanceID"),
        "member use must resolve through the struct, got:\n{hlsl}"
    );
}

// Functional casts through enum/typedef names (`Weapon(x)`) lower as
// the underlying scalar instead of degrading to undeclared calls.
const ENUM_FUNCTIONAL_CAST: &str = "enum class Weapon : uint { kA, kB };\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint w = Weapon(i % 2);\n\
     outBuf[0] = w;\n\
     }\n";

#[test]
fn enum_functional_cast_lowers_to_scalar() {
    let hlsl = emit(ENUM_FUNCTIONAL_CAST);
    assert!(
        !hlsl.contains("Weapon("),
        "enum cast must not leak as a call, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("outBuf[0]"),
        "shader must survive, got:\n{hlsl}"
    );
}

// A local lambda must shadow an unrelated struct method of the same
// name: `auto incr = [...](...)` + `x + incr(k)` must not resolve
// `incr` to some other struct's `void incr()` (which poisons the
// arithmetic with `void`). psr low_level `decompress_index` hit this.
const LOCAL_SHADOWS_METHOD: &str = "struct Stack { void incr() {} };\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     auto const incr = [](uint k) { return k + 1; };\n\
     uint b = i + incr(i);\n\
     outBuf[0] = b;\n\
     }\n";

#[test]
fn local_lambda_shadows_unrelated_method() {
    let hlsl = emit(LOCAL_SHADOWS_METHOD);
    assert!(
        hlsl.contains("outBuf[0]"),
        "lambda call must type-check; shader must survive, got:\n{hlsl}"
    );
}

// Scoped enum constants in value template args fold to their integer
// (`run<RenderingMode::kLoop, false>` → `run<0, false>`), so the callee
// monomorphizes (psr many_lights family).
const SCOPED_ENUM_VALUE_ARG: &str = "enum class RenderingMode : uint { kLoop, kStack };\n\
     RWBuffer<uint> outBuf;\n\
     template <uint mode, bool trinity> uint run(uint i) { if (mode == 0 && !trinity) { return i; } return 0; }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     outBuf[0] = run<RenderingMode::kLoop, false>(i);\n\
     }\n";

#[test]
fn scoped_enum_value_arg_monomorphizes() {
    let hlsl = emit(SCOPED_ENUM_VALUE_ARG);
    assert!(
        hlsl.contains("run__mono"),
        "run<RenderingMode::kLoop,false> must instantiate, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("RenderingMode::kLoop"),
        "scoped enum spelling must fold, got:\n{hlsl}"
    );
}
