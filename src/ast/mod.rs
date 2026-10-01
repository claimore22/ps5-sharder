pub mod attribute;
pub mod expr;
pub mod function;
pub mod resource;
pub mod stmt;
pub mod structure;
pub mod types;

pub use attribute::Attribute;
pub use expr::{BinaryOp, Expr, UnaryOp};
pub use function::{Function, Parameter};
pub use resource::ResourceDecl;
pub use stmt::{SpannedStmt, Stmt, StorageClass, SwitchCase};
pub use structure::{FieldDecl, GlobalDecl, StructDecl};
pub use types::{Direction, ResourceAccess, ScalarType, Semantic, SpannedType, Type};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

/// The top-level AST node for an entire shader.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Module {
    pub attributes: Vec<Attribute>,
    pub structs: Vec<StructDecl>,
    pub globals: Vec<GlobalDecl>,
    pub resources: Vec<ResourceDecl>,
    pub functions: Vec<Function>,
    /// Named constant values (e.g. `static constexpr uint k_size = 8;`), used
    /// to resolve identifiers in attributes such as `[NUM_THREADS(k_size, ...)]`.
    pub constants: HashMap<String, Expr>,
    /// Template-id spellings on declarations (`BcnEncodeSrt<uint2> g_srt`):
    /// function name ("" for globals) -> variable name -> normalized
    /// spelling (`Name<args>` with whitespace stripped). Template angle
    /// lists erase during type parsing, so this table preserves the
    /// arguments for monomorphization. Absent in older snapshots.
    #[serde(default)]
    pub template_spellings: HashMap<String, HashMap<String, String>>,
    /// Template-ids observed at any type parse: (base, whitespace-
    /// stripped args). Backs struct monomorphization's single-spelling
    /// rule. Absent in older snapshots.
    #[serde(default)]
    pub observed_template_ids: std::collections::HashSet<(String, String)>,
    /// Template-id spellings on locals and struct fields (`Indices<3>
    /// indices;`): scope name (function or struct) -> variable/field
    /// name -> normalized spelling. Params live in `template_spellings`
    /// above; this table covers the remaining declaration positions for
    /// struct value-template instantiation (which needs per-site
    /// arguments). Absent in older snapshots.
    #[serde(default)]
    pub template_site_spellings: HashMap<String, HashMap<String, String>>,
    /// Static member initializers by qualified `Struct::member`, as parsed
    /// (unfolded: may reference template value-params). Backs `T::kX`
    /// constexpr evaluation during monomorphization (bind params from the
    /// specialization args, substitute, fold to literal). Absent in older
    /// snapshots.
    #[serde(default)]
    pub template_static_inits: HashMap<String, Expr>,
    /// Template value-param defaults by qualified `Scope::Param` (e.g.
    /// `PackVertex::BoneCount` → `"0"`), as token strings. Backs binding
    /// of partial specialization args (trailing defaulted params fill
    /// from here). Absent in older snapshots.
    #[serde(default)]
    pub template_param_defaults: HashMap<String, String>,
    /// Multi-dimensional struct-field array dimensions by qualified
    /// `Struct::field` (e.g. `Partition3::mask3bit` → `[3, 2]`), for
    /// row-major linearization of `obj.field[i][j]` uses inline
    /// (`→ obj.field[i*D1+j]`) with the declaration flattened to the
    /// product size. Mirrors the global/local `multidim` design, keyed
    /// qualified to avoid cross-struct collisions. Absent in older
    /// snapshots.
    #[serde(default)]
    pub field_multidim: HashMap<String, Vec<u32>>,
}

impl Default for Module {
    fn default() -> Self {
        Self {
            attributes: Vec::new(),
            structs: Vec::new(),
            globals: Vec::new(),
            resources: Vec::new(),
            functions: Vec::new(),
            constants: HashMap::new(),
            template_spellings: HashMap::new(),
            observed_template_ids: std::collections::HashSet::new(),
            template_site_spellings: HashMap::new(),
            template_static_inits: HashMap::new(),
            template_param_defaults: HashMap::new(),
            field_multidim: HashMap::new(),
        }
    }
}
