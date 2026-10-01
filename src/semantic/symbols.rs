use std::collections::HashMap;

use crate::ast::{Function, GlobalDecl, ResourceDecl, StructDecl, Type};
use serde::{Deserialize, Serialize};

/// Records what was declared at the module level: structs and functions.
#[derive(Debug, Default,Serialize, Deserialize)]
pub struct SymbolTable {
    structs: HashMap<String, StructDecl>,
    functions: HashMap<String, Function>,
    resources: HashMap<String, ResourceDecl>,
    globals: HashMap<String, GlobalDecl>,
}

impl SymbolTable {
    pub fn add_struct(&mut self, decl: StructDecl) {
        self.structs.insert(decl.name.clone(), decl);
    }

    pub fn add_function(&mut self, decl: Function) {
        self.functions.insert(decl.name.clone(), decl);
    }

    pub fn add_resource(&mut self, decl: ResourceDecl) {
        self.resources.insert(decl.name.clone(), decl);
    }

    pub fn add_global(&mut self, decl: GlobalDecl) {
        self.globals.insert(decl.name.clone(), decl);
    }

    pub fn get_struct(&self, name: &str) -> Option<&StructDecl> {
        if let Some(decl) = self.structs.get(name) {
            return Some(decl);
        }
        // Scoped nested-type reference (`GeometrySRT::VertexInput`): nested
        // structs are stored under their unqualified name, so fall back to
        // the segment after the last `::`.
        if let Some(unqualified) = name.rsplit("::").next()
            && unqualified.len() != name.len()
        {
            return self.structs.get(unqualified);
        }
        None
    }

    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    pub fn has_struct(&self, name: &str) -> bool {
        self.get_struct(name).is_some()
    }

    pub fn get_resource(&self, name: &str) -> Option<&ResourceDecl> {
        self.resources.get(name)
    }

    pub fn has_resource(&self, name: &str) -> bool {
        self.resources.contains_key(name)
    }

    pub fn get_global_type(&self, name: &str) -> Option<&Type> {
        self.globals.get(name).map(|decl| &decl.ty)
    }

    pub fn get_global(&self, name: &str) -> Option<&GlobalDecl> {
        self.globals.get(name)
    }

    pub fn iter_functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.values()
    }
}

/// A scope of local variable bindings: `name -> ()`.
///
/// For Milestone 1 we only need to know whether a name is declared in scope;
/// richer type information is carried in the type checker's resolved context.
#[derive(Debug, Default)]
pub struct Scope {
    bindings: std::collections::HashSet<String>,
}

impl Scope {
    pub fn declare(&mut self, name: impl Into<String>) {
        self.bindings.insert(name.into());
    }

    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains(name)
    }
}
