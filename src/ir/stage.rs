use serde::{Deserialize, Serialize};
/// A shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    Mesh,
    Hull,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Domain,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Vertex => "vertex",
            Stage::Fragment => "fragment",
            Stage::Compute => "compute",
            Stage::Geometry => "geometry",
            Stage::Mesh => "mesh",
            Stage::Hull => "hull",
            Stage::AnyHit => "anyhit",
            Stage::ClosestHit => "closesthit",
            Stage::Miss => "miss",
            Stage::Intersection => "intersection",
            Stage::Domain => "domain",
        }
    }
}
