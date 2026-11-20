//! Texture post-process through the OIIO command line tools
//! (`TASK-texture-postprocess-oiio`).
//!
//! Runs *after* the bake/generate stages: every map is packaged into a tiled,
//! MIP-mapped `.tx` with `maketx`, tagged with its colorspace, and recorded in
//! a sidecar with exact argv, tool versions, sha256 and dimensions. When AO,
//! roughness and metalness are all present an additional Godot-convention ORM
//! pack is emitted (R = AO, G = roughness, B = metallic).
//!
//! `maketx` docs: <https://openimageio.readthedocs.io/en/stable/maketx.html>

use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Capability, Status, ToolId, Unavailable, run_tool};

/// Manifest schema version.
const MANIFEST_VERSION: u32 = 1;

/// Tile edge length used when a job does not specify one (in pixels).
pub const DEFAULT_TILE_SIZE: u32 = 256;

/// The canonical complete PBR set. A job may ship any subset: `maps` is the
/// explicit declaration of what it carries, and omissions are reported in the
/// sidecar rather than silently treated as a complete set.
const REQUIRED_MAP_ROLES: [MapRole; 5] = [
    MapRole::Albedo,
    MapRole::Normal,
    MapRole::Rough,
    MapRole::Metal,
    MapRole::Ao,
];

/// Colorspace a map is tagged with in the sidecar and in the texture metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColorSpace {
    /// Colour data: the albedo/base-colour map.
    Srgb,
    /// Data maps: normal, roughness, metalness, AO, height and ORM.
    Linear,
}

impl ColorSpace {
    /// The token written into the `oiio:ColorSpace` metadata tag.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Srgb => "srgb",
            Self::Linear => "linear",
        }
    }
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The role a map plays in a PBR set (the vocabulary Godot import copies).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MapRole {
    /// Albedo / base colour (`sRGB`).
    Albedo,
    /// Tangent-space normal (linear).
    Normal,
    /// Roughness (linear).
    Rough,
    /// Metalness (linear).
    Metal,
    /// Ambient occlusion (linear).
    Ao,
    /// Height / displacement (linear).
    Height,
    /// Packed AO/roughness/metalness (linear).
    Orm,
}

impl MapRole {
    /// Stable lowercase identifier.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Albedo => "albedo",
            Self::Normal => "normal",
            Self::Rough => "rough",
            Self::Metal => "metal",
            Self::Ao => "ao",
            Self::Height => "height",
            Self::Orm => "orm",
        }
    }

    /// The only colorspace valid for this role. Data maps are always linear;
    /// tagging one `sRGB` is a hard error (a normal map would be double-gamma'd).
    pub const fn colorspace(self) -> ColorSpace {
        match self {
            Self::Albedo => ColorSpace::Srgb,
            Self::Normal | Self::Rough | Self::Metal | Self::Ao | Self::Height | Self::Orm => {
                ColorSpace::Linear
            }
        }
    }
}

impl fmt::Display for MapRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One input map of a PBR set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputMap {
    /// What the map represents.
    pub role: MapRole,
    /// The map on disk.
    pub path: PathBuf,
    /// Explicit colorspace claim, validated against [`MapRole::colorspace`].
    /// `None` derives it from the role.
    pub colorspace: Option<ColorSpace>,
}

/// One post-process job over a PBR map set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostprocessJob {
    /// Artifact name prefix (`<name>_<role>.tx`).
    pub name: String,
    /// The maps to process.
    pub maps: Vec<InputMap>,
    /// Where the `.tx` files and the sidecar land (created when missing).
    pub output_dir: PathBuf,
    /// Tile edge length in pixels (non-zero, multiple of 16 — the `.tx`
    /// writer rejects any other tile size).
    pub tile_size: u32,
}

/// One post-processed map recorded in the sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputMap {
    /// The role of the map.
    pub role: MapRole,
    /// The tiled, MIP-mapped output.
    pub path: PathBuf,
    /// sha256 of the output file, lowercase hex.
    pub sha256: String,
    /// Width of the highest-resolution level.
    pub width: u32,
    /// Height of the highest-resolution level.
    pub height: u32,
    /// Colorspace the output is tagged with.
    pub colorspace: ColorSpace,
    /// Number of MIP levels, down to 1px.
    pub mip_levels: u32,
    /// Output size in bytes.
    pub bytes: usize,
}

/// The Godot-convention ORM pack: R = AO, G = roughness, B = metalness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrmPack {
    /// The tiled, MIP-mapped packed texture.
    pub path: PathBuf,
    /// The unpacked RGB pack, kept so a consumer can assert channel order.
    pub packed_png: PathBuf,
    /// sha256 of the packed texture, lowercase hex.
    pub sha256: String,
    /// Width of the highest-resolution level.
    pub width: u32,
    /// Height of the highest-resolution level.
    pub height: u32,
    /// Number of MIP levels, down to 1px.
    pub mip_levels: u32,
    /// Output size in bytes.
    pub bytes: usize,
    /// Source roles in R, G, B order — always AO, roughness, metalness.
    pub sources: [MapRole; 3],
}

/// Sidecar written to `<output_dir>/manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostprocessManifest {
    /// Manifest schema version.
    pub manifest: u32,
    /// `oiiotool` version, as reported by its probe.
    pub oiiotool_version: String,
    /// `maketx` version, as reported by its probe.
    pub maketx_version: String,
    /// The job, echoed back.
    pub job: PostprocessJob,
    /// Exact argv of every subprocess, in execution order (binary included).
    pub commands: Vec<Vec<String>>,
    /// One record per output map.
    pub maps: Vec<OutputMap>,
    /// The ORM pack, when all three source maps were present.
    pub orm_pack: Option<OrmPack>,
    /// Non-fatal notes (for example, why no ORM pack was emitted).
    pub warnings: Vec<String>,
    /// Required PBR roles the job did not provide. Partial sets are allowed;
    /// this is the explicit "what is missing" answer for a client.
    pub missing_required_maps: Vec<MapRole>,
    /// Roles whose bit depth the probe could not assert. Never claim a depth
    /// check succeeded for these.
    pub bit_depth_unvalidated: Vec<MapRole>,
}

/// Why a post-process job did not finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostprocessError {
    /// A required binary is absent or failed its probe. The carried
    /// [`Status`] is 501-ready: its `unavailable` names the exact reason.
    Unavailable(Status),
    /// The job is invalid (bad colorspace claim, missing input, empty set).
    InvalidJob(String),
    /// Dimensions or bit depth disagree across the map set, or a role is
    /// listed twice. A named 400 at the route.
    InvalidMapSet(String),
    /// A subprocess failed after a passing probe.
    ToolFailed(String),
    /// A filesystem error.
    Io(String),
}

impl fmt::Display for PostprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(status) => match status.unavailable.as_ref() {
                Some(reason) => write!(f, "texture post-process tool unavailable: {reason}"),
                None => f.write_str("texture post-process tool unavailable"),
            },
            Self::InvalidJob(reason) => write!(f, "invalid post-process job: {reason}"),
            Self::InvalidMapSet(reason) => write!(f, "invalid PBR map set: {reason}"),
            Self::ToolFailed(reason) => write!(f, "texture tool failed: {reason}"),
            Self::Io(reason) => write!(f, "post-process I/O failed: {reason}"),
        }
    }
}

impl std::error::Error for PostprocessError {}

impl PostprocessError {
    /// The tool status to answer a `501` with, when the tool itself failed.
    #[must_use]
    pub fn tool_status(&self) -> Option<&Status> {
        match self {
            Self::Unavailable(status) => Some(status),
            _ => None,
        }
    }
}

/// A probed OIIO toolchain (both binaries ship together).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OiioTool {
    /// `oiiotool --version` output token.
    pub oiiotool_version: String,
    /// `maketx --version` output token.
    pub maketx_version: String,
}

/// Probe both binaries without linking either one.
///
/// An absent binary is [`Unavailable::NotInstalled`]; a present binary whose
/// version probe fails is [`Unavailable::ProbeFailed`]. Both are returned, not
/// swallowed, so a caller can answer a named `501` instead of silently
/// skipping the post-process stage.
pub fn probe(oiiotool: &Path, maketx: &Path, timeout: Duration) -> Result<OiioTool, Unavailable> {
    Ok(OiioTool {
        oiiotool_version: tool_version(oiiotool, timeout)?,
        maketx_version: tool_version(maketx, timeout)?,
    })
}

/// A 501-ready status for the OIIO toolchain, shaped like the registry's
/// statuses so the server surface can reuse its existing capability gate.
#[must_use]
pub fn status(oiiotool: &Path, maketx: &Path, timeout: Duration) -> Status {
    match probe(oiiotool, maketx, timeout) {
        Ok(tool) => Status {
            id: ToolId::Oiio,
            version: Some(tool.oiiotool_version),
            capabilities: vec![Capability::TexturePostprocess],
            devices: Vec::new(),
            unavailable: None,
        },
        Err(unavailable) => unavailable_status(unavailable),
    }
}

/// Run a post-process job.
///
/// The job is validated before any process starts, then the toolchain is
/// probed: if either binary is missing or broken this returns
/// [`PostprocessError::Unavailable`] with the named reason, never a partial
/// success. On success the sidecar is written to `<output_dir>/manifest.json`.
pub fn run(
    job: &PostprocessJob,
    oiiotool: &Path,
    maketx: &Path,
    timeout: Duration,
) -> Result<PostprocessManifest, PostprocessError> {
    validate(job)?;
    let tool = probe(oiiotool, maketx, timeout).map_err(probe_error)?;
    std::fs::create_dir_all(&job.output_dir).map_err(|e| io_error(&e))?;

    let mut commands = Vec::new();
    let mut warnings = Vec::new();

    let missing = missing_required_roles(job);

    // Probe every input before writing anything: a mismatched set must not
    // leave half a map series in the bundle.
    let mut probed = Vec::with_capacity(job.maps.len());
    for map in &job.maps {
        let (width, height, bit_depth) = dimensions(oiiotool, &map.path, timeout, &mut commands)?;
        probed.push((map.role, width, height, bit_depth));
    }
    let bit_depth_unvalidated = check_map_set(&probed)?;

    let mut maps = Vec::with_capacity(job.maps.len());
    for (map, (_, width, height, _)) in job.maps.iter().zip(&probed) {
        let output = job
            .output_dir
            .join(format!("{}_{}.tx", job.name, map.role.as_str()));
        let cs = map.role.colorspace();
        invoke(
            maketx,
            &maketx_args(&map.path, &output, cs, job.tile_size),
            timeout,
            &mut commands,
        )?;
        let bytes = std::fs::read(&output).map_err(|e| io_error(&e))?;
        maps.push(OutputMap {
            role: map.role,
            path: output,
            sha256: sha256_hex(&bytes),
            width: *width,
            height: *height,
            colorspace: cs,
            mip_levels: mip_levels(*width, *height),
            bytes: bytes.len(),
        });
    }

    let orm_pack = pack_orm(job, oiiotool, maketx, timeout, &mut commands, &mut warnings)?;

    let manifest = PostprocessManifest {
        manifest: MANIFEST_VERSION,
        oiiotool_version: tool.oiiotool_version,
        maketx_version: tool.maketx_version,
        job: job.clone(),
        commands,
        maps,
        orm_pack,
        warnings,
        missing_required_maps: missing,
        bit_depth_unvalidated,
    };
    let raw = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| PostprocessError::Io(format!("cannot encode the sidecar: {e}")))?;
    std::fs::write(job.output_dir.join("manifest.json"), raw).map_err(|e| io_error(&e))?;
    Ok(manifest)
}

/// Verify one map series. Dimensions and probed bit depth must agree; a depth
/// the probe did not report is returned for the sidecar to name as unvalidated.
fn check_map_set(
    probed: &[(MapRole, u32, u32, Option<String>)],
) -> Result<Vec<MapRole>, PostprocessError> {
    let (first_role, first_width, first_height, _) = &probed[0];
    let mut bit_depth_unvalidated = Vec::new();
    // Compare against the first REPORTED depth, not the first map: an unknown
    // first depth must not hide a mismatch between later maps.
    let mut first_reported: Option<(&MapRole, &str)> = None;
    for (role, width, height, depth) in probed {
        if width != first_width || height != first_height {
            return Err(PostprocessError::InvalidMapSet(format!(
                "{role} map is {width}x{height} but {first_role} is {first_width}x{first_height}"
            )));
        }
        if let Some(depth) = depth.as_deref() {
            if let Some((reported_role, first)) = first_reported {
                if depth != first {
                    return Err(PostprocessError::InvalidMapSet(format!(
                        "{role} bit depth {depth} differs from {reported_role} ({first})"
                    )));
                }
            } else {
                first_reported = Some((role, depth));
            }
        } else {
            bit_depth_unvalidated.push(*role);
        }
    }
    Ok(bit_depth_unvalidated)
}

/// Roles of the canonical PBR set a partial job leaves out.
fn missing_required_roles(job: &PostprocessJob) -> Vec<MapRole> {
    REQUIRED_MAP_ROLES
        .iter()
        .copied()
        .filter(|role| !job.maps.iter().any(|map| map.role == *role))
        .collect()
}

/// Reject jobs that cannot produce an honest sidecar before spawning anything.
fn validate(job: &PostprocessJob) -> Result<(), PostprocessError> {
    let bad = |reason: String| PostprocessError::InvalidJob(reason);
    if job.maps.is_empty() {
        return Err(bad("no maps to post-process".to_string()));
    }
    for (index, map) in job.maps.iter().enumerate() {
        if job.maps[..index]
            .iter()
            .any(|earlier| earlier.role == map.role)
        {
            return Err(PostprocessError::InvalidMapSet(format!(
                "duplicate map role: {}",
                map.role
            )));
        }
    }
    let tile = job.tile_size;
    if tile == 0 {
        return Err(bad("tile size must be greater than zero".to_string()));
    }
    if !tile.is_multiple_of(16) {
        return Err(bad(format!(
            "tile size {tile} must be a multiple of 16 for .tx output"
        )));
    }
    for map in &job.maps {
        let expected = map.role.colorspace();
        if let Some(claimed) = map.colorspace
            && claimed != expected
        {
            // Name only the role: the input path is host state and must
            // never reach a client error body.
            return Err(bad(format!(
                "{} map is tagged {claimed} but must be {expected}",
                map.role,
            )));
        }
        if !map.path.is_file() {
            return Err(bad(format!("{} input map not found", map.role)));
        }
    }
    Ok(())
}

/// Build the Godot ORM pack when all three source maps exist; otherwise record
/// a warning rather than failing the whole map set.
fn pack_orm(
    job: &PostprocessJob,
    oiiotool: &Path,
    maketx: &Path,
    timeout: Duration,
    commands: &mut Vec<Vec<String>>,
    warnings: &mut Vec<String>,
) -> Result<Option<OrmPack>, PostprocessError> {
    let source = |role| {
        job.maps
            .iter()
            .find(|map| map.role == role)
            .map(|map| map.path.as_path())
    };
    let (Some(ao), Some(rough), Some(metal)) = (
        source(MapRole::Ao),
        source(MapRole::Rough),
        source(MapRole::Metal),
    ) else {
        warnings.push(
            "ORM pack omitted: AO, roughness and metalness maps are all required".to_string(),
        );
        return Ok(None);
    };

    let png = job.output_dir.join(format!("{}_orm.png", job.name));
    let tx = job.output_dir.join(format!("{}_orm.tx", job.name));
    let (width, height, _) = dimensions(oiiotool, ao, timeout, commands)?;
    invoke(
        oiiotool,
        &orm_args(ao, rough, metal, &png),
        timeout,
        commands,
    )?;
    invoke(
        maketx,
        &maketx_args(&png, &tx, ColorSpace::Linear, job.tile_size),
        timeout,
        commands,
    )?;
    let bytes = std::fs::read(&tx).map_err(|e| io_error(&e))?;
    Ok(Some(OrmPack {
        path: tx,
        packed_png: png,
        sha256: sha256_hex(&bytes),
        width,
        height,
        mip_levels: mip_levels(width, height),
        bytes: bytes.len(),
        sources: [MapRole::Ao, MapRole::Rough, MapRole::Metal],
    }))
}

/// Run one tool, recording the exact argv before it starts.
fn invoke(
    bin: &Path,
    args: &[String],
    timeout: Duration,
    commands: &mut Vec<Vec<String>>,
) -> Result<String, PostprocessError> {
    let mut command = Vec::with_capacity(args.len() + 1);
    command.push(bin.to_string_lossy().into_owned());
    command.extend_from_slice(args);
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let out = run_tool(bin, &refs, timeout).map_err(|e| tool_error(&e))?;
    commands.push(command);
    Ok(out)
}

/// Read the resolution of an image through `oiiotool --info` (format-agnostic).
fn dimensions(
    oiiotool: &Path,
    image: &Path,
    timeout: Duration,
    commands: &mut Vec<Vec<String>>,
) -> Result<(u32, u32, Option<String>), PostprocessError> {
    let args = vec!["--info".to_string(), image.to_string_lossy().into_owned()];
    let out = invoke(oiiotool, &args, timeout, commands)?;
    let bit_depth = parse_bit_depth(&out).map(str::to_string);
    parse_dimensions(&out)
        .map(|(width, height)| (width, height, bit_depth))
        // Neither the image path nor raw tool output belongs in an error
        // body: both can carry the operator's host layout.
        .ok_or_else(|| PostprocessError::ToolFailed("cannot read map dimensions".to_owned()))
}

/// Bit-depth token in an `oiiotool --info` line, when the probe reports one.
/// A format we cannot recognise is not a successful check: the caller records
/// the map as unvalidated instead.
fn parse_bit_depth(info: &str) -> Option<&str> {
    info.split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_ascii_alphanumeric()))
        .find(|token| {
            matches!(
                *token,
                "uint8"
                    | "uint16"
                    | "uint32"
                    | "uint64"
                    | "int8"
                    | "int16"
                    | "int32"
                    | "half"
                    | "float"
                    | "double"
            )
        })
}

/// Parse the `<width> x <height>` token out of an `oiiotool --info` line.
fn parse_dimensions(info: &str) -> Option<(u32, u32)> {
    let tokens: Vec<&str> = info.split_whitespace().collect();
    for pair in tokens.windows(3) {
        if pair[1] == "x" {
            // A stray `x` token is not a dimension pair; keep scanning.
            let Some(width) = pair[0]
                .trim_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .ok()
            else {
                continue;
            };
            let Some(height) = pair[2]
                .trim_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .ok()
            else {
                continue;
            };
            return Some((width, height));
        }
    }
    None
}

/// MIP levels from the full-resolution level down to 1px, for any aspect.
const fn mip_levels(width: u32, height: u32) -> u32 {
    let edge = if width > height { width } else { height };
    (if edge == 0 { 1 } else { edge }).ilog2() + 1
}

/// `maketx` argv: tiled, separable, periodic-wrap, high-quality MIP chain with
/// an explicit colorspace tag. MIP-mapping is `maketx`'s default (no
/// `--nomipmap`), so the chain reaches 1px without extra flags.
fn maketx_args(input: &Path, output: &Path, cs: ColorSpace, tile: u32) -> Vec<String> {
    let edge = tile.to_string();
    vec![
        input.to_string_lossy().into_owned(),
        "--tile".to_string(),
        edge.clone(),
        edge,
        "--separate".to_string(),
        "--wrap".to_string(),
        "periodic".to_string(),
        "--filter".to_string(),
        "lanczos3".to_string(),
        "--sattrib".to_string(),
        "oiio:ColorSpace".to_string(),
        cs.as_str().to_string(),
        "-o".to_string(),
        output.to_string_lossy().into_owned(),
    ]
}

/// `oiiotool` argv packing the R channel of AO/rough/metal into R/G/B.
///
/// Each source channel is renamed before `--chappend` so the pack never has
/// duplicate channel names (all three sources contribute their `R` channel).
fn orm_args(ao: &Path, rough: &Path, metal: &Path, output: &Path) -> Vec<String> {
    vec![
        ao.to_string_lossy().into_owned(),
        "--ch".to_string(),
        "R".to_string(),
        "--chnames".to_string(),
        "R".to_string(),
        rough.to_string_lossy().into_owned(),
        "--ch".to_string(),
        "R".to_string(),
        "--chnames".to_string(),
        "G".to_string(),
        "--chappend".to_string(),
        metal.to_string_lossy().into_owned(),
        "--ch".to_string(),
        "R".to_string(),
        "--chnames".to_string(),
        "B".to_string(),
        "--chappend".to_string(),
        "-o".to_string(),
        output.to_string_lossy().into_owned(),
    ]
}

/// Version token from a `--version` run; a tool that prints no version is a
/// failed probe, not a silent success.
fn tool_version(bin: &Path, timeout: Duration) -> Result<String, Unavailable> {
    let out = run_tool(bin, &["--version"], timeout)?;
    out.lines()
        .flat_map(|line| line.split_whitespace())
        .find(|token| token.chars().any(|c| c.is_ascii_digit()))
        .map(str::to_string)
        .ok_or_else(|| {
            Unavailable::ProbeFailed(format!("{}: unexpected --version output", bin.display()))
        })
}

fn unavailable_status(unavailable: Unavailable) -> Status {
    Status {
        id: ToolId::Oiio,
        version: None,
        capabilities: Vec::new(),
        devices: Vec::new(),
        unavailable: Some(unavailable),
    }
}

fn probe_error(unavailable: Unavailable) -> PostprocessError {
    PostprocessError::Unavailable(unavailable_status(unavailable))
}

fn tool_error(unavailable: &Unavailable) -> PostprocessError {
    if matches!(unavailable, Unavailable::NotInstalled) {
        probe_error(Unavailable::NotInstalled)
    } else {
        PostprocessError::ToolFailed(unavailable.to_string())
    }
}

fn io_error(err: &std::io::Error) -> PostprocessError {
    PostprocessError::Io(err.to_string())
}

/// sha256 of a byte slice, lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test's private directory: unique per process and tag, wiped on
    /// creation so a prior run cannot leak state, removed on drop even when
    /// the test panics. Each test owns exactly one tag.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("honkbox-tex-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn input(role: MapRole, dir: &Path) -> InputMap {
        InputMap {
            role,
            path: dir.join(format!("{role}.png")),
            colorspace: None,
        }
    }

    fn job(dir: &Path, maps: Vec<InputMap>) -> PostprocessJob {
        PostprocessJob {
            name: "fixture".to_string(),
            maps,
            output_dir: dir.join("out"),
            tile_size: 16,
        }
    }

    #[test]
    fn mip_levels_reach_one_pixel_for_any_aspect() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(2, 2), 2);
        assert_eq!(mip_levels(1024, 1024), 11);
        assert_eq!(mip_levels(1024, 512), 11);
        assert_eq!(mip_levels(512, 1024), 11);
        assert_eq!(mip_levels(3, 2), 2);
    }

    #[test]
    fn dimensions_parse_non_square_info_lines() {
        assert_eq!(
            parse_dimensions("map.png : 1024 x 512, 3 channel, uint8 png"),
            Some((1024, 512))
        );
        assert_eq!(parse_dimensions("no resolution here"), None);
    }

    #[test]
    fn a_normal_map_tagged_srgb_is_a_hard_error() {
        let scratch = Scratch::new("bad-normal");
        let dir = scratch.path();
        let path = dir.join("normal.png");
        std::fs::write(&path, b"not a real image").expect("fixture");
        let mut normal = input(MapRole::Normal, dir);
        normal.path = path.clone();
        normal.colorspace = Some(ColorSpace::Srgb);
        let task = job(dir, vec![normal]);

        let err = run(
            &task,
            Path::new("oiiotool"),
            Path::new("maketx"),
            Duration::from_secs(1),
        )
        .expect_err("must fail before any probe");
        assert!(matches!(err, PostprocessError::InvalidJob(_)), "{err:?}");
        assert!(err.to_string().contains("must be linear"), "{err}");
        assert!(
            !err.to_string().contains(path.to_string_lossy().as_ref()),
            "host path leaked into the error: {err}"
        );
        assert!(err.tool_status().is_none());
    }

    #[test]
    fn a_wrong_albedo_claim_is_rejected() {
        let scratch = Scratch::new("bad-albedo");
        let dir = scratch.path();
        let path = dir.join("albedo.png");
        std::fs::write(&path, b"x").expect("fixture");
        let mut albedo = input(MapRole::Albedo, dir);
        albedo.path = path;
        albedo.colorspace = Some(ColorSpace::Linear);
        let task = job(dir, vec![albedo]);

        let err = run(
            &task,
            Path::new("oiiotool"),
            Path::new("maketx"),
            Duration::from_secs(1),
        )
        .expect_err("must fail");
        assert!(matches!(err, PostprocessError::InvalidJob(_)), "{err:?}");
    }

    #[test]
    fn empty_jobs_zero_tiles_and_missing_files_are_rejected() {
        let scratch = Scratch::new("invalid");
        let dir = scratch.path();
        let mut task = job(dir, vec![]);
        assert!(matches!(
            run(
                &task,
                Path::new("oiiotool"),
                Path::new("maketx"),
                Duration::from_secs(1)
            ),
            Err(PostprocessError::InvalidJob(_))
        ));

        task = job(dir, vec![input(MapRole::Albedo, dir)]);
        assert!(matches!(
            run(
                &task,
                Path::new("oiiotool"),
                Path::new("maketx"),
                Duration::from_secs(1)
            ),
            Err(PostprocessError::InvalidJob(_))
        ));

        let path = dir.join("albedo.png");
        std::fs::write(&path, b"x").expect("fixture");
        let mut one = input(MapRole::Albedo, dir);
        one.path = path;
        task = job(dir, vec![one]);
        for invalid_tile in [0, 17] {
            task.tile_size = invalid_tile;
            assert!(matches!(
                run(
                    &task,
                    Path::new("oiiotool"),
                    Path::new("maketx"),
                    Duration::from_secs(1)
                ),
                Err(PostprocessError::InvalidJob(_))
            ));
        }
    }

    #[test]
    fn argv_tags_colorspace_tiles_and_packs_orm_order() {
        let args = maketx_args(
            Path::new("albedo.png"),
            Path::new("albedo.tx"),
            ColorSpace::Srgb,
            64,
        );
        assert!(
            args.windows(2)
                .any(|w| w[0] == "--sattrib" && w[1] == "oiio:ColorSpace")
        );
        assert!(args.iter().any(|a| a.as_str() == "srgb"));
        assert!(
            args.windows(3)
                .any(|w| w[0] == "--tile" && w[1] == "64" && w[2] == "64")
        );
        assert!(!args.iter().any(|a| a.as_str() == "--nomipmap"));

        let args = orm_args(
            Path::new("ao.png"),
            Path::new("rough.png"),
            Path::new("metal.png"),
            Path::new("orm.png"),
        );
        assert_eq!(
            args.iter().filter(|a| a.as_str() == "--chappend").count(),
            2
        );
        for name in ["R", "G", "B"] {
            assert!(args.windows(2).any(|w| w[0] == "--chnames" && w[1] == name));
        }
        assert!(
            args.iter().position(|a| a.as_str() == "ao.png").unwrap()
                < args.iter().position(|a| a.as_str() == "rough.png").unwrap()
        );
        assert!(
            args.iter().position(|a| a.as_str() == "rough.png").unwrap()
                < args.iter().position(|a| a.as_str() == "metal.png").unwrap()
        );
    }

    #[test]
    fn an_absent_binary_is_a_named_not_installed_status() {
        let missing = Path::new("/nonexistent/honkbox-oiiotool");
        let err = probe(missing, missing, Duration::from_secs(1)).expect_err("must fail");
        assert_eq!(err, Unavailable::NotInstalled);

        let st = status(missing, missing, Duration::from_secs(1));
        assert!(!st.is_available());
        assert_eq!(st.id, ToolId::Oiio);
        assert_eq!(st.unavailable, Some(Unavailable::NotInstalled));
        assert_eq!(st.capabilities, Vec::new());

        let scratch = Scratch::new("absent");
        let dir = scratch.path();
        let path = dir.join("albedo.png");
        std::fs::write(&path, b"x").expect("fixture");
        let task = job(dir, vec![input(MapRole::Albedo, dir)]);
        let err = run(&task, missing, missing, Duration::from_secs(1))
            .expect_err("absent binary must not be a silent skip");
        assert!(matches!(err, PostprocessError::Unavailable(_)), "{err:?}");
        assert_eq!(
            err.tool_status().and_then(|st| st.unavailable.clone()),
            Some(Unavailable::NotInstalled)
        );
    }

    #[cfg(unix)]
    fn fake_tool(tag: &str, script: &str) -> (Scratch, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let scratch = Scratch::new(tag);
        let path = scratch.path().join("tool");
        let tmp = scratch
            .path()
            .join(format!("tool.tmp.{}", std::process::id()));
        std::fs::write(&tmp, script).expect("fake tool");
        let mut perms = std::fs::metadata(&tmp).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&tmp, perms).expect("chmod");
        std::fs::rename(&tmp, &path).expect("rename");
        // The rename carries the mode across; no second chmod needed.
        (scratch, path)
    }

    #[cfg(unix)]
    #[test]
    fn a_versioned_binary_passes_the_probe() {
        let (_dir, bin) = fake_tool("probe-ok", "#!/bin/sh\necho '2.5.18'\n");
        let tool = probe(&bin, &bin, Duration::from_secs(2)).expect("probe");
        assert_eq!(tool.oiiotool_version, "2.5.18");
        assert_eq!(tool.maketx_version, "2.5.18");
        let st = status(&bin, &bin, Duration::from_secs(2));
        assert!(st.is_available());
        assert_eq!(st.capabilities, vec![Capability::TexturePostprocess]);
    }

    #[cfg(unix)]
    #[test]
    fn a_present_but_broken_binary_is_a_probe_failed_status() {
        let (_fail_dir, failing) = fake_tool("probe-fail", "#!/bin/sh\necho 'boom' >&2\nexit 3\n");
        let err = probe(&failing, &failing, Duration::from_secs(2)).expect_err("must fail");
        assert!(matches!(err, Unavailable::ProbeFailed(_)), "{err:?}");
        assert_eq!(
            status(&failing, &failing, Duration::from_secs(2)).unavailable,
            Some(err)
        );

        let (_garbage_dir, garbage) =
            fake_tool("probe-garbage", "#!/bin/sh\necho 'no version here'\n");
        assert!(matches!(
            probe(&garbage, &garbage, Duration::from_secs(2)),
            Err(Unavailable::ProbeFailed(_))
        ));
    }

    /// Locate a real toolchain, or `None` when it is not installed. Tests that
    /// need real images return early on `None`: a missing binary is a skip,
    /// never a failure.
    fn real_toolchain() -> Option<(PathBuf, PathBuf)> {
        let oiiotool = std::env::var("HONKBOX_OIIOTOOL").unwrap_or_else(|_| "oiiotool".to_string());
        let maketx = std::env::var("HONKBOX_MAKETX").unwrap_or_else(|_| "maketx".to_string());
        let oiiotool = PathBuf::from(oiiotool);
        let maketx = PathBuf::from(maketx);
        probe(&oiiotool, &maketx, Duration::from_secs(5))
            .ok()
            .map(|_| (oiiotool, maketx))
    }

    /// Make a solid-colour map with `oiiotool` itself (no image crate needed).
    fn solid(oiiotool: &Path, path: &Path, value: f64) {
        let fill = format!("--fill:color={value},{value},{value}");
        let size = "8x8".to_string();
        let out = path.to_string_lossy().into_owned();
        let args = vec![
            "--create".to_string(),
            size.clone(),
            "3".to_string(),
            fill,
            size,
            "-o".to_string(),
            out,
        ];
        let mut commands = Vec::new();
        invoke(oiiotool, &args, Duration::from_secs(30), &mut commands).expect("solid fixture");
    }

    #[test]
    fn a_real_run_packs_orm_and_writes_a_honest_sidecar() {
        let Some((oiiotool, maketx)) = real_toolchain() else {
            return;
        };
        let scratch = Scratch::new("real");
        let dir = scratch.path();
        let sources = [
            (MapRole::Albedo, 0.9),
            (MapRole::Normal, 0.5),
            (MapRole::Rough, 0.5),
            (MapRole::Metal, 0.75),
            (MapRole::Ao, 0.25),
        ];
        let mut maps = Vec::new();
        for (role, value) in sources {
            let map = input(role, dir);
            solid(&oiiotool, &map.path, value);
            maps.push(map);
        }
        let task = job(dir, maps);

        let manifest = run(&task, &oiiotool, &maketx, Duration::from_secs(120)).expect("manifest");

        assert_eq!(manifest.maps.len(), 5);
        assert_eq!(manifest.missing_required_maps, [] as [MapRole; 0]);
        // 5 dims + 5 maketx + 1 ORM dims + 2 ORM invocations = 13.
        assert_eq!(manifest.commands.len(), 13);
        assert_eq!(manifest.maps[0].colorspace, ColorSpace::Srgb);
        assert!(
            manifest.maps[1..]
                .iter()
                .all(|m| m.colorspace == ColorSpace::Linear)
        );
        assert!(manifest.maps.iter().all(|m| m.mip_levels == 4));
        assert!(manifest.maps.iter().all(|m| (m.width, m.height) == (8, 8)));
        assert!(manifest.maps.iter().all(|m| m.sha256.len() == 64));
        let orm = manifest.orm_pack.expect("ORM pack");
        assert_eq!(orm.sources, [MapRole::Ao, MapRole::Rough, MapRole::Metal]);
        assert_eq!(orm.mip_levels, 4);
        assert!(orm.packed_png.exists() && orm.path.exists());
        assert!(dir.join("out/manifest.json").exists());

        // Assert the channel order per pixel through `oiiotool`'s own stats.
        let mut commands = Vec::new();
        let output = invoke(
            &oiiotool,
            &[
                orm.packed_png.to_string_lossy().into_owned(),
                "--echo".to_string(),
                "{TOP.AVGCOLOR}".to_string(),
            ],
            Duration::from_secs(30),
            &mut commands,
        )
        .expect("stats");
        let avg: Vec<f64> = output
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|token| {
                token
                    .trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
                    .parse()
                    .ok()
            })
            .collect();
        assert_eq!(avg.len(), 3, "stats: {output}");
        assert!((avg[0] - 0.25).abs() < 0.03, "R must be AO: {output}");
        assert!((avg[1] - 0.50).abs() < 0.03, "G must be rough: {output}");
        assert!((avg[2] - 0.75).abs() < 0.03, "B must be metal: {output}");
    }

    #[test]
    fn a_set_without_orm_sources_degrades_gracefully() {
        let Some((oiiotool, maketx)) = real_toolchain() else {
            return;
        };
        let scratch = Scratch::new("no-orm");
        let dir = scratch.path();
        let mut albedo = input(MapRole::Albedo, dir);
        solid(&oiiotool, &albedo.path, 0.9);
        albedo.colorspace = None;
        let task = job(dir, vec![albedo]);

        let manifest = run(&task, &oiiotool, &maketx, Duration::from_secs(120)).expect("manifest");
        assert!(manifest.orm_pack.is_none());
        assert_eq!(manifest.warnings.len(), 1);
        assert!(manifest.warnings[0].contains("ORM pack omitted"));
        assert_eq!(manifest.maps.len(), 1);
    }

    #[test]
    fn bit_depth_is_only_claimed_when_the_probe_reports_it() {
        assert_eq!(
            parse_bit_depth("map.png : 1024 x 512, 3 channel, uint16 png"),
            Some("uint16")
        );
        assert_eq!(parse_bit_depth("map.png : 8 x 8, 3 channel, png"), None);
    }

    #[test]
    fn unknown_first_depth_does_not_hide_a_later_mismatch() {
        let probed = [
            (MapRole::Albedo, 8, 8, None),
            (MapRole::Normal, 8, 8, Some("uint8".to_owned())),
            (MapRole::Rough, 8, 8, Some("uint16".to_owned())),
        ];
        let err = check_map_set(&probed).expect_err("two reported depths disagree");
        assert!(err.to_string().contains("uint16"), "{err}");
    }

    #[test]
    fn dimensions_skip_a_stray_x_token() {
        assert_eq!(parse_dimensions("map.png : x 8 x 8 png"), Some((8, 8)));
    }

    #[test]
    fn duplicate_roles_are_a_named_map_set_error() {
        let scratch = Scratch::new("dup-roles");
        let dir = scratch.path();
        let mut first = input(MapRole::Albedo, dir);
        std::fs::write(&first.path, b"x").expect("fixture");
        first.colorspace = Some(ColorSpace::Srgb);
        let task = job(dir, vec![first, input(MapRole::Albedo, dir)]);

        let err = run(
            &task,
            Path::new("oiiotool"),
            Path::new("maketx"),
            Duration::from_secs(1),
        )
        .expect_err("two maps with one role overwrite each other's output");
        assert!(matches!(err, PostprocessError::InvalidMapSet(_)), "{err:?}");
        assert!(err.to_string().contains("duplicate map role"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn mismatched_dimensions_are_a_named_map_set_error() {
        let (_dir, oiiotool) = fake_tool(
            "dims-mismatch",
            "#!/bin/sh\ncase \"$1\" in --version) echo '2.5.18';; --info) case \"$2\" in *albedo*) echo 'a.png : 8 x 8, 3 channel, uint8 png';; *normal*) echo 'n.png : 16 x 16, 3 channel, uint8 png';; esac;; esac\n",
        );
        let scratch = Scratch::new("dims-job");
        let dir = scratch.path();
        let albedo = input(MapRole::Albedo, dir);
        let normal = input(MapRole::Normal, dir);
        std::fs::write(&albedo.path, b"x").expect("fixture");
        std::fs::write(&normal.path, b"x").expect("fixture");
        let task = job(dir, vec![albedo, normal]);

        let err = run(&task, &oiiotool, &oiiotool, Duration::from_secs(2))
            .expect_err("a mismatched set must not produce a bundle");
        assert!(matches!(err, PostprocessError::InvalidMapSet(_)), "{err:?}");
        assert!(err.to_string().contains("16x16"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn an_unreported_bit_depth_is_recorded_as_unvalidated() {
        let (_dir, oiiotool) = fake_tool(
            "depth-unvalidated",
            "#!/bin/sh\ncase \"$1\" in --version) echo '2.5.18';; --info) echo 'map.png : 8 x 8, 3 channel, png';; esac\n",
        );
        let (_tool_dir, maketx) = fake_tool(
            "depth-maketx",
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo '7.1.0'; exit 0; fi\nwhile [ \"$#\" -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then shift; printf 'fake-tx' > \"$1\"; exit 0; fi; shift; done\nexit 1\n",
        );
        let scratch = Scratch::new("depth-job");
        let dir = scratch.path();
        let albedo = input(MapRole::Albedo, dir);
        let normal = input(MapRole::Normal, dir);
        std::fs::write(&albedo.path, b"x").expect("fixture");
        std::fs::write(&normal.path, b"x").expect("fixture");
        let task = job(dir, vec![albedo, normal]);

        let manifest = run(&task, &oiiotool, &maketx, Duration::from_secs(2)).expect("manifest");
        assert_eq!(
            manifest.bit_depth_unvalidated,
            vec![MapRole::Albedo, MapRole::Normal]
        );
        assert_eq!(
            manifest.missing_required_maps,
            vec![MapRole::Rough, MapRole::Metal, MapRole::Ao]
        );
        assert_eq!(manifest.maps.len(), 2);
        assert!(
            manifest
                .maps
                .iter()
                .all(|m| m.sha256 == sha256_hex(b"fake-tx")),
            "per-map hash of the published output"
        );
        assert_eq!(manifest.oiiotool_version, "2.5.18");
        assert_eq!(manifest.maketx_version, "7.1.0");
    }
}
