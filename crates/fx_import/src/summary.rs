//! FPackageFileSummary: the header of an editor .uasset.

use crate::reader::{Error, Guid, Reader, Result};

pub const PACKAGE_FILE_TAG: u32 = 0x9E2A_83C1;

/// UE4 object version every UE5 package carries (the last UE4 version).
pub const UE4_VERSION_UE5: i32 = 522;

/// EUnrealEngineObjectUE5Version values this reader branches on.
pub mod ue5 {
    pub const INITIAL_VERSION: i32 = 1000;
    pub const NAMES_REFERENCED_FROM_EXPORT_DATA: i32 = 1001;
    pub const PAYLOAD_TOC: i32 = 1002;
    pub const OPTIONAL_RESOURCES: i32 = 1003;
    pub const LARGE_WORLD_COORDINATES: i32 = 1004;
    pub const REMOVE_OBJECT_EXPORT_PACKAGE_GUID: i32 = 1005;
    pub const TRACK_OBJECT_EXPORT_IS_INHERITED: i32 = 1006;
    pub const FSOFTOBJECTPATH_REMOVE_ASSET_PATH_FNAMES: i32 = 1007;
    pub const ADD_SOFTOBJECTPATH_LIST: i32 = 1008;
    pub const DATA_RESOURCES: i32 = 1009;
    pub const SCRIPT_SERIALIZATION_OFFSET: i32 = 1010;
    pub const PROPERTY_TAG_EXTENSION_AND_OVERRIDABLE_SERIALIZATION: i32 = 1011;
    pub const PROPERTY_TAG_COMPLETE_TYPE_NAME: i32 = 1012;
    pub const ASSETREGISTRY_PACKAGEBUILDDEPENDENCIES: i32 = 1013;
    pub const METADATA_SERIALIZATION_OFFSET: i32 = 1014;
    pub const VERSE_CELLS: i32 = 1015;
    pub const PACKAGE_SAVED_HASH: i32 = 1016;
    pub const OS_SUB_OBJECT_SHADOW_SERIALIZATION: i32 = 1017;
    pub const IMPORT_TYPE_HIERARCHIES: i32 = 1018;
    /// Newest version this reader knows.
    pub const LATEST: i32 = IMPORT_TYPE_HIERARCHIES;
}

/// Custom version GUIDs and the values this reader branches on.
pub mod custom {
    use crate::reader::Guid;
    pub const NIAGARA: Guid = Guid([0xFCF57AFA, 0x50764283, 0xB9A9E658, 0xFFA02D32]);
    /// FNiagaraCustomVersion::VariablesUseTypeDefRegistry: FNiagaraVariableBase
    /// is saved natively (name + type definition struct).
    pub const NIAGARA_VARIABLES_USE_TYPE_DEF_REGISTRY: i32 = 64;
    /// FNiagaraCustomVersion::AddGeneratedFunctionsToGPUParamInfo.
    pub const NIAGARA_GENERATED_FUNCTIONS_IN_GPU_PARAM_INFO: i32 = 55;
    /// FNiagaraCustomVersion::AddVariadicParametersToGPUFunctionInfo.
    pub const NIAGARA_VARIADIC_PARAMETERS_IN_GPU_FUNCTION_INFO: i32 = 77;
    /// FNiagaraCustomVersion::SerializeUsageBitMaskToGPUFunctionInfo.
    pub const NIAGARA_USAGE_BITMASK_IN_GPU_FUNCTION_INFO: i32 = 91;
    pub const EDITOR_OBJECT: Guid = Guid([0xE4B068ED, 0xF49442E9, 0xA231DA0B, 0x2E46BB41]);
    /// FEditorObjectVersion::CultureInvariantTextSerializationKeyStability.
    pub const EDITOR_CULTURE_INVARIANT_TEXT: i32 = 32;
    pub const FORTNITE_MAIN: Guid = Guid([0x601D1886, 0xAC644F84, 0xAA16D3DE, 0x0DEAC7D6]);
    /// FFortniteMainBranchObjectVersion::SoftObjectPathUtf8SubPaths.
    pub const FORTNITE_SOFT_OBJECT_PATH_UTF8: i32 = 192;
    pub const FRAMEWORK_OBJECT: Guid = Guid([0xCFFC743F, 0x43B04480, 0x939114DF, 0x171D2073]);
    /// FFrameworkObjectVersion::EdGraphPinContainerType.
    pub const FRAMEWORK_PIN_CONTAINER_TYPE: i32 = 26;
    /// FFrameworkObjectVersion::PinsStoreFName.
    pub const FRAMEWORK_PINS_STORE_FNAME: i32 = 31;
}

/// EPackageFlags::PKG_FilterEditorOnly: a cooked package without editor data.
pub const PKG_FILTER_EDITOR_ONLY: u32 = 0x8000_0000;
/// EPackageFlags::PKG_UnversionedProperties: properties saved without tags.
pub const PKG_UNVERSIONED_PROPERTIES: u32 = 0x0000_2000;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub changelist: u32,
    pub branch: String,
}

impl std::fmt::Display for EngineVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}-{}+{}",
            self.major,
            self.minor,
            self.patch,
            self.changelist & 0x7FFF_FFFF,
            self.branch
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    pub legacy_file_version: i32,
    pub ue4_version: i32,
    pub ue5_version: i32,
    pub licensee_version: i32,
    pub custom_versions: Vec<(Guid, i32)>,
    pub total_header_size: i32,
    pub package_name: String,
    pub package_flags: u32,
    pub name_count: usize,
    pub name_offset: usize,
    pub soft_object_paths_count: usize,
    pub soft_object_paths_offset: usize,
    pub localization_id: String,
    pub export_count: usize,
    pub export_offset: usize,
    pub import_count: usize,
    pub import_offset: usize,
    pub depends_offset: i32,
    pub thumbnail_table_offset: i32,
    pub saved_by: EngineVersion,
    pub compatible_with: EngineVersion,
    pub asset_registry_data_offset: i32,
    pub bulk_data_start_offset: i64,
}

impl Summary {
    pub fn custom_version(&self, guid: Guid) -> Option<i32> {
        self.custom_versions
            .iter()
            .find(|(g, _)| *g == guid)
            .map(|(_, v)| *v)
    }

    pub fn filter_editor_only(&self) -> bool {
        self.package_flags & PKG_FILTER_EDITOR_ONLY != 0
    }

    pub fn read(r: &mut Reader) -> Result<Summary> {
        let mut s = Summary::default();
        let tag = r.u32()?;
        if tag != PACKAGE_FILE_TAG {
            return Err(Error::new(0, format!("not an Unreal package (tag {tag:#010x})")));
        }
        s.legacy_file_version = r.i32()?;
        if !(-9..=-8).contains(&s.legacy_file_version) {
            return Err(Error::new(
                4,
                format!(
                    "legacy file version {} is not a UE5 package (expected -8 or -9)",
                    s.legacy_file_version
                ),
            ));
        }
        let _ue3 = r.i32()?;
        s.ue4_version = r.i32()?;
        s.ue5_version = r.i32()?;
        s.licensee_version = r.i32()?;
        if s.ue4_version == 0 && s.ue5_version == 0 {
            return Err(Error::new(
                12,
                "unversioned package (cooked); only editor saves are supported",
            ));
        }
        if s.ue4_version != UE4_VERSION_UE5 || !(ue5::INITIAL_VERSION..=ue5::LATEST).contains(&s.ue5_version) {
            return Err(Error::new(
                12,
                format!(
                    "unsupported package version UE4 {} / UE5 {} (supported: UE4 522, UE5 1000..={})",
                    s.ue4_version,
                    s.ue5_version,
                    ue5::LATEST
                ),
            ));
        }
        let v5 = s.ue5_version;
        if v5 >= ue5::PACKAGE_SAVED_HASH {
            r.bytes(20)?; // FIoHash PackageSavedHash
            s.total_header_size = r.i32()?;
        }
        let n = r.count(20)?;
        for _ in 0..n {
            s.custom_versions.push((r.guid()?, r.i32()?));
        }
        if v5 < ue5::PACKAGE_SAVED_HASH {
            s.total_header_size = r.i32()?;
        }
        s.package_name = r.fstring()?;
        s.package_flags = r.u32()?;
        if s.package_flags & PKG_UNVERSIONED_PROPERTIES != 0 {
            return Err(r.err("package uses unversioned properties (cooked); only editor saves are supported"));
        }
        s.name_count = r.count(0)?;
        s.name_offset = offset(r)?;
        if v5 >= ue5::ADD_SOFTOBJECTPATH_LIST {
            s.soft_object_paths_count = r.count(0)?;
            s.soft_object_paths_offset = offset(r)?;
        }
        if !s.filter_editor_only() {
            s.localization_id = r.fstring()?;
        }
        let _gatherable_text_count = r.i32()?;
        let _gatherable_text_offset = r.i32()?;
        s.export_count = r.count(0)?;
        s.export_offset = offset(r)?;
        s.import_count = r.count(0)?;
        s.import_offset = offset(r)?;
        if v5 >= ue5::VERSE_CELLS {
            for _ in 0..4 {
                r.i32()?; // cell export/import count and offset
            }
        }
        if v5 >= ue5::METADATA_SERIALIZATION_OFFSET {
            r.i32()?; // MetaDataOffset
        }
        s.depends_offset = r.i32()?;
        let _soft_package_references_count = r.i32()?;
        let _soft_package_references_offset = r.i32()?;
        let _searchable_names_offset = r.i32()?;
        s.thumbnail_table_offset = r.i32()?;
        if v5 >= ue5::IMPORT_TYPE_HIERARCHIES {
            r.i32()?;
            r.i32()?;
        }
        if v5 < ue5::PACKAGE_SAVED_HASH {
            r.guid()?; // Guid
        }
        if !s.filter_editor_only() {
            r.guid()?; // PersistentGuid
        }
        let generations = r.count(8)?;
        for _ in 0..generations {
            r.i32()?; // ExportCount
            r.i32()?; // NameCount
        }
        s.saved_by = engine_version(r)?;
        s.compatible_with = engine_version(r)?;
        let compression_flags = r.u32()?;
        let chunks = r.i32()?;
        if compression_flags != 0 || chunks != 0 {
            return Err(r.err("compressed packages are not supported"));
        }
        let _package_source = r.u32()?;
        let additional = r.count(4)?;
        for _ in 0..additional {
            r.fstring()?;
        }
        s.asset_registry_data_offset = r.i32()?;
        s.bulk_data_start_offset = r.i64()?;
        let _world_tile_info_data_offset = r.i32()?;
        let chunk_ids = r.count(4)?;
        for _ in 0..chunk_ids {
            r.i32()?;
        }
        let _preload_dependency_count = r.i32()?;
        let _preload_dependency_offset = r.i32()?;
        if v5 >= ue5::NAMES_REFERENCED_FROM_EXPORT_DATA {
            r.i32()?;
        }
        if v5 >= ue5::PAYLOAD_TOC {
            r.i64()?;
        }
        if v5 >= ue5::DATA_RESOURCES {
            r.i32()?;
        }
        Ok(s)
    }
}

fn offset(r: &mut Reader) -> Result<usize> {
    let at = r.pos();
    let v = r.i32()?;
    if v < 0 {
        return Err(Error::new(at, format!("negative offset {v}")));
    }
    Ok(v as usize)
}

fn engine_version(r: &mut Reader) -> Result<EngineVersion> {
    Ok(EngineVersion {
        major: r.u16()?,
        minor: r.u16()?,
        patch: r.u16()?,
        changelist: r.u32()?,
        branch: r.fstring()?,
    })
}
