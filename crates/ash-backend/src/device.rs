//! [`VulkanDevice`]: loader + headless instance + logical device + compute
//! queue, with explicit physical-device selection (properties, limits,
//! device-local VRAM) feeding capacity admission and bounded dispatch.
//!
//! P0 groundwork per `.tmp/research/ReviewedAsh-DirectVulkan.md` §6: init,
//! selection, and compute-queue acquisition only — no buffers, pools, or
//! dispatch yet (those arrive with the allocator work, P1+).

use crate::{VkError, raw};
use ash::{Device, Entry, Instance, vk};
use std::ffi::CStr;

/// Preference rank for [`PhysicalDeviceInfo::device_type`]; the derived
/// `Ord` follows variant order (discrete > integrated > virtual > cpu >
/// other), implementing ReviewedAsh-FeaturePlan's selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceType {
    /// Unknown or driver-reported other (ranked last).
    Other,
    /// Software implementation (lavapipe / CPU rasterizer).
    Cpu,
    /// Virtualized GPU (VM passthrough).
    Virtual,
    /// Integrated GPU (shares system memory).
    Integrated,
    /// Discrete GPU (dedicated memory) — preferred.
    Discrete,
}

impl DeviceType {
    /// Map a raw Vulkan device type; unknown values fall back to [`Self::Other`].
    pub fn from_raw(ty: vk::PhysicalDeviceType) -> Self {
        if ty == vk::PhysicalDeviceType::DISCRETE_GPU {
            Self::Discrete
        } else if ty == vk::PhysicalDeviceType::INTEGRATED_GPU {
            Self::Integrated
        } else if ty == vk::PhysicalDeviceType::VIRTUAL_GPU {
            Self::Virtual
        } else if ty == vk::PhysicalDeviceType::CPU {
            Self::Cpu
        } else {
            Self::Other
        }
    }
}

/// Capability bits reported by the driver and consumed by the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFeatures {
    /// `storageBuffer8BitAccess` (VK_KHR_8bit_storage).
    pub storage_8_bit: bool,
    /// `storageBuffer16BitAccess` (VK_KHR_16bit_storage).
    pub storage_16_bit: bool,
    /// Driver-reported subgroup size (`0` on pre-1.1 devices).
    pub subgroup_size: u32,
}

/// Minimum capabilities a model needs on the selected device before its
/// load is reserved. Defaults mean "no additional requirement".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapacityRequirements {
    /// Require 8-bit storage-buffer access.
    pub storage_8_bit: bool,
    /// Require 16-bit storage-buffer access.
    pub storage_16_bit: bool,
    /// Minimum subgroup size (`0` disables the check).
    pub subgroup_size: u32,
    /// Largest per-axis workgroup count the model will dispatch.
    pub workgroup_count: [u32; 3],
    /// Largest single storage-buffer binding the model will use.
    pub storage_range: u64,
}

/// A dispatch already clamped to the selected device's reported limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPlan {
    /// Per-axis workgroup counts, each within `maxComputeWorkGroupCount`.
    pub workgroups: [u32; 3],
    /// Storage binding size, within `maxStorageBufferRange`.
    pub storage_range: u64,
}

/// One physical device as the instance sees it: identity plus the numbers
/// the capacity ledger needs (device-local VRAM, limits, features).
#[derive(Debug, Clone)]
pub struct PhysicalDeviceInfo {
    /// `VkPhysicalDevice` handle (valid while the creating instance lives).
    pub handle: vk::PhysicalDevice,
    /// Driver-reported device name (invalid bytes replaced, per lossy UTF-8).
    pub name: String,
    /// Vendor PCI id (e.g. AMD `0x1002`, NVIDIA `0x10DE`).
    pub vendor_id: u32,
    /// Device PCI id.
    pub device_id: u32,
    /// Coarse kind driving the selection preference.
    pub device_type: DeviceType,
    /// Sum of `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT` heap sizes, in bytes.
    pub vram_bytes: u64,
    /// Full driver properties (API versions, vendor strings, ...).
    pub properties: vk::PhysicalDeviceProperties,
    /// Full limit block (`max_compute_*`, `max_storage_buffer_range`, ...).
    pub limits: vk::PhysicalDeviceLimits,
    /// Feature bits the admission gates check.
    pub features: DeviceFeatures,
    /// First queue family exposing `VK_QUEUE_COMPUTE_BIT`, if any.
    pub compute_queue_family: Option<u32>,
}

const WORKGROUP_AXES: [&str; 3] = [
    "maxComputeWorkGroupCount.x",
    "maxComputeWorkGroupCount.y",
    "maxComputeWorkGroupCount.z",
];

impl PhysicalDeviceInfo {
    /// Query every physical device visible to `instance`.
    pub fn enumerate(instance: &Instance) -> Result<Vec<Self>, VkError> {
        let handles = raw::enumerate_physical_devices(instance)?;
        Ok(handles
            .iter()
            .map(|handle| {
                let properties = raw::physical_device_properties(instance, *handle);
                let features = query_features(instance, *handle, properties.api_version);
                let name = device_name(&properties);
                let memory = raw::physical_device_memory_properties(instance, *handle);
                let vram_bytes = (0..memory.memory_heap_count)
                    .filter(|&i| {
                        memory.memory_heaps[i as usize]
                            .flags
                            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                    })
                    .map(|i| memory.memory_heaps[i as usize].size)
                    .sum();
                let compute_queue_family = raw::queue_family_properties(instance, *handle)
                    .iter()
                    .enumerate()
                    .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|(i, _)| u32::try_from(i).expect("queue family count fits u32"));
                Self {
                    handle: *handle,
                    name,
                    vendor_id: properties.vendor_id,
                    device_id: properties.device_id,
                    device_type: DeviceType::from_raw(properties.device_type),
                    vram_bytes,
                    properties,
                    limits: properties.limits,
                    features,
                    compute_queue_family,
                }
            })
            .collect())
    }

    /// Reject `req` when this device cannot satisfy it, naming the first
    /// missing feature or limit in [`VkError::InsufficientCapacity`].
    pub fn admit(&self, req: &CapacityRequirements) -> Result<(), VkError> {
        if self.compute_queue_family.is_none() {
            return Err(VkError::NoComputeQueue);
        }
        if req.storage_8_bit && !self.features.storage_8_bit {
            return Err(VkError::InsufficientCapacity {
                capability: "storageBuffer8BitAccess",
                required: 1,
                available: 0,
            });
        }
        if req.storage_16_bit && !self.features.storage_16_bit {
            return Err(VkError::InsufficientCapacity {
                capability: "storageBuffer16BitAccess",
                required: 1,
                available: 0,
            });
        }
        if req.subgroup_size > self.features.subgroup_size {
            return Err(VkError::InsufficientCapacity {
                capability: "subgroupSize",
                required: u64::from(req.subgroup_size),
                available: u64::from(self.features.subgroup_size),
            });
        }
        for (axis, &needed) in req.workgroup_count.iter().enumerate() {
            let available = self.limits.max_compute_work_group_count[axis];
            if needed > available {
                return Err(VkError::InsufficientCapacity {
                    capability: WORKGROUP_AXES[axis],
                    required: u64::from(needed),
                    available: u64::from(available),
                });
            }
        }
        let available = u64::from(self.limits.max_storage_buffer_range);
        if req.storage_range > available {
            return Err(VkError::InsufficientCapacity {
                capability: "maxStorageBufferRange",
                required: req.storage_range,
                available,
            });
        }
        Ok(())
    }

    /// Clamp a proposed dispatch to this device's reported limits.
    ///
    /// Returns `None` for an empty dispatch or a device without a compute
    /// queue; the plan can never exceed `maxComputeWorkGroupCount` or
    /// `maxStorageBufferRange`.
    pub fn bound_dispatch(&self, workgroups: [u32; 3], storage_range: u64) -> Option<DispatchPlan> {
        if self.compute_queue_family.is_none() || workgroups.iter().any(|&n| n == 0) {
            return None;
        }
        let max = self.limits.max_compute_work_group_count;
        Some(DispatchPlan {
            workgroups: [
                workgroups[0].min(max[0]),
                workgroups[1].min(max[1]),
                workgroups[2].min(max[2]),
            ],
            storage_range: storage_range.min(u64::from(self.limits.max_storage_buffer_range)),
        })
    }
}

/// Driver-reported device name as UTF-8 (Vulkan pads with NUL; lossy on the
/// rare non-UTF-8 driver string, empty when the NUL terminator is missing).
fn device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    let mut bytes = [0u8; vk::MAX_PHYSICAL_DEVICE_NAME_SIZE];
    for (dst, &src) in bytes.iter_mut().zip(properties.device_name.iter()) {
        // Bit-copy `i8` -> `u8` without a `as` cast (clippy cast_sign_loss).
        *dst = u8::from_le_bytes(src.to_le_bytes());
    }
    CStr::from_bytes_until_nul(&bytes)
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Query the capability bits the ledger gates on. Same 1:1 audited-query
/// shape as [`crate::raw`]; only plain data is read, no device objects made.
fn query_features(
    instance: &Instance,
    handle: vk::PhysicalDevice,
    api_version: u32,
) -> DeviceFeatures {
    // `vkGetPhysicalDeviceFeatures2` / `vkGetPhysicalDeviceProperties2` are
    // core only from Vulkan 1.1; no extension queries are enabled here.
    if api_version < vk::API_VERSION_1_1 {
        return DeviceFeatures {
            storage_8_bit: false,
            storage_16_bit: false,
            subgroup_size: 0,
        };
    }
    let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
    raw::physical_device_storage_features(instance, handle, &mut storage8, &mut storage16);
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    raw::physical_device_subgroup_properties(instance, handle, &mut subgroup);
    DeviceFeatures {
        storage_8_bit: storage8.storage_buffer8_bit_access != 0,
        storage_16_bit: storage16.storage_buffer16_bit_access != 0,
        subgroup_size: subgroup.subgroup_size,
    }
}

/// Selection policy (ReviewedAsh-FeaturePlan §6): prefer discrete over
/// integrated over cpu, then more device-local VRAM, then the lower device
/// id as a deterministic tie-break. Only compute-capable devices qualify —
/// a device without a compute queue cannot host the future kernels.
///
/// Caveat: `device_id` is the vendor PCI device id, so two identical GPU
/// models (same type, VRAM, and device_id) tie completely and `max_by_key`
/// then falls back to driver enumeration order (`max_by_key` returns the
/// last of equal maxima). A stable tie-break needs a per-device UUID (e.g. VK_KHR deviceUUID), deferred until the instance
/// requests extensions (P1 allocator work).
pub fn select_device(devices: &[PhysicalDeviceInfo]) -> Result<&PhysicalDeviceInfo, VkError> {
    if devices.is_empty() {
        return Err(VkError::NoDevices);
    }
    devices
        .iter()
        .filter(|d| d.compute_queue_family.is_some())
        .max_by_key(|d| (d.device_type, d.vram_bytes, std::cmp::Reverse(d.device_id)))
        .ok_or(VkError::NoComputeQueue)
}

/// Owned Vulkan context: loader, headless instance, logical device, and the
/// compute queue future pipelines dispatch on. Destruction order (logical
/// device → instance → loader) is enforced in `Drop`.
pub struct VulkanDevice {
    /// Vulkan loader entry. Never read after init: it exists so the dlopened
    /// loader library outlives every instance/device handle — it is the
    /// first-declared field, dropped last (after `Drop` tears down the
    /// device and instance).
    #[allow(dead_code)]
    entry: Entry,
    instance: Instance,
    device: Device,
    /// Handle of the single compute queue created from
    /// [`PhysicalDeviceInfo::compute_queue_family`].
    queue: vk::Queue,
    /// Family index of [`Self::queue`].
    queue_family: u32,
    /// Version from `vkEnumerateInstanceVersion` (`None`: 1.0-only loader);
    /// surfaced for the future `doctor` probe (ReviewedAsh-FeaturePlan §2).
    pub instance_version: Option<u32>,
    /// The selected physical device.
    pub info: PhysicalDeviceInfo,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ash handles carry no Debug impl; report the identity instead.
        f.debug_struct("VulkanDevice")
            .field("name", &self.info.name)
            .field("device_type", &self.info.device_type)
            .field("vram_bytes", &self.info.vram_bytes)
            .field("queue_family", &self.queue_family)
            .field("instance_version", &self.instance_version)
            .finish_non_exhaustive()
    }
}

impl VulkanDevice {
    /// Bring up the full headless chain: loader → instance → enumerate →
    /// select → logical device + compute queue.
    ///
    /// Errors: [`VkError::Loader`] (no system libvulkan),
    /// [`VkError::NoDevices`], [`VkError::NoComputeQueue`], or wrapped API
    /// failures ([`VkError::Instance`] / [`VkError::Device`]).
    pub fn headless() -> Result<Self, VkError> {
        let entry = raw::load_entry()?;
        let instance_version = raw::instance_version(&entry)?;
        let instance = raw::create_instance(&entry)?;
        let devices = PhysicalDeviceInfo::enumerate(&instance)?;
        let info = select_device(&devices)?.clone();
        let queue_family = info
            .compute_queue_family
            .expect("select_device only returns compute-capable devices");
        let (device, queue) = raw::create_logical_device(&instance, info.handle, queue_family)?;
        Ok(Self {
            entry,
            instance,
            device,
            queue,
            queue_family,
            instance_version,
            info,
        })
    }

    /// The logical device future pipelines allocate and dispatch on.
    pub fn logical(&self) -> &Device {
        &self.device
    }

    /// The compute queue.
    pub fn compute_queue(&self) -> vk::Queue {
        self.queue
    }

    /// Family index of [`Self::compute_queue`].
    pub fn compute_queue_family(&self) -> u32 {
        self.queue_family
    }

    /// The selected physical device handle.
    pub fn physical(&self) -> vk::PhysicalDevice {
        self.info.handle
    }

    /// Build a dispatch that cannot exceed the selected device's reported
    /// workgroup-count or storage-range limits.
    pub fn dispatch_plan(&self, workgroups: [u32; 3], storage_range: u64) -> Option<DispatchPlan> {
        self.info.bound_dispatch(workgroups, storage_range)
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Child-first destruction: the logical device must die before the
        // instance; the loader (`entry`) drops last via normal field drop.
        raw::destroy_device(&self.device);
        raw::destroy_instance(&self.instance);
    }
}

// `SdPipeline` and the future LLM/STT/TTS seams require `Send + Sync`; ash
// handles are plain wrappers over driver tables with an owned loader field,
// so this must keep holding as fields grow.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VulkanDevice>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    /// Placeholder row for the pure selection tests (no Vulkan needed).
    fn info(device_type: DeviceType, vram: u64, id: u32, compute: bool) -> PhysicalDeviceInfo {
        PhysicalDeviceInfo {
            handle: vk::PhysicalDevice::default(),
            name: format!("fake-{id}"),
            vendor_id: 0,
            device_id: id,
            device_type,
            vram_bytes: vram,
            properties: vk::PhysicalDeviceProperties::default(),
            limits: vk::PhysicalDeviceLimits::default(),
            features: DeviceFeatures {
                storage_8_bit: true,
                storage_16_bit: true,
                subgroup_size: 8,
            },
            compute_queue_family: compute.then_some(0),
        }
    }

    #[test]
    fn device_type_order_is_discrete_integrated_cpu() {
        assert!(DeviceType::Discrete > DeviceType::Integrated);
        assert!(DeviceType::Integrated > DeviceType::Cpu);
        assert!(DeviceType::Cpu > DeviceType::Other);
    }

    #[test]
    fn selection_prefers_discrete_then_most_vram() {
        let devices = [
            info(DeviceType::Integrated, 16 << 30, 1, true),
            info(DeviceType::Discrete, 8 << 30, 2, true),
            info(DeviceType::Discrete, 24 << 30, 3, true),
            info(DeviceType::Cpu, 1 << 30, 4, true),
        ];
        let picked = select_device(&devices).expect("a compute-capable device exists");
        assert_eq!(picked.device_id, 3, "discrete with most VRAM wins");
        assert_eq!(picked.name, "fake-3");
    }

    #[test]
    fn selection_skips_devices_without_compute_queue() {
        let devices = [info(DeviceType::Discrete, 8 << 30, 1, false)];
        assert!(matches!(
            select_device(&devices),
            Err(VkError::NoComputeQueue)
        ));
    }

    #[test]
    fn selection_between_identical_devices_is_enumeration_order_dependent() {
        // Two identical GPU models tie on (type, VRAM, device_id); document
        // that `max_by_key` then resolves the tie by enumeration order.
        // Distinct handles make the pick observable: all key fields tie, so
        // which element wins is decided purely by iteration order.
        // `Iterator::max_by_key` returns the LAST of equal maxima, so the
        // second array element wins in both orders.
        let mut order_a = [
            info(DeviceType::Discrete, 8 << 30, 7, true),
            info(DeviceType::Discrete, 8 << 30, 7, true),
        ];
        order_a[0].handle = vk::PhysicalDevice::from_raw(1);
        order_a[1].handle = vk::PhysicalDevice::from_raw(2);
        let mut order_b = [
            info(DeviceType::Discrete, 8 << 30, 7, true),
            info(DeviceType::Discrete, 8 << 30, 7, true),
        ];
        order_b[0].handle = vk::PhysicalDevice::from_raw(1);
        order_b[1].handle = vk::PhysicalDevice::from_raw(2);
        assert!(select_device(&order_a).is_ok());
        assert!(select_device(&order_b).is_ok());
        assert_eq!(select_device(&order_a).unwrap().handle, order_a[1].handle);
        assert_eq!(select_device(&order_b).unwrap().handle, order_b[1].handle);
    }

    #[test]
    fn selection_reports_empty_enumeration() {
        assert!(matches!(select_device(&[]), Err(VkError::NoDevices)));
    }

    fn demanding_req() -> CapacityRequirements {
        CapacityRequirements {
            storage_8_bit: true,
            storage_16_bit: true,
            subgroup_size: 8,
            workgroup_count: [8, 8, 1],
            storage_range: 1024,
        }
    }

    #[test]
    fn admit_accepts_default_requirements() {
        let dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        assert_eq!(dev.admit(&CapacityRequirements::default()), Ok(()));
    }

    #[test]
    fn admit_names_missing_8_bit_storage() {
        let mut dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        dev.features.storage_8_bit = false;
        assert_eq!(
            dev.admit(&demanding_req()),
            Err(VkError::InsufficientCapacity {
                capability: "storageBuffer8BitAccess",
                required: 1,
                available: 0,
            })
        );
    }

    #[test]
    fn admit_names_missing_16_bit_storage() {
        let mut dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        dev.features.storage_16_bit = false;
        assert!(matches!(
            dev.admit(&demanding_req()),
            Err(VkError::InsufficientCapacity {
                capability: "storageBuffer16BitAccess",
                ..
            })
        ));
    }

    #[test]
    fn admit_names_small_subgroup_size() {
        let mut dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        dev.features.subgroup_size = 4;
        assert_eq!(
            dev.admit(&demanding_req()),
            Err(VkError::InsufficientCapacity {
                capability: "subgroupSize",
                required: 8,
                available: 4,
            })
        );
    }

    #[test]
    fn admit_names_workgroup_count_over_limit() {
        let mut dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        dev.limits.max_compute_work_group_count = [4, 8, 8];
        assert_eq!(
            dev.admit(&demanding_req()),
            Err(VkError::InsufficientCapacity {
                capability: "maxComputeWorkGroupCount.x",
                required: 8,
                available: 4,
            })
        );
    }

    #[test]
    fn admit_names_storage_range_over_limit() {
        let mut dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        dev.limits.max_compute_work_group_count = [16, 16, 16];
        dev.limits.max_storage_buffer_range = 512;
        assert_eq!(
            dev.admit(&demanding_req()),
            Err(VkError::InsufficientCapacity {
                capability: "maxStorageBufferRange",
                required: 1024,
                available: 512,
            })
        );
    }

    #[test]
    fn admit_rejects_device_without_compute_queue() {
        let dev = info(DeviceType::Cpu, 1 << 30, 1, false);
        assert_eq!(dev.admit(&demanding_req()), Err(VkError::NoComputeQueue));
    }

    #[test]
    fn bound_dispatch_clamps_to_reported_limits() {
        let mut dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        dev.limits.max_compute_work_group_count = [4, 8, 16];
        dev.limits.max_storage_buffer_range = 512;
        assert_eq!(
            dev.bound_dispatch([9, 5, 2], 4096),
            Some(DispatchPlan {
                workgroups: [4, 5, 2],
                storage_range: 512,
            })
        );
    }

    #[test]
    fn bound_dispatch_rejects_empty_or_queue_less() {
        let mut dev = info(DeviceType::Cpu, 1 << 30, 1, true);
        assert_eq!(dev.bound_dispatch([0, 1, 1], 0), None);
        dev.compute_queue_family = None;
        assert_eq!(dev.bound_dispatch([1, 1, 1], 0), None);
    }

    #[test]
    fn insufficient_capacity_error_names_capability() {
        let err = VkError::InsufficientCapacity {
            capability: "storageBuffer8BitAccess",
            required: 1,
            available: 0,
        };
        assert!(err.to_string().contains("storageBuffer8BitAccess"));
        let err = VkError::InsufficientCapacity {
            capability: "maxStorageBufferRange",
            required: 1024,
            available: 512,
        };
        assert!(err.to_string().contains("maxStorageBufferRange"));
    }

    #[test]
    fn headless_init_smoke() {
        // P0 smoke: loader → instance → select → compute queue. Skips
        // gracefully on loader-less machines; the default CI matrix builds
        // without `experimental-ash-sd`, so this only runs in opt-in
        // (--features) local/dev runs.
        match VulkanDevice::headless() {
            Ok(dev) => {
                assert_ne!(dev.info.name, "");
                // lavapipe and real GPUs both expose a device-local heap;
                // soften for exotic drivers reporting zero.
                assert!(dev.info.vram_bytes > 0 || dev.info.device_type == DeviceType::Cpu);
                assert_ne!(dev.compute_queue_family(), u32::MAX);
                eprintln!(
                    "headless smoke: selected {} ({:?}, {} MiB device-local, instance version {:?})",
                    dev.info.name,
                    dev.info.device_type,
                    dev.info.vram_bytes / (1 << 20),
                    dev.instance_version,
                );
            }
            // Unavailable environment: not a test failure.
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping headless smoke: no usable Vulkan environment");
            }
            // Real API failures indicate a bug in this crate: fail loudly.
            Err(
                e @ (VkError::Instance(_)
                | VkError::Device(_)
                | VkError::InsufficientCapacity { .. }),
            ) => {
                panic!("vulkan api failed during headless smoke: {e:?}")
            }
        }
    }
}
