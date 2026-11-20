//! The single `unsafe` boundary of this crate.
//!
//! Every function here wraps exactly one Vulkan entry point (via `ash`) in
//! its own `unsafe` block with a SAFETY comment; no logic, no state. ash
//! marks each binding `unsafe` because the Vulkan C ABI carries non-local
//! validity rules (loader lifetime, handle validity, out-pointer writes).
//! Keeping the calls 1:1 in one audited module lets everything above stay
//! `#![deny(unsafe_code)]` (crate root) while each call stays reviewable.
//!
//! Contract: arguments must come from handles this crate created (see
//! [`crate::device`]); destruction happens in `VulkanDevice::drop` in
//! child-before-parent order (logical device, then instance; the loader
//! outlives both as the last-dropped field).
#![allow(unsafe_code)]

use crate::VkError;
use ash::{Device, Entry, Instance, vk};

/// dlopen the system Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`).
pub fn load_entry() -> Result<Entry, VkError> {
    // SAFETY: `Entry::load` only dlopens the loader and resolves its
    // function table; it issues no Vulkan commands. The returned `Entry`
    // uniquely owns the library handle (unloaded on drop).
    unsafe { Entry::load() }.map_err(|e| VkError::Loader(e.to_string()))
}

/// Report `vkEnumerateInstanceVersion`; `None` when the loader predates 1.1.
pub fn instance_version(entry: &Entry) -> Result<Option<u32>, VkError> {
    // SAFETY: `entry` carries a fully resolved loader function table (from
    // `load_entry`); the call writes only to its own returned value.
    unsafe { entry.try_enumerate_instance_version() }
        .map_err(|res| VkError::instance("vkEnumerateInstanceVersion", res))
}

/// Create a headless `VkInstance`: no extensions, no layers, API 1.3 requested.
pub fn create_instance(entry: &Entry) -> Result<Instance, VkError> {
    let app = vk::ApplicationInfo::default()
        .application_name(c"honkbox-ash-backend")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: `info` borrows `app`, alive for the whole call; no extensions
    // or layers are requested, so the loader resolves every function from
    // its own dispatch table (no unvalidated extension entry points). The
    // returned `Instance` is uniquely owned and destroyed by
    // `destroy_instance` before `entry` drops (field order in
    // `VulkanDevice`).
    unsafe { entry.create_instance(&info, None) }
        .map_err(|res| VkError::instance("vkCreateInstance", res))
}

/// Destroy a previously created instance (its children must already be gone).
pub fn destroy_instance(instance: &Instance) {
    // SAFETY: `instance` was created by `create_instance` from the same
    // loader; per spec no instance children may remain — guaranteed by
    // `VulkanDevice::drop` destroying the device first.
    unsafe { instance.destroy_instance(None) };
}

/// Enumerate every physical device visible to `instance`.
pub fn enumerate_physical_devices(instance: &Instance) -> Result<Vec<vk::PhysicalDevice>, VkError> {
    // SAFETY: `instance` is a valid, live handle from `create_instance`;
    // the call writes only to its returned `Vec`.
    unsafe { instance.enumerate_physical_devices() }
        .map_err(|res| VkError::instance("vkEnumeratePhysicalDevices", res))
}

/// Physical device identity and limits (pure query).
pub fn physical_device_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceProperties {
    // SAFETY: `physical_device` was enumerated from this live `instance`;
    // per spec a pure query writing only to its return value.
    unsafe { instance.get_physical_device_properties(physical_device) }
}

/// Memory heaps/types of `physical_device` (pure query; feeds VRAM math).
pub fn physical_device_memory_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceMemoryProperties {
    // SAFETY: same handle contract as `physical_device_properties`; pure query.
    unsafe { instance.get_physical_device_memory_properties(physical_device) }
}

/// Queue family capabilities of `physical_device` (pure query).
pub fn queue_family_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<vk::QueueFamilyProperties> {
    // SAFETY: same handle contract as `physical_device_properties`; pure query.
    unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
}

/// Storage-buffer 8/16-bit access bits of `physical_device`, queried via the
/// core 1.1 `vkGetPhysicalDeviceFeatures2` with both storage-feature structs
/// chained into `p_next`; the driver fills `storage8` / `storage16`.
pub fn physical_device_storage_features(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    storage8: &mut vk::PhysicalDevice8BitStorageFeatures<'_>,
    storage16: &mut vk::PhysicalDevice16BitStorageFeatures<'_>,
) {
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(storage8)
        .push_next(storage16);
    // SAFETY: `physical_device` was enumerated from this live `instance`;
    // `features2` chains only the two caller-owned structs, alive across
    // the call. Pure query.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
}

/// Subgroup properties of `physical_device`, queried via the core 1.1
/// `vkGetPhysicalDeviceProperties2` with the subgroup struct chained into
/// `p_next`; the driver fills `subgroup`.
pub fn physical_device_subgroup_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    subgroup: &mut vk::PhysicalDeviceSubgroupProperties<'_>,
) {
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(subgroup);
    // SAFETY: `physical_device` was enumerated from this live `instance`;
    // `properties2` chains only the caller-owned subgroup struct, alive
    // across the call. Pure query.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
}

/// Create a logical device owning exactly one compute queue from
/// `queue_family`, and retrieve that queue.
pub fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<(Device, vk::Queue), VkError> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities);
    let info =
        vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
    // SAFETY: `physical_device` was enumerated from this live `instance`;
    // `queue_family` is a valid family index of that device whose COMPUTE
    // bit the caller verified (see `PhysicalDeviceInfo::enumerate`). `info`
    // borrows only stack data alive across the call. The returned `Device`
    // is uniquely owned, destroyed by `destroy_device` before
    // `destroy_instance` runs.
    let device = unsafe { instance.create_device(physical_device, &info, None) }
        .map_err(|res| VkError::device("vkCreateDevice", res))?;
    // SAFETY: queue index 0 exists — `create_device` just created this
    // family with one queue (priorities table index 0).
    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    Ok((device, queue))
}

/// Destroy a previously created logical device (before its instance).
pub fn destroy_device(device: &Device) {
    // SAFETY: `device` was created by `create_logical_device`; per spec no
    // child objects may outlive it — none are ever created from it (P0).
    unsafe { device.destroy_device(None) };
}
