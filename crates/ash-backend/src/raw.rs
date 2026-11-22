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

// --- P1 compute path: one audited wrapper per Vulkan entry point. -----------
// All resource lifetime decisions stay in `crate::compute`; this module only
// translates a safe call into one C ABI call with a SAFETY note.

/// Upload SPIR-V words into a shader module.
pub fn create_shader_module(device: &Device, code: &[u32]) -> Result<vk::ShaderModule, VkError> {
    let info = vk::ShaderModuleCreateInfo::default().code(code);
    // SAFETY: `code` is a caller-owned slice of SPIR-V words alive for the
    // call; ash copies it into the driver. The handle is owned by the caller.
    unsafe { device.create_shader_module(&info, None) }
        .map_err(|res| VkError::device("vkCreateShaderModule", res))
}

/// Destroy a shader module after its pipelines are gone.
pub fn destroy_shader_module(device: &Device, module: vk::ShaderModule) {
    // SAFETY: caller owns `module` and guarantees no pipeline references it.
    unsafe { device.destroy_shader_module(module, None) };
}

/// Four storage-buffer bindings: lhs, rhs, result, element count.
pub fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VkError> {
    let bindings: Vec<_> = (0..4)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: `info` borrows only the stack-local `bindings`, alive for the
    // call; the returned handle is owned by the caller.
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|res| VkError::device("vkCreateDescriptorSetLayout", res))
}

/// Destroy a descriptor set layout after its sets and pipelines are gone.
pub fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    // SAFETY: caller owns `layout`; no live set or pipeline references it.
    unsafe { device.destroy_descriptor_set_layout(layout, None) };
}

/// Pipeline layout holding the single descriptor set used by all ops.
pub fn create_pipeline_layout(
    device: &Device,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VkError> {
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&layout));
    // SAFETY: `layout` is a live handle owned by the caller and outlives the
    // returned layout (enforced by the caller's Drop order).
    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|res| VkError::device("vkCreatePipelineLayout", res))
}

/// Destroy a pipeline layout after its pipelines are gone.
pub fn destroy_pipeline_layout(device: &Device, layout: vk::PipelineLayout) {
    // SAFETY: caller owns `layout`; pipelines were destroyed first.
    unsafe { device.destroy_pipeline_layout(layout, None) };
}

/// Create the single compute pipeline from `module` and `layout`.
pub fn create_compute_pipeline(
    device: &Device,
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, VkError> {
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    // SAFETY: `module` and `layout` are live caller-owned handles; the
    // one-element create info is alive for the call. Exactly one pipeline is
    // requested, so the success branch always has element 0.
    unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map(|mut pipelines| {
        pipelines
            .pop()
            .expect("one pipeline was requested and creation succeeded")
    })
    .map_err(|(_, res)| VkError::device("vkCreateComputePipelines", res))
}

/// Destroy a compute pipeline.
pub fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline) {
    // SAFETY: caller owns `pipeline`; no command buffer is still using it.
    unsafe { device.destroy_pipeline(pipeline, None) };
}

/// Allocate a storage buffer of `size` bytes.
pub fn create_buffer(
    device: &Device,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<vk::Buffer, VkError> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `info` is stack-local and alive for the call; the returned
    // buffer is owned by the caller and must be bound before use.
    unsafe { device.create_buffer(&info, None) }
        .map_err(|res| VkError::device("vkCreateBuffer", res))
}

/// Destroy a buffer before its backing memory is freed.
pub fn destroy_buffer(device: &Device, buffer: vk::Buffer) {
    // SAFETY: caller owns `buffer`; no in-flight command references it.
    unsafe { device.destroy_buffer(buffer, None) };
}

/// Memory requirements (size, alignment, allowed type bits) of `buffer`.
pub fn buffer_memory_requirements(device: &Device, buffer: vk::Buffer) -> vk::MemoryRequirements {
    // SAFETY: `buffer` is a live caller-owned handle; pure query.
    unsafe { device.get_buffer_memory_requirements(buffer) }
}

/// Allocate `size` bytes from memory type `type_index`.
pub fn allocate_memory(
    device: &Device,
    size: u64,
    type_index: u32,
) -> Result<vk::DeviceMemory, VkError> {
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(type_index);
    // SAFETY: `info` is stack-local; `type_index` was selected from the
    // physical device's queried memory types by the caller.
    unsafe { device.allocate_memory(&info, None) }
        .map_err(|res| VkError::device("vkAllocateMemory", res))
}

/// Release device memory after every buffer bound to it is destroyed.
pub fn free_memory(device: &Device, memory: vk::DeviceMemory) {
    // SAFETY: caller owns `memory`; no live buffer or mapping references it.
    unsafe { device.free_memory(memory, None) };
}

/// Bind a buffer to the whole of a compatible allocation.
pub fn bind_buffer_memory(
    device: &Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    offset: u64,
) -> Result<(), VkError> {
    // SAFETY: both handles are live and owned by the caller; `offset` was
    // chosen to respect the queried alignment (zero in this crate).
    unsafe { device.bind_buffer_memory(buffer, memory, offset) }
        .map_err(|res| VkError::device("vkBindBufferMemory", res))
}

/// Copy host bytes into a mapped host-visible allocation.
pub fn write_host_memory(
    device: &Device,
    memory: vk::DeviceMemory,
    bytes: &[u8],
) -> Result<(), VkError> {
    let size =
        u64::try_from(bytes.len()).map_err(|_| VkError::Device("buffer too large".into()))?;
    // SAFETY: `memory` is host-visible + host-coherent and `size` was taken
    // from its allocation; the mapping is unmapped before this returns.
    let ptr = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
        .map_err(|res| VkError::device("vkMapMemory", res))?;
    // SAFETY: `ptr` points to `bytes.len()` writable mapped bytes; `bytes`
    // is a disjoint caller-owned slice of the same length.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len()) };
    // SAFETY: the mapping created just above is still current.
    unsafe { device.unmap_memory(memory) };
    Ok(())
}

/// Read `len` bytes from a mapped host-visible allocation.
pub fn read_host_memory(
    device: &Device,
    memory: vk::DeviceMemory,
    len: usize,
) -> Result<Vec<u8>, VkError> {
    let size = u64::try_from(len).map_err(|_| VkError::Device("buffer too large".into()))?;
    // SAFETY: see `write_host_memory`; host-coherent means no invalidate is
    // needed after the GPU fence has signaled.
    let ptr = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
        .map_err(|res| VkError::device("vkMapMemory", res))?;
    // SAFETY: `ptr` is valid for `len` reads until unmapped; `from_raw_parts`
    // copies the bytes into an owned Vec before the unmap.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) }.to_vec();
    // SAFETY: the mapping created just above is still current.
    unsafe { device.unmap_memory(memory) };
    Ok(bytes)
}

/// Pool for the one descriptor set every op reuses.
pub fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VkError> {
    let sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(4)];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&sizes);
    // SAFETY: `info` borrows only stack-local data; the pool is caller-owned.
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|res| VkError::device("vkCreateDescriptorPool", res))
}

/// Destroy the pool (and implicitly every set allocated from it).
pub fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    // SAFETY: caller owns `pool`; no command buffer is still using its sets.
    unsafe { device.destroy_descriptor_pool(pool, None) };
}

/// Allocate the single descriptor set from `pool` against `layout`.
pub fn allocate_descriptor_set(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, VkError> {
    let layouts = [layout];
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    // SAFETY: `layout` is live, `pool` has room for one set, and `info`
    // borrows only stack data; the set is owned by `pool`.
    unsafe { device.allocate_descriptor_sets(&info) }
        .map_err(|res| VkError::device("vkAllocateDescriptorSets", res))?
        .into_iter()
        .next()
        .ok_or_else(|| VkError::Device("vkAllocateDescriptorSets returned no set".into()))
}

/// Point the four bindings at the current op's buffers.
pub fn write_storage_descriptors(
    device: &Device,
    set: vk::DescriptorSet,
    buffers: [vk::Buffer; 4],
    sizes: [u64; 4],
) {
    let infos: Vec<_> = buffers
        .iter()
        .zip(sizes)
        .map(|(&buffer, range)| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(range)
        })
        .collect();
    let writes: Vec<_> = infos
        .iter()
        .enumerate()
        .map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(u32::try_from(binding).expect("four bindings"))
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    // SAFETY: `set` is live, every buffer stays alive until the GPU fence
    // signals, and all borrowed arrays outlive the call.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Command pool for the compute queue, resettable so one buffer is reused.
pub fn create_command_pool(device: &Device, family: u32) -> Result<vk::CommandPool, VkError> {
    let info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    // SAFETY: `family` was verified compute-capable during device selection;
    // `info` is stack-local; the pool is caller-owned.
    unsafe { device.create_command_pool(&info, None) }
        .map_err(|res| VkError::device("vkCreateCommandPool", res))
}

/// Destroy the command pool (and its command buffers).
pub fn destroy_command_pool(device: &Device, pool: vk::CommandPool) {
    // SAFETY: caller owns `pool`; no submitted buffer is still in flight.
    unsafe { device.destroy_command_pool(pool, None) };
}

/// Allocate one reusable primary command buffer.
pub fn allocate_command_buffer(
    device: &Device,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, VkError> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: `pool` is live, resettable, and belongs to the compute queue;
    // the returned buffer is owned by the pool.
    unsafe { device.allocate_command_buffers(&info) }
        .map_err(|res| VkError::device("vkAllocateCommandBuffers", res))?
        .into_iter()
        .next()
        .ok_or_else(|| VkError::Device("vkAllocateCommandBuffers returned none".into()))
}

/// Prepare a previously submitted command buffer for re-recording.
pub fn reset_command_buffer(device: &Device, command: vk::CommandBuffer) -> Result<(), VkError> {
    // SAFETY: `command` came from a RESET_COMMAND_BUFFER pool and the prior
    // submission's fence was waited on by the caller.
    unsafe { device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()) }
        .map_err(|res| VkError::device("vkResetCommandBuffer", res))
}

/// Record bind + dispatch + end into `command`.
pub fn record_dispatch(
    device: &Device,
    command: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
    workgroups: [u32; 3],
) -> Result<(), VkError> {
    // SAFETY: all handles are live caller-owned objects created for the same
    // device and queue family; the command buffer is not currently recording
    // or in flight. Descriptor buffers outlive the fence wait.
    let begin = vk::CommandBufferBeginInfo::default();
    unsafe { device.begin_command_buffer(command, &begin) }
        .map_err(|res| VkError::device("vkBeginCommandBuffer", res))?;
    unsafe {
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            std::slice::from_ref(&set),
            &[],
        );
        device.cmd_dispatch(command, workgroups[0], workgroups[1], workgroups[2]);
    }
    unsafe { device.end_command_buffer(command) }
        .map_err(|res| VkError::device("vkEndCommandBuffer", res))
}

/// An unsignaled fence for one submission.
pub fn create_fence(device: &Device) -> Result<vk::Fence, VkError> {
    let info = vk::FenceCreateInfo::default();
    // SAFETY: `info` is stack-local; the fence is caller-owned and destroyed
    // after the wait.
    unsafe { device.create_fence(&info, None) }.map_err(|res| VkError::device("vkCreateFence", res))
}

/// Destroy a fence after its submission completed.
pub fn destroy_fence(device: &Device, fence: vk::Fence) {
    // SAFETY: caller owns `fence` and already waited on it.
    unsafe { device.destroy_fence(fence, None) };
}

/// Submit `command` on `queue` and block until its fence signals.
pub fn submit_and_wait(
    device: &Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    fence: vk::Fence,
) -> Result<(), VkError> {
    let submits = [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command))];
    // SAFETY: the command buffer was recorded for this queue's family, all
    // referenced resources are still alive, and `fence` is unsignaled.
    unsafe { device.queue_submit(queue, &submits, fence) }
        .map_err(|res| VkError::device("vkQueueSubmit", res))?;
    // SAFETY: `fence` belongs to this queue submission; `wait_all = true`
    // with a single fence is exact, and the timeout only bounds the wait.
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX) }
        .map_err(|res| VkError::device("vkWaitForFences", res))
}

// --- Capability discovery (queries only; no device objects created). --------

/// Shader float16/int8 support, queried with core `vkGetPhysicalDeviceFeatures2`.
pub fn physical_device_shader_features(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    shader: &mut vk::PhysicalDeviceShaderFloat16Int8Features<'_>,
) {
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(shader);
    // SAFETY: `physical_device` was enumerated from this live `instance`;
    // `features2` chains only the caller-owned struct, alive across the call.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
}

/// Names of all device extensions exposed by `physical_device`.
pub fn enumerate_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<Vec<String>, VkError> {
    // SAFETY: `physical_device` was enumerated from this live `instance`;
    // the call writes only its returned vector. No layer name is requested.
    let properties = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .map_err(|res| VkError::instance("vkEnumerateDeviceExtensionProperties", res))?;
    Ok(properties
        .iter()
        .filter_map(|property| {
            property
                .extension_name_as_c_str()
                .ok()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .collect())
}
