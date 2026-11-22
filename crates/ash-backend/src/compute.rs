//! Shared Vulkan compute scaffold plus the first elementwise kernel.
//!
//! [`Kernel`] owns the shader, pipeline, descriptor set and command buffer;
//! each op only validates, uploads, dispatches and decodes. All Vulkan calls
//! go through [`crate::raw`], so unsafe stays confined to that module.
//!
//! ponytail: host-visible buffers for every binding — no staging or
//! device-local memory yet; add those when transfer throughput matters.

use crate::{VkError, device::CapacityRequirements, device::VulkanDevice, raw};
use ash::vk;
use std::sync::{Arc, Mutex};

const SPIRV_MAGIC: u32 = 0x0723_0203;
/// `local_size_x` of `shaders/elementwise_add.comp`.
const ELEMENTWISE_LOCAL_SIZE: u32 = 64;

pub(crate) static ELEMENTWISE_SPIRV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/elementwise_add.spv"));
pub(crate) static MATMUL_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv"));
pub(crate) static SOFTMAX_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax.spv"));
pub(crate) static LAYER_NORM_SPIRV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/layer_norm.spv"));
pub(crate) static CONV2D_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv2d.spv"));
pub(crate) static SILU_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu.spv"));
pub(crate) static GROUP_NORM_SPIRV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/group_norm.spv"));

/// Reinterpret a little-endian SPIR-V byte stream as Vulkan words.
pub(crate) fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V must be word-aligned");
    let (words, trailing) = bytes.as_chunks::<4>();
    assert!(trailing.is_empty(), "SPIR-V must be word-aligned");
    let words: Vec<u32> = words.iter().map(|c| u32::from_le_bytes(*c)).collect();
    assert_eq!(
        words.first(),
        Some(&SPIRV_MAGIC),
        "build.rs must embed SPIR-V"
    );
    words
}

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Decode a GPU output buffer into f32 values.
pub(crate) fn f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(
        bytes.len() % size_of::<f32>(),
        0,
        "f32 output is word-aligned"
    );
    bytes
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// A host-visible storage buffer, bound and uploaded on construction.
pub(crate) struct GpuBuffer<'a> {
    device: &'a VulkanDevice,
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
}

impl<'a> GpuBuffer<'a> {
    pub(crate) fn new(device: &'a VulkanDevice, bytes: &[u8]) -> Result<Self, VkError> {
        let size =
            u64::try_from(bytes.len()).map_err(|_| VkError::Device("buffer too large".into()))?;
        let buffer =
            raw::create_buffer(device.logical(), size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let requirements = raw::buffer_memory_requirements(device.logical(), buffer);
        let properties =
            raw::physical_device_memory_properties(device.instance(), device.physical());
        let type_index = (0..properties.memory_type_count).find(|&i| {
            let index = usize::try_from(i).expect("memory type index fits usize");
            let kind = properties.memory_types[index];
            requirements.memory_type_bits & (1u32 << i) != 0
                && kind.property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        });
        let Some(type_index) = type_index else {
            raw::destroy_buffer(device.logical(), buffer);
            return Err(VkError::InsufficientCapacity {
                capability: "hostVisibleCoherentMemory",
                required: 1,
                available: 0,
            });
        };
        let memory = raw::allocate_memory(device.logical(), requirements.size, type_index)?;
        if let Err(error) = raw::bind_buffer_memory(device.logical(), buffer, memory, 0) {
            raw::free_memory(device.logical(), memory);
            raw::destroy_buffer(device.logical(), buffer);
            return Err(error);
        }
        if let Err(error) = raw::write_host_memory(device.logical(), memory, bytes) {
            raw::free_memory(device.logical(), memory);
            raw::destroy_buffer(device.logical(), buffer);
            return Err(error);
        }
        Ok(Self {
            device,
            buffer,
            memory,
        })
    }
}

impl Drop for GpuBuffer<'_> {
    fn drop(&mut self) {
        // Child-first: the buffer must not outlive its allocation.
        raw::destroy_buffer(self.device.logical(), self.buffer);
        raw::free_memory(self.device.logical(), self.memory);
    }
}

/// Reusable pipeline for a four-binding storage-buffer shader.
///
/// Binding 2 is the output in both current shaders; binding 3 carries shape
/// metadata (element count or matrix dimensions). At most one dispatch may
/// run at a time because the descriptor set and command buffer are reused.
pub(crate) struct Kernel {
    dispatch_lock: Mutex<()>,
    pub(crate) device: Arc<VulkanDevice>,
    module: vk::ShaderModule,
    layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
}

impl Kernel {
    pub(crate) fn new(device: Arc<VulkanDevice>, words: &[u32]) -> Result<Self, VkError> {
        let module = raw::create_shader_module(device.logical(), words)?;
        let layout = match raw::create_descriptor_set_layout(device.logical()) {
            Ok(layout) => layout,
            Err(e) => {
                raw::destroy_shader_module(device.logical(), module);
                return Err(e);
            }
        };
        let pipeline_layout = match raw::create_pipeline_layout(device.logical(), layout) {
            Ok(layout) => layout,
            Err(e) => {
                raw::destroy_descriptor_set_layout(device.logical(), layout);
                raw::destroy_shader_module(device.logical(), module);
                return Err(e);
            }
        };
        let pipeline = match raw::create_compute_pipeline(device.logical(), module, pipeline_layout)
        {
            Ok(pipeline) => pipeline,
            Err(e) => {
                raw::destroy_pipeline_layout(device.logical(), pipeline_layout);
                raw::destroy_descriptor_set_layout(device.logical(), layout);
                raw::destroy_shader_module(device.logical(), module);
                return Err(e);
            }
        };
        let pool = match raw::create_descriptor_pool(device.logical()) {
            Ok(pool) => pool,
            Err(e) => {
                Self::destroy_pipeline(&device, pipeline, pipeline_layout, layout, module);
                return Err(e);
            }
        };
        let set = match raw::allocate_descriptor_set(device.logical(), pool, layout) {
            Ok(set) => set,
            Err(e) => {
                raw::destroy_descriptor_pool(device.logical(), pool);
                Self::destroy_pipeline(&device, pipeline, pipeline_layout, layout, module);
                return Err(e);
            }
        };
        let command_pool =
            match raw::create_command_pool(device.logical(), device.compute_queue_family()) {
                Ok(pool) => pool,
                Err(e) => {
                    raw::destroy_descriptor_pool(device.logical(), pool);
                    Self::destroy_pipeline(&device, pipeline, pipeline_layout, layout, module);
                    return Err(e);
                }
            };
        let command = match raw::allocate_command_buffer(device.logical(), command_pool) {
            Ok(command) => command,
            Err(e) => {
                raw::destroy_command_pool(device.logical(), command_pool);
                raw::destroy_descriptor_pool(device.logical(), pool);
                Self::destroy_pipeline(&device, pipeline, pipeline_layout, layout, module);
                return Err(e);
            }
        };
        Ok(Self {
            dispatch_lock: Mutex::new(()),
            device,
            module,
            layout,
            pipeline_layout,
            pipeline,
            pool,
            set,
            command_pool,
            command,
        })
    }

    fn destroy_pipeline(
        device: &VulkanDevice,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        layout: vk::DescriptorSetLayout,
        module: vk::ShaderModule,
    ) {
        raw::destroy_pipeline(device.logical(), pipeline);
        raw::destroy_pipeline_layout(device.logical(), pipeline_layout);
        raw::destroy_descriptor_set_layout(device.logical(), layout);
        raw::destroy_shader_module(device.logical(), module);
    }

    /// Bind `buffers` to the four bindings, dispatch, fence-wait, read output.
    pub(crate) fn dispatch(
        &self,
        buffers: &[GpuBuffer<'_>],
        sizes: [u64; 4],
        output: usize,
        output_len: usize,
        workgroups: [u32; 3],
    ) -> Result<Vec<u8>, VkError> {
        let _guard = self
            .dispatch_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(buffers.len(), 4, "shaders declare four bindings");
        assert!(output < buffers.len(), "output binding must exist");
        let handles = [
            buffers[0].buffer,
            buffers[1].buffer,
            buffers[2].buffer,
            buffers[3].buffer,
        ];
        raw::write_storage_descriptors(self.device.logical(), self.set, handles, sizes);
        raw::reset_command_buffer(self.device.logical(), self.command)?;
        raw::record_dispatch(
            self.device.logical(),
            self.command,
            self.pipeline,
            self.pipeline_layout,
            self.set,
            workgroups,
        )?;
        let fence = raw::create_fence(self.device.logical())?;
        let submitted = raw::submit_and_wait(
            self.device.logical(),
            self.device.compute_queue(),
            self.command,
            fence,
        );
        raw::destroy_fence(self.device.logical(), fence);
        submitted?;
        raw::read_host_memory(self.device.logical(), buffers[output].memory, output_len)
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        let device = self.device.logical();
        raw::destroy_command_pool(device, self.command_pool);
        raw::destroy_descriptor_pool(device, self.pool);
        raw::destroy_pipeline(device, self.pipeline);
        raw::destroy_pipeline_layout(device, self.pipeline_layout);
        raw::destroy_descriptor_set_layout(device, self.layout);
        raw::destroy_shader_module(device, self.module);
    }
}

/// Reusable Vulkan pipeline for `result[i] = lhs[i] + rhs[i]`.
pub struct ElementwiseAdd {
    kernel: Kernel,
}

impl ElementwiseAdd {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(ELEMENTWISE_SPIRV))?,
        })
    }

    /// Dispatch the add on the device's compute queue and return the result.
    ///
    /// Rejects empty and unequal-length inputs; element count is passed to the
    /// shader so a non-multiple of the local size does not write out of bounds.
    pub fn run(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, VkError> {
        if lhs.is_empty() || lhs.len() != rhs.len() {
            return Err(VkError::Device(
                "elementwise_add: inputs must be non-empty and equal length".into(),
            ));
        }
        let count = u32::try_from(lhs.len())
            .map_err(|_| VkError::Device("elementwise_add: too many elements".into()))?;
        let byte_len = lhs
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("elementwise_add: byte length overflow".into()))?;
        let local_size = usize::try_from(ELEMENTWISE_LOCAL_SIZE).expect("64 fits usize");
        let workgroups = u32::try_from(lhs.len().div_ceil(local_size))
            .map_err(|_| VkError::Device("elementwise_add: too many workgroups".into()))?;
        let byte_len_u64 = u64::try_from(byte_len)
            .map_err(|_| VkError::Device("elementwise_add: buffer too large".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: byte_len_u64,
            ..CapacityRequirements::default()
        })?;

        let left = GpuBuffer::new(&self.kernel.device, &f32_bytes(lhs))?;
        let right = GpuBuffer::new(&self.kernel.device, &f32_bytes(rhs))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; byte_len])?;
        let length = GpuBuffer::new(&self.kernel.device, &count.to_le_bytes())?;
        let bytes = self.kernel.dispatch(
            &[left, right, output, length],
            [byte_len_u64, byte_len_u64, byte_len_u64, 4],
            2,
            byte_len,
            [workgroups, 1, 1],
        )?;
        Ok(f32_from_bytes(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_shaders_are_valid_spirv() {
        assert_eq!(spirv_words(ELEMENTWISE_SPIRV).first(), Some(&SPIRV_MAGIC));
        assert_eq!(spirv_words(MATMUL_SPIRV).first(), Some(&SPIRV_MAGIC));
        assert_eq!(spirv_words(SOFTMAX_SPIRV).first(), Some(&SPIRV_MAGIC));
        assert_eq!(spirv_words(LAYER_NORM_SPIRV).first(), Some(&SPIRV_MAGIC));
        assert_eq!(spirv_words(CONV2D_SPIRV).first(), Some(&SPIRV_MAGIC));
        assert_eq!(spirv_words(SILU_SPIRV).first(), Some(&SPIRV_MAGIC));
        assert_eq!(spirv_words(GROUP_NORM_SPIRV).first(), Some(&SPIRV_MAGIC));
    }

    #[test]
    fn elementwise_add_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let kernel = ElementwiseAdd::new(device).expect("pipeline creation");
        // 4097 is deliberately not a multiple of the 64-wide local size: it
        // exercises the shader's count guard and the workgroup ceil.
        let lhs = vec![1.5_f32; 4097];
        let rhs = vec![2.25_f32; 4097];
        assert_eq!(
            kernel.run(&lhs, &rhs).expect("gpu dispatch"),
            vec![3.75; 4097]
        );
        assert!(matches!(kernel.run(&[], &[]), Err(VkError::Device(_))));
    }
}
