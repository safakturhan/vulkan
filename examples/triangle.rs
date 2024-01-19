use std::{ffi::CString, mem, slice, sync::Arc};

use vulkan::{device::Device, instance::Instance, surface::Surface, swapchain::Swapchain};

use ash::{util::Align, vk};
use memoffset::offset_of;
use raw_window_handle::HasWindowHandle;
use winit::{
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

struct Renderer {
    surface: Arc<Surface>,
    device: Arc<Device>,
    swapchain: Swapchain,
    image_views: Vec<vk::ImageView>,
    vertex_buffer: vk::Buffer,
    command_buffer: vk::CommandBuffer,
    queue_submit_fence: vk::Fence,
    acquire_semaphore: vk::Semaphore,
    release_semaphore: vk::Semaphore,
    graphics_pipeline: vk::Pipeline,
}

impl Renderer {
    fn new(window: &impl HasWindowHandle) -> anyhow::Result<Self> {
        let instance = Arc::new(Instance::new()?);
        let surface = Arc::new(Surface::new(&window, &instance)?);
        let physical_device = instance
            .physical_devices()?
            .into_iter()
            .find(|physical_device| {
                physical_device
                    .properties
                    .device_type
                    .eq(&vk::PhysicalDeviceType::DISCRETE_GPU)
            })
            .expect("Failed to find discrete gpu");
        let device = Arc::new(Device::new(instance.clone(), physical_device, &surface)?);
        let swapchain = Swapchain::new(&device, &surface)?;

        let image_views: Vec<vk::ImageView> = create_image_views(&device, &swapchain);

        let memory_properties = &device.physical_device.memory_properties;

        let vertices = [
            Vertex {
                position: [0.0, -0.5, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.0],
                color: [0.0, 0.0, 1.0],
            },
        ];

        let vertex_buffer = {
            let create_info = vk::BufferCreateInfo::builder()
                .size(mem::size_of_val(&vertices) as u64)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            unsafe { device.handle.create_buffer(&create_info, None)? }
        };
        let vertex_buffer_memory = {
            let memory_requirements =
                unsafe { device.handle.get_buffer_memory_requirements(vertex_buffer) };
            let memory_type_index = find_memory_type_index(
                &memory_requirements,
                memory_properties,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("Failed to find memory type index");
            let allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(memory_requirements.size)
                .memory_type_index(memory_type_index);
            let memory = unsafe { device.handle.allocate_memory(&allocate_info, None)? };

            let ptr = unsafe {
                device.handle.map_memory(
                    memory,
                    0,
                    memory_requirements.size,
                    vk::MemoryMapFlags::empty(),
                )?
            };
            let mut vertex_align = unsafe {
                Align::new(
                    ptr,
                    mem::align_of::<Vertex>() as u64,
                    memory_requirements.size,
                )
            };
            vertex_align.copy_from_slice(&vertices);
            unsafe {
                device.handle.unmap_memory(memory);
                device.handle.bind_buffer_memory(vertex_buffer, memory, 0)?;
            }
            memory
        };
        _ = vertex_buffer_memory;

        let command_pool = unsafe {
            device.handle.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(0)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let command_buffer = unsafe {
            device.handle.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };

        let queue_submit_fence = unsafe {
            device.handle.create_fence(
                &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        let acquire_semaphore = unsafe {
            device
                .handle
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };
        let release_semaphore = unsafe {
            device
                .handle
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };

        let graphics_pipeline = create_graphics_pipeline(&device, swapchain.extent)?;

        Ok(Self {
            surface,
            device,
            swapchain,
            image_views,
            vertex_buffer,
            command_buffer,
            queue_submit_fence,
            acquire_semaphore,
            release_semaphore,
            graphics_pipeline,
        })
    }

    fn resize(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.device.handle.device_wait_idle()?;
            self.swapchain = {
                self.swapchain
                    .functions
                    .destroy_swapchain(self.swapchain.handle, None);
                Swapchain::new(&self.device, &self.surface)?
            };
            self.image_views = {
                for &image_view in &self.image_views {
                    self.device.handle.destroy_image_view(image_view, None);
                }
                create_image_views(&self.device, &self.swapchain)
            };
        }
        Ok(())
    }

    fn draw(&mut self) -> anyhow::Result<()> {
        let extent = self.swapchain.extent;
        let device = &self.device;

        unsafe {
            device.handle.wait_for_fences(
                slice::from_ref(&self.queue_submit_fence),
                true,
                u64::MAX,
            )?;
            device
                .handle
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.handle.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        let index = unsafe {
            self.swapchain
                .functions
                .acquire_next_image(
                    self.swapchain.handle,
                    u64::MAX,
                    self.acquire_semaphore,
                    vk::Fence::default(),
                )
                .map(|(index, _)| index)?
        };

        let color_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .base_array_layer(0)
            .layer_count(vk::REMAINING_ARRAY_LAYERS)
            .build();

        unsafe {
            device.handle.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                slice::from_ref(
                    &vk::ImageMemoryBarrier::builder()
                        .image(self.swapchain.images[index as usize])
                        .src_access_mask(vk::AccessFlags::NONE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .subresource_range(color_range),
                ),
            );
        }

        let image_view = self.image_views[index as usize];
        let render_area = vk::Rect2D::builder().extent(extent).build();
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.2, 0.4, 1.0],
            },
        };
        let color_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::NONE)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);
        let rendering_info = vk::RenderingInfoKHR::builder()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attachment));
        unsafe {
            device
                .dynamic_rendering
                .cmd_begin_rendering(self.command_buffer, &rendering_info);
            device.handle.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
            device.handle.cmd_set_viewport(
                self.command_buffer,
                0,
                slice::from_ref(
                    &vk::Viewport::builder()
                        .width(extent.width as f32)
                        .height(extent.height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0),
                ),
            );
            device.handle.cmd_set_scissor(
                self.command_buffer,
                0,
                slice::from_ref(&vk::Rect2D::builder().extent(extent)),
            );
            device.handle.cmd_bind_vertex_buffers(
                self.command_buffer,
                0,
                slice::from_ref(&self.vertex_buffer),
                &[0],
            );
            device.handle.cmd_draw(self.command_buffer, 3, 1, 0, 0);

            device
                .dynamic_rendering
                .cmd_end_rendering(self.command_buffer);

            device.handle.end_command_buffer(self.command_buffer)?;
            device
                .handle
                .reset_fences(slice::from_ref(&self.queue_submit_fence))?;

            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(slice::from_ref(&self.acquire_semaphore))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .signal_semaphores(slice::from_ref(&self.release_semaphore))
                .command_buffers(slice::from_ref(&self.command_buffer));
            device.handle.queue_submit(
                device.queue.handle,
                slice::from_ref(&submit_info),
                self.queue_submit_fence,
            )?;

            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(slice::from_ref(&self.release_semaphore))
                .swapchains(slice::from_ref(&self.swapchain.handle))
                .image_indices(slice::from_ref(&index))
                .build();
            self.swapchain
                .functions
                .queue_present(self.device.queue.handle, &present_info)?;
            device.handle.queue_wait_idle(device.queue.handle)?;
        }

        Ok(())
    }
}

fn create_graphics_pipeline(device: &Device, extent: vk::Extent2D) -> anyhow::Result<vk::Pipeline> {
    let pipeline_layout = unsafe {
        device
            .handle
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?
    };
    let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(std::mem::size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()];
    let vertex_attribute_descriptions = [
        vk::VertexInputAttributeDescription::builder()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, position) as u32)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32)
            .build(),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&vertex_attribute_descriptions)
        .vertex_binding_descriptions(&vertex_binding_descriptions);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(slice::from_ref(&blend_attachment));
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let viewport = vk::Viewport::builder()
        .width(extent.width as f32)
        .height(extent.height as f32);
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1)
        .viewports(slice::from_ref(&viewport));

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let mut compiler = shaderc::Compiler::new().unwrap();

    let vertex_shader_module = {
        let code = compile_shader(
            &mut compiler,
            "shaders/triangle.vert",
            "triangle.vert",
            shaderc::ShaderKind::Vertex,
            "main",
        )?;
        unsafe {
            device.handle.create_shader_module(
                &vk::ShaderModuleCreateInfo::builder()
                    .code(&code)
                    .flags(vk::ShaderModuleCreateFlags::empty()),
                None,
            )?
        }
    };
    let fragment_shader_module = {
        let code = compile_shader(
            &mut compiler,
            "shaders/triangle.frag",
            "triangle.frag",
            shaderc::ShaderKind::Fragment,
            "main",
        )?;
        unsafe {
            device.handle.create_shader_module(
                &vk::ShaderModuleCreateInfo::builder()
                    .code(&code)
                    .flags(vk::ShaderModuleCreateFlags::empty()),
                None,
            )?
        }
    };
    let name = CString::new("main")?;
    let vertex_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(&name)
        .build();
    let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(&name)
        .build();
    let shader_modules = [vertex_shader_module, fragment_shader_module];
    let shader_stages = [vertex_shader_stage, fragment_shader_stage];

    let pipeline_cache = unsafe {
        device
            .handle
            .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?
    };

    let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
        .color_attachment_formats(slice::from_ref(&vk::Format::B8G8R8A8_UNORM));
    let create_info = vk::GraphicsPipelineCreateInfo::builder()
        .push_next(&mut rendering_info)
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .rasterization_state(&rasterization)
        .color_blend_state(&blend)
        .multisample_state(&multisample)
        .viewport_state(&viewport_state)
        .depth_stencil_state(&depth_stencil)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .build();
    let pipelines = unsafe {
        device
            .handle
            .create_graphics_pipelines(pipeline_cache, slice::from_ref(&create_info), None)
            .expect("Failed to create graphics pipelines")
    };
    for shader_module in shader_modules {
        unsafe {
            device.handle.destroy_shader_module(shader_module, None);
        }
    }
    Ok(pipelines[0])
}

fn create_image_views(device: &Arc<Device>, swapchain: &Swapchain) -> Vec<vk::ImageView> {
    swapchain
        .images
        .iter()
        .map(|&image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain.surface_format.format)
                .components(
                    vk::ComponentMapping::builder()
                        .r(vk::ComponentSwizzle::R)
                        .g(vk::ComponentSwizzle::G)
                        .b(vk::ComponentSwizzle::B)
                        .a(vk::ComponentSwizzle::A)
                        .build(),
                )
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .build();
            unsafe {
                device
                    .handle
                    .create_image_view(&create_info, None)
                    .expect("Failed to create an image view")
            }
        })
        .collect()
}

fn find_memory_type_index(
    memory_requirements: &vk::MemoryRequirements,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_properties.memory_types[..memory_properties.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_requirements.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _)| index as u32)
}

fn compile_shader(
    compiler: &mut shaderc::Compiler,
    path: &str,
    file_name: &str,
    kind: shaderc::ShaderKind,
    entry_point: &str,
) -> anyhow::Result<Vec<u32>> {
    let source = std::fs::read_to_string(path)?;
    let result = compiler.compile_into_spirv(&source, kind, file_name, entry_point, None)?;
    Ok(result.as_binary().to_owned())
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("vulkan")
        .build(&event_loop)?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut renderer = Renderer::new(&window)?;

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        }
        | Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                },
            ..
        } => {
            elwt.exit();
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } if size.width > 0 && size.height > 0 => {
            _ = renderer.resize();
        }
        Event::AboutToWait => {
            window.request_redraw();
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            _ = renderer.draw();
        }
        _ => (),
    })?;

    Ok(())
}
