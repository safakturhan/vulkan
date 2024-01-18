use std::sync::Arc;

use ash::{extensions::khr, vk};

use crate::{
    instance::Instance,
    physical_device::{PhysicalDevice, QueueFamily},
    surface::Surface,
};

#[derive(Debug)]
pub struct Queue {
    pub handle: vk::Queue,
    pub queue_family: QueueFamily,
}

pub struct Device {
    pub handle: ash::Device,
    pub instance: Arc<Instance>,
    pub physical_device: PhysicalDevice,
    pub queue: Queue,
    pub dynamic_rendering: khr::DynamicRendering,
}

impl Device {
    pub fn new(
        instance: Arc<Instance>,
        physical_device: PhysicalDevice,
        surface: &Arc<Surface>,
    ) -> anyhow::Result<Self> {
        let queue_family = physical_device
            .queue_families
            .iter()
            .copied()
            .find(|queue_family| {
                let surface_support = unsafe {
                    surface
                        .functions
                        .get_physical_device_surface_support(
                            physical_device.handle,
                            queue_family.index,
                            surface.handle,
                        )
                        .unwrap()
                };
                queue_family
                    .properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    && surface_support
            })
            .unwrap();
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family.index)
            .queue_priorities(&[1.0]);
        let enabled_extension_names = [
            khr::Swapchain::name().as_ptr(),
            khr::DynamicRendering::name().as_ptr(),
        ];
        let handle = {
            let mut dynamic_rendering_features =
                vk::PhysicalDeviceDynamicRenderingFeatures::builder().dynamic_rendering(true);
            unsafe {
                instance.handle.create_device(
                    physical_device.handle,
                    &vk::DeviceCreateInfo::builder()
                        .push_next(&mut dynamic_rendering_features)
                        .queue_create_infos(std::slice::from_ref(&queue_create_info))
                        .enabled_extension_names(&enabled_extension_names),
                    None,
                )?
            }
        };
        let queue = Queue {
            handle: unsafe { handle.get_device_queue(queue_family.index, 0) },
            queue_family,
        };
        let dynamic_rendering = khr::DynamicRendering::new(&instance.handle, &handle);
        Ok(Self {
            handle,
            instance,
            physical_device,
            queue,
            dynamic_rendering,
        })
    }
}
