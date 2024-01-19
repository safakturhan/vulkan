use std::sync::Arc;

use ash::{extensions::khr, vk};

use crate::{device::Device, surface::Surface};

#[derive(Clone)]
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub functions: khr::Swapchain,
    pub surface_format: vk::SurfaceFormatKHR,
    pub images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(device: &Arc<Device>, surface: &Arc<Surface>) -> anyhow::Result<Self> {
        let functions = khr::Swapchain::new(&device.instance.handle, &device.handle);

        let surface_capabilities = unsafe {
            surface.functions.get_physical_device_surface_capabilities(
                device.physical_device.handle,
                surface.handle,
            )?
        };

        let min_image_count =
            (surface_capabilities.min_image_count + 1).min(surface_capabilities.max_image_count);

        let present_mode = {
            let present_modes = unsafe {
                surface
                    .functions
                    .get_physical_device_surface_present_modes(
                        device.physical_device.handle,
                        surface.handle,
                    )?
            };
            present_modes
                .into_iter()
                .find(|present_mode| present_mode.eq(&vk::PresentModeKHR::MAILBOX))
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        let surface_format = {
            let surface_formats = unsafe {
                surface.functions.get_physical_device_surface_formats(
                    device.physical_device.handle,
                    surface.handle,
                )?
            };
            surface_formats
                .into_iter()
                .find(|surface_format| {
                    surface_format.format.eq(&vk::Format::B8G8R8A8_UNORM)
                        && surface_format
                            .color_space
                            .eq(&vk::ColorSpaceKHR::SRGB_NONLINEAR)
                })
                .expect("Failed to find specified surface format")
        };

        let extent = surface_capabilities.current_extent;

        let handle = unsafe {
            functions.create_swapchain(
                &vk::SwapchainCreateInfoKHR::builder()
                    .surface(surface.handle)
                    .min_image_count(min_image_count)
                    .image_format(surface_format.format)
                    .image_color_space(surface_format.color_space)
                    .image_extent(extent)
                    .image_array_layers(surface_capabilities.max_image_array_layers)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(surface_capabilities.current_transform)
                    .queue_family_indices(&[0])
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(present_mode)
                    .clipped(false)
                    .old_swapchain(vk::SwapchainKHR::null()),
                None,
            )?
        };

        let images = unsafe { functions.get_swapchain_images(handle)? };

        Ok(Self {
            handle,
            functions,
            surface_format,
            images,
            extent,
        })
    }
}
