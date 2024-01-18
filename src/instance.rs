use std::ffi::{CStr, CString};

use ash::{extensions::khr, vk};

use crate::physical_device::{PhysicalDevice, QueueFamily};

pub struct Instance {
    pub entry: ash::Entry,
    pub handle: ash::Instance,
}

impl Instance {
    pub fn new() -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

        let handle = {
            let application_name = CString::new("vulkan")?;
            let application_version = vk::make_api_version(0, 1, 0, 0);
            let engine_name = CString::new("vulkan")?;
            let engine_version = vk::make_api_version(0, 1, 0, 0);

            let validation_layer = CString::new("VK_LAYER_KHRONOS_validation")?;
            let enabled_layer_names = [validation_layer.as_ptr()];
            let enabled_extension_names = [
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(),
            ];

            unsafe {
                entry.create_instance(
                    &vk::InstanceCreateInfo::builder()
                        .application_info(
                            &vk::ApplicationInfo::builder()
                                .application_name(&application_name)
                                .application_version(application_version)
                                .engine_name(&engine_name)
                                .engine_version(engine_version)
                                .api_version(vk::API_VERSION_1_3),
                        )
                        .enabled_layer_names(&enabled_layer_names)
                        .enabled_extension_names(&enabled_extension_names),
                    None,
                )?
            }
        };

        Ok(Self { entry, handle })
    }

    pub fn physical_devices(&self) -> anyhow::Result<Vec<PhysicalDevice>> {
        let physical_devices = unsafe { self.handle.enumerate_physical_devices()? };
        let physical_devices = physical_devices
            .iter()
            .map(|&physical_device| {
                let properties =
                    unsafe { self.handle.get_physical_device_properties(physical_device) };
                let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                let queue_families: Vec<QueueFamily> = {
                    let queue_families = unsafe {
                        self.handle
                            .get_physical_device_queue_family_properties(physical_device)
                    };
                    queue_families
                        .iter()
                        .enumerate()
                        .map(|(index, &properties)| QueueFamily {
                            index: index as u32,
                            properties,
                        })
                        .collect()
                };
                let memory_properties = unsafe {
                    self.handle
                        .get_physical_device_memory_properties(physical_device)
                };
                PhysicalDevice {
                    handle: physical_device,
                    name,
                    properties,
                    queue_families,
                    memory_properties,
                }
            })
            .collect::<Vec<PhysicalDevice>>();
        Ok(physical_devices)
    }
}
