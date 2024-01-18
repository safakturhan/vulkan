use ash::vk;

#[derive(Debug, Clone, Copy)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

#[derive(Debug, Clone)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub name: String,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_families: Vec<QueueFamily>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}
