use std::{ffi::c_void, sync::Arc};

use ash::{extensions::khr, vk, RawPtr};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use crate::instance::Instance;

pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub functions: khr::Surface,
}

impl Surface {
    pub fn new(window: &impl HasWindowHandle, instance: &Arc<Instance>) -> anyhow::Result<Self> {
        let handle = match window.window_handle().unwrap().as_raw() {
            RawWindowHandle::Win32(window_handle) => {
                let hinstance = window_handle.hinstance.as_ref().as_raw_ptr() as *const c_void;
                let hwnd = window_handle.hwnd.get() as *const isize as *const c_void;
                unsafe {
                    khr::Win32Surface::new(&instance.entry, &instance.handle).create_win32_surface(
                        &vk::Win32SurfaceCreateInfoKHR::builder()
                            .hinstance(hinstance)
                            .hwnd(hwnd),
                        None,
                    )?
                }
            }
            _ => unimplemented!(),
        };
        let functions = khr::Surface::new(&instance.entry, &instance.handle);
        Ok(Self { handle, functions })
    }
}
