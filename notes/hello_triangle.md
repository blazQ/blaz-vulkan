# Drawing a Triangle — In-Depth Notes

## Setup

- Simple project structure with the application wrapped in a **class**
- Generic methods for **initialization**, **main loop**, and **resource cleanup**
- Every new Vulkan object becomes a private class member in subsequent chapters
- Objects are created using **RAII**: fill in a struct describing the object, then instantiate it via a `vk::raii::Xxxx` wrapper — handles creation, allocation, and cleanup automatically
- Vulkan can render offscreen, but needs a **surface** to display to the screen — created via a platform-aware library like **GLFW**

## Instance

- The **instance** is the connection between the application and the Vulkan library
- From Vulkan 1.3 onwards, it's managed using RAII
- Required info: application name, Vulkan version, API version, flags, **layers**, and **extensions**
  - Window system extensions (e.g. `VK_KHR_surface`) are required to present to a surface — GLFW provides the required list via `glfwGetRequiredInstanceExtensions`

## Validation Layers

- Optional but essential during development — they hook into Vulkan function calls to add extra behavior
- Vulkan's design philosophy prioritizes minimal driver overhead and explicit intent, so **the core API does almost no error checking by itself**
- What validation layers can do:
  - Check parameters against the spec to detect misuse
  - Track **object creation and destruction** (catch leaks)
  - Check **thread safety**
  - Log calls and profile performance
- Layers are stackable, togglable (e.g. disabled via `NDEBUG` in release builds), and must be installed on the system (bundled with the **Vulkan SDK**)
- Requested layers are validated against those supported by the **instance context**
- A **debug messenger callback** lets you filter and handle layer messages (errors, warnings, info) however you like

## Physical Device

- A **physical device** (`vk::raii::PhysicalDevice`) represents a GPU available on the system
- `pickPhysicalDevice()` enumerates all devices and selects the first suitable one via `isDeviceSuitable()`
- Suitability is checked against four criteria:
  - **API version** ≥ Vulkan 1.3 — queried via `physicalDevice.getProperties().apiVersion`
  - **Graphics queue** support — at least one queue family with `vk::QueueFlagBits::eGraphics`, found via `getQueueFamilyProperties()`
  - **Required extensions** — `VK_KHR_swapchain` (`vk::KHRSwapchainExtensionName`), checked via `enumerateDeviceExtensionProperties()`
  - **Required features** — `dynamicRendering` (Vulkan 1.3) and `extendedDynamicState` (EXT), checked via a `vk::StructureChain` of feature structs
- In general, we can define suitability however we need via code

## Logical Device

- A **logical device** (`vk::raii::Device`) is the interface through which you actually interact with the physical device — multiple logical devices can be created from the same physical device
- Created in `createLogicalDevice()`, which also finds a **queue family** that supports both graphics (`vk::QueueFlagBits::eGraphics`) and **surface presentation** (checked via `getSurfaceSupportKHR()`) — a unified queue simplifies synchronization
- The queue family index is stored as `graphicsIndex` and reused later for command pool creation
- Device features are enabled via a `vk::StructureChain` passed through `pNext`:
  - `vk::PhysicalDeviceVulkan11Features` — `shaderDrawParameters`
  - `vk::PhysicalDeviceVulkan13Features` — `dynamicRendering`
  - `vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT` — `extendedDynamicState`
- Structure chaining avoids having to manually link each structure to the next one, so that the logical device has the pointer to the whole list once we pass it.
- `vk::DeviceQueueCreateInfo` specifies one queue from the chosen family at priority `0.5f`
- `vk::DeviceCreateInfo` ties it together: the feature chain, queue info, and required extensions (`VK_KHR_swapchain`)
- The resulting `vk::raii::Queue` is retrieved immediately via `vk::raii::Queue(device, queueIndex, 0)` and stored as `queue`
