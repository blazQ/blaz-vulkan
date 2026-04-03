#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

/* This block is necessary for picking the right import path.
Importing Vulkan using an #include preprocessor directive is the classic C
approach that requires compiling Vulkan from scratch, but always works.
Importing with the C++ import module is the modern approach introduced in
C++ 20. It uses a precompiled unit, but requires a modern build system. It
results in faster compile times.
*/
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

/* This block is necessary because Vulkan itself is not aware of platform
specific components, like the OS Windowing System. Vulkan knows about abstract
concepts, like a VkSurface, but has no idea how to create one in Windows, MacOS
etc. and each platform has its own native surface extension. GLFW solves this,
but requires specifying with this macro to include vulkan.h into the library.
This makes the library aware that it needs to return the correct VkSurface. */
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.

/* GLFW is an Open Source, multi-platform library for OpenGL, OpenGL ES and
Vulkan development on the desktop. It provides a simple API for creating
windows, contexts and surfaces, receiving input and events. */
#include <GLFW/glfw3.h>

/* These are constants that specify the surface dimensions.*/
constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

/* This is used to make the drawing process more efficient by avoiding the CPU
being idle while the GPU renders frame N. By having multiple frames in flight,
while the CPU records commands for frame N+1, the GPU renders frame N.*/
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

/* Validation layers are explained in the notes/hello_triangle.md section. */
const std::vector<char const *> validationLayers = {
	"VK_LAYER_KHRONOS_validation"};

/* Uses a standard C macro automatically defined by the build system to disable
 * validation layers when compiling in Release mode. */
#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

// Per-vertex data sent to the vertex shader. The GPU reads this from the vertex
// buffer and it describes the layout of each vertex in memory.
struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
	}

	static std::array<vk::VertexInputAttributeDescription, 3>
	getAttributeDescriptions()
	{
		return {vk::VertexInputAttributeDescription(
					0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
				vk::VertexInputAttributeDescription(
					1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
				vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat,
													offsetof(Vertex, texCoord))};
	}
};

class HelloTriangleApplication
{
public:
	// Entry point: initializes everything, runs the loop, then tears down.
	void run()
	{
		initWindow();
		initVulkan();
		imGuiInit();
		mainLoop();
		cleanup();
	}

private:
	// =========================================================================
	// MEMBER VARIABLES — grouped by concern
	// =========================================================================

	// --- Window ---
	GLFWwindow *window = nullptr;
	bool framebufferResized = false;

	// --- Vulkan core objects ---
	// Context -> Instance -> DebugMessenger -> Surface -> PhysicalDevice ->
	// Device -> Queue. Each one depends on the previous.
	vk::raii::Context context;
	vk::raii::Instance instance = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR surface = nullptr;
	vk::raii::PhysicalDevice physicalDevice = nullptr;
	vk::raii::Device device = nullptr;
	vk::raii::Queue queue = nullptr;
	uint32_t graphicsIndex = ~0;

	// --- Swapchain ---
	// The swapchain is a ring buffer of images we render into and present.
	vk::raii::SwapchainKHR swapChain = nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::SurfaceFormatKHR swapChainSurfaceFormat;
	vk::Extent2D swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;

	// --- Graphics pipeline ---
	// The pipeline encodes the full state of the GPU rendering stages: shaders,
	// rasterization, blending, depth test, etc.
	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline graphicsPipeline = nullptr;

	vk::PresentModeKHR presentMode;
	vk::PresentModeKHR pendingPresentMode = vk::PresentModeKHR::eMailbox;

	// --- Command recording ---
	// One pool owns the memory; one command buffer per frame in flight.
	vk::raii::CommandPool commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;
	uint32_t frameIndex = 0;

	// --- Render attachments (recreated alongside the swapchain on resize) ---
	// MSAA: we render into a multisampled color image and resolve into the
	// swapchain image. The depth image tests fragment depth for occlusion.
	vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
	vk::SampleCountFlagBits maxMsaaSamples = vk::SampleCountFlagBits::e1;
	vk::SampleCountFlagBits pendingMsaaSamples = vk::SampleCountFlagBits::e1;
	vk::SampleCountFlags supportedMsaaSamples;
	vk::raii::Image colorImage = nullptr;
	vk::raii::DeviceMemory colorImageMemory = nullptr;
	vk::raii::ImageView colorImageView = nullptr;
	vk::raii::Image depthImage = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView depthImageView = nullptr;

	// --- Texture ---
	// A 2D image on the GPU plus a sampler that describes how to read it in
	// shaders (filtering, wrapping, etc.) and a view into the image.
	uint32_t mipLevels;
	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView textureImageView = nullptr;
	vk::raii::Sampler textureSampler = nullptr;

	// --- Scene geometry ---
	// CPU-side data, uploaded once to GPU-local buffers.
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	vk::raii::Buffer vertexBuffer = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer indexBuffer = nullptr;
	vk::raii::DeviceMemory indexBufferMemory = nullptr;

	// --- Uniform buffers (one per frame in flight) ---
	// Written by the CPU each frame to pass transform matrices to the shader.
	// Persistently mapped (no map/unmap per frame).
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void *> uniformBuffersMapped;

	// --- Descriptors ---
	// Descriptors connect shader binding slots to actual GPU resources
	// (uniform buffers, textures). The pool allocates the sets;
	// the sets are updated to point to the actual buffers/images.
	vk::raii::DescriptorPool descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	// --- Synchronization primitives ---
	// presentComplete: signals when the swapchain image is available to render
	// into. renderFinished: signals when rendering is done and the image can be
	// presented. inFlightFence:  blocks the CPU until the GPU is done with this
	// frame slot.
	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;

	// --- Configuration ---
	std::vector<const char *> requiredDeviceExtension = {
		vk::KHRSwapchainExtensionName};

	// Data layout of the uniform buffer as the shader sees it.
	struct UniformBufferObject
	{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};

	vk::raii::DescriptorPool imguiDescriptorPool = nullptr;

	// =========================================================================
	// SECTION 1: WINDOW
	// Creates the OS window and registers a resize callback so we know when to
	// recreate the swapchain.
	// =========================================================================

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		// glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); /* TODO: Make it optional*/

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	// GLFW calls this when the window is resized. We can't recreate the
	// swapchain here directly (we're outside the render loop), so we just set a
	// flag that drawFrame checks at the end of each frame.
	static void framebufferResizeCallback(GLFWwindow *window, int width,
										  int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication *>(
			glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	// =========================================================================
	// SECTION 2: VULKAN BOOTSTRAP (Instance → Debug → Surface)
	// The Vulkan instance is the connection between the application and the
	// Vulkan library. The debug messenger routes validation messages to our
	// callback. The surface is the abstract handle to the OS window's
	// presentation target.
	// =========================================================================

	// initVulkan is the master initialization sequence. Reading it top-to-bottom
	// gives the full picture of what gets created and in what order.
	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createColorResources();
		createDepthResources();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		loadModel();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	// Builds the VkInstance: declares which Vulkan version, layers (validation),
	// and instance-level extensions (e.g. surface, debug utils) we need.
	void createInstance()
	{
		constexpr vk::ApplicationInfo appInfo{
			.pApplicationName = "Hello Triangle",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "No Engine",
			.engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = vk::ApiVersion14};

		// Get the required layers
		std::vector<char const *> requiredLayers;
		if (enableValidationLayers)
		{
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		// Check if the required layers are supported by the Vulkan implementation.
		auto layerProperties = context.enumerateInstanceLayerProperties();
		auto unsupportedLayerIt = std::ranges::find_if(
			requiredLayers, [&layerProperties](auto const &requiredLayer)
			{ return std::ranges::none_of(
				  layerProperties, [requiredLayer](auto const &layerProperty)
				  { return strcmp(layerProperty.layerName, requiredLayer) == 0; }); });
		if (unsupportedLayerIt != requiredLayers.end())
		{
			throw std::runtime_error("Required layer not supported: " +
									 std::string(*unsupportedLayerIt));
		}

		// Get the required extensions.
		auto requiredExtensions = getRequiredInstanceExtensions();

		// Check if the required extensions are supported by the Vulkan
		// implementation.
		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		auto unsupportedPropertyIt = std::ranges::find_if(
			requiredExtensions,
			[&extensionProperties](auto const &requiredExtension)
			{
				return std::ranges::none_of(
					extensionProperties,
					[requiredExtension](auto const &extensionProperty)
					{
						return strcmp(extensionProperty.extensionName,
									  requiredExtension) == 0;
					});
			});
		if (unsupportedPropertyIt != requiredExtensions.end())
		{
			throw std::runtime_error("Required extension not supported: " +
									 std::string(*unsupportedPropertyIt));
		}

		vk::InstanceCreateInfo createInfo{
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
			.ppEnabledLayerNames = requiredLayers.data(),
			.enabledExtensionCount =
				static_cast<uint32_t>(requiredExtensions.size()),
			.ppEnabledExtensionNames = requiredExtensions.data()};
		instance = vk::raii::Instance(context, createInfo);
	}

	// Returns the list of instance extensions we need. GLFW tells us which
	// platform surface extension it needs; we add the debug utils on top when
	// validation is enabled.
	std::vector<const char *> getRequiredInstanceExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		auto glfwExtensions =
			glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
		}

		return extensions;
	}

	// Vulkan calls this whenever the validation layer has a message to report.
	// We only print warnings and errors to avoid spam.
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
		vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
		vk::DebugUtilsMessageTypeFlagsEXT type,
		const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *)
	{
		if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
			severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
		{
			std::cerr << "validation layer: type " << to_string(type)
					  << " msg: " << pCallbackData->pMessage << std::endl;
		}

		return vk::False;
	}

	// Registers our debugCallback with Vulkan so validation messages are routed
	// to it. No-op in Release builds.
	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
		vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
			vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
			vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
		vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
			.messageSeverity = severityFlags,
			.messageType = messageTypeFlags,
			.pfnUserCallback = &debugCallback};
		debugMessenger =
			instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
	}

	// Creates the Vulkan surface by asking GLFW to bridge to the platform's
	// native windowing API (Win32, Xlib, Wayland, etc.).
	void createSurface()
	{
		VkSurfaceKHR _surface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	// =========================================================================
	// SECTION 3: PHYSICAL & LOGICAL DEVICE
	// The physical device is the GPU; we pick one that meets our requirements.
	// The logical device is our handle to the chosen GPU: we also retrieve the
	// queue we'll use to submit rendering commands.
	// =========================================================================

	// Iterates all available GPUs and picks the first one that passes
	// isDeviceSuitable(). Also determines the highest supported MSAA sample
	// count at this point.
	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> physicalDevices =
			instance.enumeratePhysicalDevices();
		for (const auto &device : physicalDevices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				msaaSamples = getMaxUsableSampleCount();
				maxMsaaSamples = msaaSamples;
				pendingMsaaSamples = msaaSamples;
				break;
			}
		}
		if (physicalDevice == nullptr)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	// Checks that the candidate GPU supports: Vulkan 1.3, a graphics queue,
	// the required device extensions (swapchain), and the features we use
	// (dynamic rendering, extended dynamic state).
	bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice)
	{
		// Check if the device is a discrete GPU.
		bool isDiscreteGPU = physicalDevice.getProperties().deviceType ==
							 vk::PhysicalDeviceType::eDiscreteGpu;

		// Check if the physicalDevice supports the Vulkan 1.3 API version
		bool supportsVulkan1_3 =
			physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

		// Check if any of the queue families support graphics operations
		auto queueFamilies = physicalDevice.getQueueFamilyProperties();
		bool supportsGraphics =
			std::ranges::any_of(queueFamilies, [](auto const &qfp)
								{ return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

		// Check if all required physicalDevice extensions are available
		auto availableDeviceExtensions =
			physicalDevice.enumerateDeviceExtensionProperties();
		bool supportsAllRequiredExtensions = std::ranges::all_of(
			requiredDeviceExtension,
			[&availableDeviceExtensions](auto const &requiredDeviceExtension)
			{
				return std::ranges::any_of(
					availableDeviceExtensions,
					[requiredDeviceExtension](auto const &availableDeviceExtension)
					{
						return strcmp(availableDeviceExtension.extensionName,
									  requiredDeviceExtension) == 0;
					});
			});

		// Check if the physicalDevice supports the required features
		auto features = physicalDevice.template getFeatures2<
			vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
			vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
		bool supportsRequiredFeatures =
			features.template get<vk::PhysicalDeviceVulkan13Features>()
				.dynamicRendering &&
			features
				.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
				.extendedDynamicState;

		// Return true if the physicalDevice meets all the criteria
		return supportsVulkan1_3 && supportsGraphics &&
			   supportsAllRequiredExtensions && supportsRequiredFeatures &&
			   isDiscreteGPU;
	}

	// Queries the GPU for the highest sample count supported by both the color
	// and depth framebuffers. Used to set msaaSamples.
	vk::SampleCountFlagBits getMaxUsableSampleCount()
	{
		vk::PhysicalDeviceProperties physicalDeviceProperties =
			physicalDevice.getProperties();

		vk::SampleCountFlags counts =
			physicalDeviceProperties.limits.framebufferColorSampleCounts &
			physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		supportedMsaaSamples = counts;
		if (counts & vk::SampleCountFlagBits::e64)
		{
			return vk::SampleCountFlagBits::e64;
		}
		if (counts & vk::SampleCountFlagBits::e32)
		{
			return vk::SampleCountFlagBits::e32;
		}
		if (counts & vk::SampleCountFlagBits::e16)
		{
			return vk::SampleCountFlagBits::e16;
		}
		if (counts & vk::SampleCountFlagBits::e8)
		{
			return vk::SampleCountFlagBits::e8;
		}
		if (counts & vk::SampleCountFlagBits::e4)
		{
			return vk::SampleCountFlagBits::e4;
		}
		if (counts & vk::SampleCountFlagBits::e2)
		{
			return vk::SampleCountFlagBits::e2;
		}

		return vk::SampleCountFlagBits::e1;
	}

	// Creates the logical device (our interface to the GPU) by requesting the
	// queue family that supports both graphics and present, and enabling the
	// specific Vulkan features we depend on.
	void createLogicalDevice()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
			physicalDevice.getQueueFamilyProperties();

		// get the first index into queueFamilyProperties which supports both
		// graphics and present
		uint32_t queueIndex = ~0;
		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size();
			 qfpIndex++)
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags &
				 vk::QueueFlagBits::eGraphics) &&
				physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
			{
				// found a queue family that supports both graphics and present
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0)
		{
			throw std::runtime_error(
				"Could not find a queue for graphics and present -> terminating");
		}

		graphicsIndex = queueIndex;

		// query for Vulkan 1.3 features
		vk::StructureChain<vk::PhysicalDeviceFeatures2,
						   vk::PhysicalDeviceVulkan11Features,
						   vk::PhysicalDeviceVulkan13Features,
						   vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
			featureChain = {
				{.features = {.samplerAnisotropy =
								  true}}, // vk::PhysicalDeviceFeatures2
				{.shaderDrawParameters =
					 true}, // vk::PhysicalDeviceVulkan11Features
				{.synchronization2 = true,
				 .dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
				{.extendedDynamicState =
					 true} // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
			};

		// create a Device
		float queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
			.queueFamilyIndex = queueIndex,
			.queueCount = 1,
			.pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo deviceCreateInfo{
			.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &deviceQueueCreateInfo,
			.enabledExtensionCount =
				static_cast<uint32_t>(requiredDeviceExtension.size()),
			.ppEnabledExtensionNames = requiredDeviceExtension.data()};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue = vk::raii::Queue(device, queueIndex, 0);
	}

	// =========================================================================
	// SECTION 4: SWAPCHAIN
	// The swapchain manages a pool of images we render into. After rendering,
	// we "present" the image so the display shows it. We also create an
	// ImageView for each swapchain image (a view into the image that tells
	// Vulkan how to interpret it as a 2D color target).
	// =========================================================================

	// Queries the surface capabilities and formats, picks the best options, and
	// creates the swapchain. On resize this is recreated via recreateSwapChain().
	void createSwapChain()
	{
		vk::SurfaceCapabilitiesKHR surfaceCapabilities =
			physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapChainExtent = chooseSwapExtent(surfaceCapabilities);
		uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

		std::vector<vk::SurfaceFormatKHR> availableFormats =
			physicalDevice.getSurfaceFormatsKHR(*surface);
		swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

		std::vector<vk::PresentModeKHR> availablePresentModes =
			physicalDevice.getSurfacePresentModesKHR(*surface);
		if (!std::ranges::any_of(availablePresentModes, [this](auto m)
								 { return m == pendingPresentMode; }))
			pendingPresentMode = chooseSwapPresentMode(availablePresentModes);
		presentMode = pendingPresentMode;

		vk::SwapchainCreateInfoKHR swapChainCreateInfo{
			.surface = *surface,
			.minImageCount = minImageCount,
			.imageFormat = swapChainSurfaceFormat.format,
			.imageColorSpace = swapChainSurfaceFormat.colorSpace,
			.imageExtent = swapChainExtent,
			.imageArrayLayers = 1,
			.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
			.imageSharingMode = vk::SharingMode::eExclusive,
			.preTransform = surfaceCapabilities.currentTransform,
			.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
			.presentMode = presentMode,
			.clipped = true};

		swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	// Aims for 3 images (triple buffering). Falls back to the surface's maximum
	// if the hardware doesn't support that many.
	static uint32_t chooseSwapMinImageCount(
		vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) &&
			(surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	// Prefers B8G8R8A8_SRGB with sRGB color space. Falls back to the first
	// available format if not found.
	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
		std::vector<vk::SurfaceFormatKHR> const &availableFormats)
	{
		assert(!availableFormats.empty());
		const auto formatIt =
			std::ranges::find_if(availableFormats, [](const auto &format)
								 { return format.format == vk::Format::eB8G8R8A8Srgb &&
										  format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	// Prefers Mailbox (renders as fast as possible, replaces queued frames
	// without tearing). Falls back to FIFO (VSync), which is always guaranteed.
	static vk::PresentModeKHR chooseSwapPresentMode(
		std::vector<vk::PresentModeKHR> const &availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode)
								   { return presentMode == vk::PresentModeKHR::eFifo; }));
		return std::ranges::any_of(availablePresentModes,
								   [](const vk::PresentModeKHR value)
								   {
									   return vk::PresentModeKHR::eMailbox == value;
								   })
				   ? vk::PresentModeKHR::eMailbox
				   : vk::PresentModeKHR::eFifo;
	}

	// Returns the swapchain extent matching the window's framebuffer size in
	// pixels. Uses the surface's currentExtent when available; otherwise clamps
	// the GLFW framebuffer size to the surface's min/max limits.
	vk::Extent2D
	chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities)
	{
		if (capabilities.currentExtent.width !=
			std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
									 capabilities.maxImageExtent.width),
				std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
									 capabilities.maxImageExtent.height)};
	}

	// Creates one ImageView per swapchain image. An ImageView tells Vulkan how
	// to interpret the raw image memory (format, aspect, mip range, etc.).
	void createImageViews()
	{
		swapChainImageViews.reserve(swapChainImages.size());

		for (uint32_t i = 0; i < swapChainImages.size(); i++)
		{
			swapChainImageViews.push_back(
				createImageView(swapChainImages[i], swapChainSurfaceFormat.format,
								vk::ImageAspectFlagBits::eColor, 1));
		}
	}

	// Generic ImageView factory used for swapchain images, the texture, the
	// depth image, and the MSAA color image.
	vk::raii::ImageView createImageView(vk::Image image, vk::Format format,
										vk::ImageAspectFlags aspectFlags,
										uint32_t mipLevels)
	{
		vk::ImageViewCreateInfo viewInfo{
			.image = image,
			.viewType = vk::ImageViewType::e2D,
			.format = format,
			.subresourceRange = {aspectFlags, 0, mipLevels, 0, 1}};
		return vk::raii::ImageView(device, viewInfo);
	}

	// =========================================================================
	// SECTION 5: GRAPHICS PIPELINE
	// The pipeline is an immutable object that describes the entire GPU rendering
	// state: which shaders to run, how to rasterize, blend, test depth, etc.
	// We also set up the descriptor set layout here, which describes the
	// interface between the CPU-side buffers/textures and shader binding slots.
	// =========================================================================

	// Declares the layout of our descriptor sets: binding 0 is a uniform buffer
	// (used in the vertex shader for transforms), binding 1 is a combined
	// image+sampler (used in the fragment shader for texturing).
	void createDescriptorSetLayout()
	{
		std::array bindings = {vk::DescriptorSetLayoutBinding(
								   0, vk::DescriptorType::eUniformBuffer, 1,
								   vk::ShaderStageFlagBits::eVertex, nullptr),
							   vk::DescriptorSetLayoutBinding(
								   1, vk::DescriptorType::eCombinedImageSampler, 1,
								   vk::ShaderStageFlagBits::eFragment, nullptr)};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = bindings.size(), .pBindings = bindings.data()};
		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	// Assembles the full graphics pipeline by wiring together:
	//   - Shader stages (vertex + fragment, loaded from a single SPIR-V file)
	//   - Vertex input layout (how to read Vertex structs from the buffer)
	//   - Input assembly (triangle list topology)
	//   - Viewport and scissor (dynamic, set at draw time)
	//   - Rasterizer (fill mode, back-face culling, CCW winding)
	//   - Multisampling (MSAA sample count)
	//   - Depth/stencil test (depth test on, depth write on, less-than compare)
	//   - Color blend (blending off, full RGBA write mask)
	//   - PipelineRenderingCreateInfo (replaces render pass in dynamic rendering)
	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule =
			createShaderModule(readFile("shaders/slang.spv"));
		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = shaderModule,
			.pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = shaderModule,
			.pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
															fragShaderStageInfo};
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount =
				static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data()};

		std::vector dynamicStates = {vk::DynamicState::eViewport,
									 vk::DynamicState::eScissor};

		vk::PipelineDynamicStateCreateInfo dynamicState{
			.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
			.pDynamicStates = dynamicStates.data()};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
			.topology = vk::PrimitiveTopology::eTriangleList};
		vk::Viewport viewport{.x = 0.0f,
							  .y = 0.0f,
							  .width = static_cast<float>(swapChainExtent.width),
							  .height = static_cast<float>(swapChainExtent.height),
							  .minDepth = 0.0f,
							  .maxDepth = 1.0f};
		vk::Rect2D scissor{.offset = vk::Offset2D{0, 0}, .extent = swapChainExtent};
		vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1,
														  .pViewports = &viewport,
														  .scissorCount = 1,
														  .pScissors = &scissor};

		vk::PipelineRasterizationStateCreateInfo rasterizer{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.depthBiasSlopeFactor = 1.0f,
			.lineWidth = 1.0f};
		vk::PipelineMultisampleStateCreateInfo multisampling{
			.rasterizationSamples = msaaSamples, .sampleShadingEnable = vk::False};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable = vk::False,
			.colorWriteMask =
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
		vk::PipelineColorBlendStateCreateInfo colorBlending{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment};
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = 1,
			.pSetLayouts = &*descriptorSetLayout,
			.pushConstantRangeCount = 0};

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
		vk::Format depthFormat = findDepthFormat();
		vk::PipelineDepthStencilStateCreateInfo depthStencil{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False};
		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapChainSurfaceFormat.format,
			.depthAttachmentFormat = depthFormat};
		vk::GraphicsPipelineCreateInfo pipelineInfo{
			.pNext = &pipelineRenderingCreateInfo,
			.stageCount = 2,
			.pStages = shaderStages,
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = pipelineLayout,
			.renderPass = nullptr};
		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
	}

	// Reads a binary file into a byte buffer. Used to load the SPIR-V shader.
	static std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		file.close();

		return buffer;
	}

	// Wraps the SPIR-V bytecode in a VkShaderModule, which is the handle Vulkan
	// uses when building the pipeline.
	[[nodiscard]] vk::raii::ShaderModule
	createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{
			.codeSize = code.size() * sizeof(char),
			.pCode = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule shaderModule{device, createInfo};
		return shaderModule;
	}

	// Iterates a candidate list of depth formats and returns the first one the
	// GPU supports with optimal tiling and depth-stencil usage.
	vk::Format findDepthFormat()
	{
		return findSupportedFormat(
			{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
			 vk::Format::eD24UnormS8Uint},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment);
	}

	// Generic format selection: asks the GPU which of the candidate formats
	// supports the requested tiling and feature flags.
	vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
								   vk::ImageTiling tiling,
								   vk::FormatFeatureFlags features)
	{
		for (const auto format : candidates)
		{
			vk::FormatProperties props = physicalDevice.getFormatProperties(format);
			if (tiling == vk::ImageTiling::eLinear &&
				(props.linearTilingFeatures & features) == features)
			{
				return format;
			}
			if (tiling == vk::ImageTiling::eOptimal &&
				(props.optimalTilingFeatures & features) == features)
			{
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	// Returns true if the given depth format also includes a stencil component.
	bool hasStencilComponent(vk::Format format)
	{
		return format == vk::Format::eD32SfloatS8Uint ||
			   format == vk::Format::eD24UnormS8Uint;
	}

	// =========================================================================
	// SECTION 6: COMMAND INFRASTRUCTURE
	// Commands in Vulkan are not issued directly; they are recorded into command
	// buffers and then submitted to a queue. The command pool is the allocator
	// for these buffers — it is tied to a specific queue family.
	// =========================================================================

	// Creates the command pool for our graphics queue family. The
	// eResetCommandBuffer flag lets us re-record each buffer every frame without
	// needing to free and reallocate.
	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = graphicsIndex};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	// Allocates one primary command buffer per frame-in-flight slot. These are
	// the buffers we record rendering commands into each frame.
	void createCommandBuffers()
	{
		commandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool = commandPool,
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = MAX_FRAMES_IN_FLIGHT};

		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	// =========================================================================
	// SECTION 7: RENDER ATTACHMENTS
	// These images are the rendering targets for each frame. They are sized to
	// match the swapchain extent and must be recreated whenever the swapchain is
	// recreated (e.g. on window resize).
	//
	// MSAA color image: the GPU renders into this at multiple samples/pixel.
	//   After the render pass it is resolved (averaged) into the swapchain image.
	//   It is declared eTransient because its contents never need to leave the
	//   GPU — they are consumed immediately by the resolve.
	//
	// Depth image: stores the depth (Z) value of the closest fragment at each
	//   pixel. The depth test discards fragments that are behind already-drawn
	//   geometry.
	// =========================================================================

	// Creates the MSAA color image and its view. Both use the same format as the
	// swapchain so the resolve step is a straight copy.
	void createColorResources()
	{
		vk::Format colorFormat = swapChainSurfaceFormat.format;

		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
					colorFormat, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransientAttachment |
						vk::ImageUsageFlagBits::eColorAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage,
					colorImageMemory);
		colorImageView = createImageView(colorImage, colorFormat,
										 vk::ImageAspectFlagBits::eColor, 1);
	}

	// Creates the depth image and its view. The format is chosen at runtime
	// based on what the GPU supports (D32, D32S8, or D24S8).
	// The depth image must have the same MSAA sample count as the color image.
	void createDepthResources()
	{
		vk::Format depthFormat = findDepthFormat();
		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
					depthFormat, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eDepthStencilAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
					depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat,
										 vk::ImageAspectFlagBits::eDepth, 1);
	}

	// =========================================================================
	// SECTION 8: GPU RESOURCE UTILITIES
	// Low-level helpers for allocating GPU memory, creating images/buffers, and
	// issuing one-time commands. These are used throughout sections 7-10.
	// =========================================================================

	// Allocates a VkImage and binds it to a new VkDeviceMemory allocation.
	// All image creation goes through here: texture, depth, color MSAA.
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
					 vk::SampleCountFlagBits numSamples, vk::Format format,
					 vk::ImageTiling tiling, vk::ImageUsageFlags usage,
					 vk::MemoryPropertyFlags properties, vk::raii::Image &image,
					 vk::raii::DeviceMemory &imageMemory)
	{
		vk::ImageCreateInfo imageInfo{
			.imageType = vk::ImageType::e2D,
			.format = format,
			.extent = {width, height, 1},
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = numSamples,
			.tiling = tiling,
			.usage = usage,
			.sharingMode = vk::SharingMode::eExclusive,
		};

		image = vk::raii::Image(device, imageInfo);

		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex =
				findMemoryType(memRequirements.memoryTypeBits, properties)};
		imageMemory = vk::raii::DeviceMemory(device, allocInfo);
		image.bindMemory(imageMemory, 0);
	}

	// Allocates a VkBuffer and binds it to a new VkDeviceMemory allocation.
	// All buffer creation goes through here: staging, vertex, index, uniform.
	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
					  vk::MemoryPropertyFlags properties,
					  vk::raii::Buffer &buffer,
					  vk::raii::DeviceMemory &bufferMemory)
	{
		vk::BufferCreateInfo bufferInfo{.size = size,
										.usage = usage,
										.sharingMode = vk::SharingMode::eExclusive};
		buffer = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo memoryAllocateInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex =
				findMemoryType(memRequirements.memoryTypeBits, properties)};
		bufferMemory = vk::raii::DeviceMemory(device, memoryAllocateInfo);
		buffer.bindMemory(*bufferMemory, 0);
	}

	// Searches the GPU's memory heaps for one that satisfies both the type
	// filter (which heaps the resource can use) and the property flags (e.g.
	// device-local for fast GPU access, or host-visible for CPU writes).
	uint32_t findMemoryType(uint32_t typeFilter,
							vk::MemoryPropertyFlags properties)
	{
		vk::PhysicalDeviceMemoryProperties memProperties =
			physicalDevice.getMemoryProperties();
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) &&
				(memProperties.memoryTypes[i].propertyFlags & properties) ==
					properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	// Allocates and begins a one-time-submit command buffer. Used for upload
	// operations (buffer copies, image transitions, mip generation) that happen
	// once at load time and do not need to be re-recorded.
	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool = commandPool,
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = 1};
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer =
			std::make_unique<vk::raii::CommandBuffer>(
				std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

		vk::CommandBufferBeginInfo beginInfo{
			.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
		commandBuffer->begin(beginInfo);

		return commandBuffer;
	}

	// Ends, submits, and waits for a one-time command buffer. After this returns
	// the GPU has finished executing the commands.
	void endSingleTimeCommands(vk::raii::CommandBuffer &commandBuffer)
	{
		commandBuffer.end();

		vk::SubmitInfo submitInfo{.commandBufferCount = 1,
								  .pCommandBuffers = &*commandBuffer};
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();
	}

	// Copies a buffer into another (typically staging → device-local).
	void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
					vk::DeviceSize size)
	{
		auto commandCopyBuffer = beginSingleTimeCommands();
		commandCopyBuffer->copyBuffer(srcBuffer, dstBuffer,
									  vk::BufferCopy(0, 0, size));
		endSingleTimeCommands(*commandCopyBuffer);
	}

	// Copies a staging buffer into a GPU image. The image must already be in
	// TransferDstOptimal layout before this is called.
	void copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image,
						   uint32_t width, uint32_t height)
	{
		auto commandBuffer = beginSingleTimeCommands();
		vk::BufferImageCopy region{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
			.imageOffset = {0, 0, 0},
			.imageExtent = {width, height, 1}};
		commandBuffer->copyBufferToImage(
			buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
		// Submit the buffer copy to the graphics queue
		endSingleTimeCommands(*commandBuffer);
	}

	// One-time image layout transition using synchronization2 barriers.
	// Supports Undefined→TransferDst (before upload) and
	// TransferDst→ShaderReadOnly (after upload, before sampling).
	void transitionImageLayout(const vk::raii::Image &image,
							   vk::ImageLayout oldLayout,
							   vk::ImageLayout newLayout)
	{
		auto commandBuffer = beginSingleTimeCommands();

		vk::AccessFlags2 srcAccessMask;
		vk::AccessFlags2 dstAccessMask;
		vk::PipelineStageFlags2 srcStageMask;
		vk::PipelineStageFlags2 dstStageMask;

		if (oldLayout == vk::ImageLayout::eUndefined &&
			newLayout == vk::ImageLayout::eTransferDstOptimal)
		{
			srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
			dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
			dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
				 newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
			srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
			dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
			dstAccessMask = vk::AccessFlagBits2::eShaderRead;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}

		vk::ImageMemoryBarrier2 barrier{
			.srcStageMask = srcStageMask,
			.srcAccessMask = srcAccessMask,
			.dstStageMask = dstStageMask,
			.dstAccessMask = dstAccessMask,
			.oldLayout = oldLayout,
			.newLayout = newLayout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = *image,
			.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0,
								 1}};

		vk::DependencyInfo dependencyInfo{.imageMemoryBarrierCount = 1,
										  .pImageMemoryBarriers = &barrier};
		commandBuffer->pipelineBarrier2(dependencyInfo);

		endSingleTimeCommands(*commandBuffer);
	}

	// =========================================================================
	// SECTION 9: TEXTURE & MODEL LOADING
	// Loads the texture image from disk, uploads it to a GPU-local image, and
	// generates mip levels for it. Then loads the 3D model geometry.
	// =========================================================================

	// Loads a PNG from disk into a staging buffer, creates the GPU image, copies
	// the data over, then calls generateMipmaps to fill the remaining mip levels.
	void createTextureImage()
	{
		int texWidth, texHeight, texChannels;
		stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight,
									&texChannels, STBI_rgb_alpha);
		vk::DeviceSize imageSize = texWidth * texHeight * 4;

		mipLevels = static_cast<uint32_t>(
						std::floor(std::log2(std::max(texWidth, texHeight)))) +
					1;

		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image!");
		}

		vk::raii::Buffer stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
					 vk::MemoryPropertyFlagBits::eHostVisible |
						 vk::MemoryPropertyFlagBits::eHostCoherent,
					 stagingBuffer, stagingBufferMemory);
		void *data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, pixels, imageSize);
		stagingBufferMemory.unmapMemory();

		stbi_image_free(pixels);
		createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
					vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransferSrc |
						vk::ImageUsageFlagBits::eTransferDst |
						vk::ImageUsageFlagBits::eSampled,
					vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage,
					textureImageMemory);
		transitionImageLayout(textureImage, vk::ImageLayout::eUndefined,
							  vk::ImageLayout::eTransferDstOptimal);
		copyBufferToImage(stagingBuffer, textureImage,
						  static_cast<uint32_t>(texWidth),
						  static_cast<uint32_t>(texHeight));
		generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth,
						texHeight, mipLevels);
	}

	// Generates all mip levels after mip 0 has been uploaded. For each level i:
	//   1. Barrier: transition level i-1 from TransferDst → TransferSrc
	//   2. Blit: downsample level i-1 into level i (half size each dimension)
	//   3. Barrier: transition level i-1 → ShaderReadOnly (done with it)
	// After the loop, the last level is transitioned to ShaderReadOnly separately
	// (it was never used as a blit source).
	void generateMipmaps(vk::raii::Image &image, vk::Format imageFormat,
						 int32_t texWidth, int32_t texHeight,
						 uint32_t mipLevels)
	{
		vk::FormatProperties props =
			physicalDevice.getFormatProperties(imageFormat);
		if (!(props.optimalTilingFeatures &
			  vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
			throw std::runtime_error(
				"texture image format does not support linear blitting!");

		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer =
			beginSingleTimeCommands();

		vk::ImageMemoryBarrier barrier{
			.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
			.dstAccessMask = vk::AccessFlagBits::eTransferRead,
			.oldLayout = vk::ImageLayout::eTransferDstOptimal,
			.newLayout = vk::ImageLayout::eTransferSrcOptimal,
			.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
			.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
			.image = *image};
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;
		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++)
		{
			barrier.subresourceRange.baseMipLevel = i - 1;

			barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
			commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
										   vk::PipelineStageFlagBits::eTransfer, {},
										   {}, {}, barrier);
			vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
			offsets[0] = vk::Offset3D(0, 0, 0);
			offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
			dstOffsets[0] = vk::Offset3D(0, 0, 0);
			dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1,
										 mipHeight > 1 ? mipHeight / 2 : 1, 1);
			vk::ImageBlit blit = {.srcSubresource = {},
								  .srcOffsets = offsets,
								  .dstSubresource = {},
								  .dstOffsets = dstOffsets};
			blit.srcSubresource = vk::ImageSubresourceLayers(
				vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
			blit.dstSubresource =
				vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1);
			commandBuffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
									 image, vk::ImageLayout::eTransferDstOptimal,
									 {blit}, vk::Filter::eLinear);
			barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
										   vk::PipelineStageFlagBits::eFragmentShader,
										   {}, {}, {}, barrier);
			if (mipWidth > 1)
				mipWidth /= 2;
			if (mipHeight > 1)
				mipHeight /= 2;
		}
		// Transition the last mip level (it was only ever a blit destination,
		// never a source, so the loop never transitioned it).
		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
									   vk::PipelineStageFlagBits::eFragmentShader,
									   {}, {}, {}, barrier);

		endSingleTimeCommands(*commandBuffer);
	}

	// Creates the ImageView for the texture (all mip levels, color aspect).
	void createTextureImageView()
	{
		textureImageView =
			createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb,
							vk::ImageAspectFlagBits::eColor, mipLevels);
	}

	// Creates the sampler that describes how the fragment shader reads the
	// texture: linear filtering (smooth), repeat wrapping, anisotropic
	// filtering up to the hardware maximum, and full mip range.
	void createTextureSampler()
	{
		vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
		vk::SamplerCreateInfo samplerInfo{
			.magFilter = vk::Filter::eLinear,
			.minFilter = vk::Filter::eLinear,
			.mipmapMode = vk::SamplerMipmapMode::eLinear,
			.addressModeU = vk::SamplerAddressMode::eRepeat,
			.addressModeV = vk::SamplerAddressMode::eRepeat,
			.addressModeW = vk::SamplerAddressMode::eRepeat,
			.mipLodBias = 0.0,
			.anisotropyEnable = vk::True,
			.maxAnisotropy = properties.limits.maxSamplerAnisotropy};
		samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
		samplerInfo.unnormalizedCoordinates = vk::False;
		samplerInfo.compareEnable = vk::False;
		samplerInfo.compareOp = vk::CompareOp::eAlways;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = vk::LodClampNone;
		textureSampler = vk::raii::Sampler(device, samplerInfo);
	}

	// Parses an OBJ file and fills the vertices/indices arrays.
	// All vertices start with white color (1,1,1); texcoords flip the V axis
	// because OBJ uses bottom-left origin while Vulkan uses top-left.
	void loadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
							  MODEL_PATH.c_str()))
		{
			throw std::runtime_error(warn + err);
		}

		for (const auto &shape : shapes)
		{
			for (const auto &index : shape.mesh.indices)
			{
				Vertex vertex{};
				vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
							  attrib.vertices[3 * index.vertex_index + 1],
							  attrib.vertices[3 * index.vertex_index + 2]};

				vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
								   1.0f -
									   attrib.texcoords[2 * index.texcoord_index + 1]};

				vertex.color = {1.0f, 1.0f, 1.0f};

				vertices.push_back(vertex);
				indices.push_back(indices.size());
			}
		}
	}

	// =========================================================================
	// SECTION 10: SCENE BUFFERS
	// Uploads vertex, index, and uniform data to the GPU. Vertex and index
	// buffers are immutable (uploaded once via a staging buffer). Uniform
	// buffers are persistently mapped and updated every frame.
	// =========================================================================

	// Creates a host-visible staging buffer, copies vertices into it, then
	// copies from the staging buffer into a device-local vertex buffer.
	void createVertexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
		vk::raii::Buffer stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
					 vk::MemoryPropertyFlagBits::eHostVisible |
						 vk::MemoryPropertyFlagBits::eHostCoherent,
					 stagingBuffer, stagingBufferMemory);

		void *data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, vertices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize,
					 vk::BufferUsageFlagBits::eTransferDst |
						 vk::BufferUsageFlagBits::eVertexBuffer,
					 vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer,
					 vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	// Same pattern as createVertexBuffer but for the index buffer.
	void createIndexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
		vk::raii::Buffer stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
					 vk::MemoryPropertyFlagBits::eHostVisible |
						 vk::MemoryPropertyFlagBits::eHostCoherent,
					 stagingBuffer, stagingBufferMemory);

		void *data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, indices.data(), (size_t)bufferSize);
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize,
					 vk::BufferUsageFlagBits::eTransferDst |
						 vk::BufferUsageFlagBits::eIndexBuffer,
					 vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer,
					 indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}

	// Creates one host-visible, persistently mapped uniform buffer per
	// frame-in-flight. The CPU writes the MVP matrices directly into these
	// each frame via the mapped pointer, with no explicit map/unmap needed.
	void createUniformBuffers()
	{
		uniformBuffers.clear();
		uniformBuffersMemory.clear();
		uniformBuffersMapped.clear();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
			vk::raii::Buffer buffer({});
			vk::raii::DeviceMemory bufferMem({});
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
						 vk::MemoryPropertyFlagBits::eHostVisible |
							 vk::MemoryPropertyFlagBits::eHostCoherent,
						 buffer, bufferMem);
			uniformBuffers.emplace_back(std::move(buffer));
			uniformBuffersMemory.emplace_back(std::move(bufferMem));
			uniformBuffersMapped.emplace_back(
				uniformBuffersMemory[i].mapMemory(0, bufferSize));
		}
	}

	// =========================================================================
	// SECTION 11: DESCRIPTORS
	// Descriptors are the mechanism that binds GPU resources (buffers, images)
	// to shader binding slots. The pool allocates sets; each set is then updated
	// to point to the actual buffer/image objects.
	// =========================================================================

	// Creates the descriptor pool with enough capacity for MAX_FRAMES_IN_FLIGHT
	// sets, each containing one uniform buffer descriptor and one combined
	// image+sampler descriptor.
	void createDescriptorPool()
	{
		std::array poolSize{
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
								   MAX_FRAMES_IN_FLIGHT),
			vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,
								   MAX_FRAMES_IN_FLIGHT)};
		vk::DescriptorPoolCreateInfo poolInfo{
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
			.pPoolSizes = poolSize.data()};
		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

	// Allocates one descriptor set per frame-in-flight and writes the actual
	// buffer/image addresses into each set (binding 0 → uniform buffer,
	// binding 1 → texture sampler).
	void createDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
													 *descriptorSetLayout);
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data()};
		descriptorSets.clear();
		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo bufferInfo{.buffer = uniformBuffers[i],
												.offset = 0,
												.range = sizeof(UniformBufferObject)};
			vk::DescriptorImageInfo imageInfo{
				.sampler = textureSampler,
				.imageView = textureImageView,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
			std::array descriptorWrites{
				vk::WriteDescriptorSet{.dstSet = descriptorSets[i],
									   .dstBinding = 0,
									   .dstArrayElement = 0,
									   .descriptorCount = 1,
									   .descriptorType =
										   vk::DescriptorType::eUniformBuffer,
									   .pBufferInfo = &bufferInfo},
				vk::WriteDescriptorSet{.dstSet = descriptorSets[i],
									   .dstBinding = 1,
									   .dstArrayElement = 0,
									   .descriptorCount = 1,
									   .descriptorType =
										   vk::DescriptorType::eCombinedImageSampler,
									   .pImageInfo = &imageInfo}};
			device.updateDescriptorSets(descriptorWrites, {});
		}
	}

	// =========================================================================
	// SECTION 12: SYNCHRONIZATION
	// Three primitives coordinate the CPU-GPU and GPU-GPU handshakes:
	//   - presentComplete semaphore: GPU signals when the swapchain image is
	//     ready to render into (i.e. the display is done reading it).
	//   - renderFinished semaphore: GPU signals when rendering is done and the
	//     image can be handed to the presentation engine.
	//   - inFlightFence: CPU blocks on this until the GPU finishes the frame,
	//     ensuring we don't overwrite resources still in use.
	// =========================================================================

	void createSyncObjects()
	{
		assert(presentCompleteSemaphores.empty() &&
			   renderFinishedSemaphores.empty() && inFlightFences.empty());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
			inFlightFences.emplace_back(
				device,
				vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		}
	}

	// =========================================================================
	// SECTION 13: PER-FRAME RENDERING
	// This is the hot path that runs every frame. The flow is:
	//   mainLoop → drawFrame → recordCommandBuffer → [GPU executes] → present
	// =========================================================================

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}

		device.waitIdle();
	}

	/* Architecture of the drawFrame function:
																																																																																																																																	CPU GPU | |
																																																																																																																																	waitForFences
	   ◄──────────────── fence signaled (prev frame done) | | acquireNextImage | |
	   | reset
	   + record               │ (GPU still busy on other slot) | | queue.submit
	   ─────────────────► starts rendering |          wait on semaphore |
																																																																																																																																	queue.present
	   ◄─────────────── renderFinished semaphore signaled | | frameIndex++ |
	*/
	void drawFrame()
	{
		if (pendingMsaaSamples != msaaSamples)
		{
			msaaSamples = pendingMsaaSamples;
			rebuildMsaa();
		}

		if (pendingPresentMode != presentMode)
			recreateSwapChain();

		// Wait for the previous frame to finish rendering (CPU)
		auto fenceResult =
			device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}

		// Ask the swapchain for the next available framebuffer (image) to render
		// to.
		auto [result, imageIndex] = swapChain.acquireNextImage(
			UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

		// Handles when swap chain is no longer compatible (window drastically
		// resized or screen orientation changed, for example) The rendering
		// proceeds even if the surface is suboptimal (eSuboptimalKHR) (window
		// slightly resized, for example). Cases in which the surface is suboptimal:
		//
		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapChain();
			return;
		}
		else if (result != vk::Result::eSuccess &&
				 result != vk::Result::eSuboptimalKHR)
		{
			assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// Only reset the fence if we are submitting work, to avoid deadlocks.
		device.resetFences(*inFlightFences[frameIndex]);

		commandBuffers[frameIndex].reset();

		drawImGui();

		// Record the next rendering command
		recordCommandBuffer(imageIndex);

		// A flag that allows free execution of all other pipeline stages until
		// Color Attachment output.
		vk::PipelineStageFlags waitDestinationStageMask(
			vk::PipelineStageFlagBits::eColorAttachmentOutput);

		updateUniformBuffer(frameIndex);

		// Detailed information about the command to execute: wait on
		// presentCompleteSemaphore (don't start until the image is ready to be
		// written to) Signal renderFinishedSemaphores when done.
		const vk::SubmitInfo submitInfo{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
			.pWaitDstStageMask = &waitDestinationStageMask,
			.commandBufferCount = 1,
			.pCommandBuffers = &*commandBuffers[frameIndex],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]};

		// Submits a command with the previous info and the fence to signal when
		// done, so the CPU knows the next free slot at the next draw iteration.
		queue.submit(submitInfo, *inFlightFences[frameIndex]);

		// Present the image, only after render has finished.
		const vk::PresentInfoKHR presentInfoKHR{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
			.swapchainCount = 1,
			.pSwapchains = &*swapChain,
			.pImageIndices = &imageIndex};

		result = queue.presentKHR(presentInfoKHR);

		// Swapchain rewcreation: if at this point the swapchain is out of date or
		// suboptimal or the framebuffer has been resized, we need to recreate the
		// swap chain.
		if ((result == vk::Result::eSuboptimalKHR) ||
			(result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else
		{
			// There are no other success codes than eSuccess; on any error code,
			// presentKHR already threw an exception.
			assert(result == vk::Result::eSuccess);
		}

		switch (result)
		{
		case vk::Result::eSuccess:
			break;
		case vk::Result::eSuboptimalKHR:
			std::cout
				<< "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR !\n";
			break;
		default:
			break; // an unexpected result is returned!
		}

		// Update frame index.
		frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	// Computes and writes the MVP transform matrices for this frame into the
	// uniform buffer. The model rotates over time; view and projection are fixed.
	// Note: proj[1][1] is negated because GLM was designed for OpenGL, which
	// uses an inverted Y clip axis compared to Vulkan.
	void updateUniformBuffer(uint32_t currentImage)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(
						 currentTime - startTime)
						 .count();

		UniformBufferObject ubo{};
		ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
						   glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
						  glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f),
									static_cast<float>(swapChainExtent.width) /
										static_cast<float>(swapChainExtent.height),
									0.1f, 10.0f);

		// Flip the scaling factor of the Y axis in the projection matrix
		ubo.proj[1][1] *= -1;

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	// Records all GPU commands for one frame into commandBuffers[frameIndex].
	// The sequence is:
	//   1. Transition the MSAA color and swapchain images to color attachment
	//      layout, and the depth image to depth attachment layout.
	//   2. Begin dynamic rendering, specifying the MSAA image as color target
	//      (with the swapchain image as the resolve destination) and the depth
	//      image as the depth target.
	//   3. Bind pipeline, buffers, descriptors, set viewport/scissor, draw.
	//   4. End rendering.
	//   5. Transition the swapchain image to PresentSrcKHR so it can be shown.
	void recordCommandBuffer(uint32_t imageIndex)
	{
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});

		bool msaaEnabled = msaaSamples != vk::SampleCountFlagBits::e1;

		if (msaaEnabled)
		{
			// Transition MSAA color image to color attachment optimal (render target)
			transition_image_layout(
				*colorImage, vk::ImageLayout::eUndefined,
				vk::ImageLayout::eColorAttachmentOptimal, {},		// srcAccessMask
				vk::AccessFlagBits2::eColorAttachmentWrite,			// dstAccessMask
				vk::PipelineStageFlagBits2::eTopOfPipe,				// srcStageMask
				vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStageMask
				vk::ImageAspectFlagBits::eColor);
		}

		// Transition swapchain image to color attachment optimal (resolve target,
		// or direct render target when no MSAA)
		transition_image_layout(
			swapChainImages[imageIndex], vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal, {},		// srcAccessMask
			vk::AccessFlagBits2::eColorAttachmentWrite,			// dstAccessMask
			vk::PipelineStageFlagBits2::eTopOfPipe,				// srcStageMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStageMask
			vk::ImageAspectFlagBits::eColor);

		transition_image_layout(*depthImage, vk::ImageLayout::eUndefined,
								vk::ImageLayout::eDepthAttachmentOptimal,
								vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
								vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
								vk::PipelineStageFlagBits2::eEarlyFragmentTests |
									vk::PipelineStageFlagBits2::eLateFragmentTests,
								vk::PipelineStageFlagBits2::eEarlyFragmentTests |
									vk::PipelineStageFlagBits2::eLateFragmentTests,
								vk::ImageAspectFlagBits::eDepth);

		// Set up rendering attachment
		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

		// With MSAA: render into the MSAA color image, resolve into the swapchain
		// image at the end of the pass (storeOp DontCare since the MSAA image is
		// transient). Without MSAA: render directly into the swapchain image.
		vk::RenderingAttachmentInfo attachmentInfo =
			msaaEnabled
				? vk::RenderingAttachmentInfo{.imageView = *colorImageView,
											  .imageLayout = vk::ImageLayout::
												  eColorAttachmentOptimal,
											  .resolveMode =
												  vk::ResolveModeFlagBits::eAverage,
											  .resolveImageView =
												  *swapChainImageViews[imageIndex],
											  .resolveImageLayout =
												  vk::ImageLayout::
													  eColorAttachmentOptimal,
											  .loadOp =
												  vk::AttachmentLoadOp::eClear,
											  .storeOp =
												  vk::AttachmentStoreOp::eDontCare,
											  .clearValue = clearColor}
				: vk::RenderingAttachmentInfo{
					  .imageView = *swapChainImageViews[imageIndex],
					  .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
					  .loadOp = vk::AttachmentLoadOp::eClear,
					  .storeOp = vk::AttachmentStoreOp::eStore,
					  .clearValue = clearColor};

		vk::RenderingAttachmentInfo depthAttachmentInfo = {
			.imageView = depthImageView,
			.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eDontCare,
			.clearValue = clearDepth};

		// Set up rendering info
		vk::RenderingInfo renderingInfo{
			.renderArea = {vk::Offset2D{0, 0}, swapChainExtent},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachmentInfo,
			.pDepthAttachment = &depthAttachmentInfo};

		// Begin rendering
		commandBuffer.beginRendering(renderingInfo);

		// Bind pipeline
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
								   *graphicsPipeline);

		commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
		commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
		// Set dynamic viewport and scissor
		commandBuffer.setViewport(
			0,
			vk::Viewport{0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
						 static_cast<float>(swapChainExtent.height), 0.0f, 1.0f});
		commandBuffer.setScissor(0,
								 vk::Rect2D{vk::Offset2D{0, 0}, swapChainExtent});

		commandBuffers[frameIndex].bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
			*descriptorSets[frameIndex], nullptr);
		// Draw the triangle
		commandBuffer.drawIndexed(indices.size(), 1, 0, 0, 0);

		// End rendering
		commandBuffer.endRendering();

		vk::MemoryBarrier2 barrier{
			.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
			.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
							 vk::AccessFlagBits2::eColorAttachmentWrite,
		};
		commandBuffer.pipelineBarrier2(vk::DependencyInfo{
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &barrier,
		});

		// second pass: ImGui renders directly into swapchain image
		vk::RenderingAttachmentInfo imguiColorAttachment{
			.imageView = *swapChainImageViews[imageIndex],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eLoad, // load, don't clear — preserve
												   // your scene
			.storeOp = vk::AttachmentStoreOp::eStore};

		vk::RenderingInfo imguiRenderingInfo{
			.renderArea = {vk::Offset2D{0, 0}, swapChainExtent},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &imguiColorAttachment};

		commandBuffer.beginRendering(imguiRenderingInfo);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);
		commandBuffer.endRendering();

		// Transition image back to present source after rendering
		transition_image_layout(
			swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,			// srcAccessMask
			{},													// dstAccessMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStageMask
			vk::PipelineStageFlagBits2::eBottomOfPipe,			// dstStageMask
			vk::ImageAspectFlagBits::eColor);

		commandBuffer.end();
	}

	// Inline image layout transition used inside recordCommandBuffer.
	// Unlike transitionImageLayout (which allocates its own one-time command
	// buffer), this variant records into the current frame's command buffer
	// so it participates in the same submission as the rest of the frame.
	void transition_image_layout(vk::Image image, vk::ImageLayout oldLayout,
								 vk::ImageLayout newLayout,
								 vk::AccessFlags2 srcAccessMask,
								 vk::AccessFlags2 dstAccessMask,
								 vk::PipelineStageFlags2 srcStageMask,
								 vk::PipelineStageFlags2 dstStageMask,
								 vk::ImageAspectFlags image_aspect_flags)
	{
		vk::ImageMemoryBarrier2 barrier = {
			.srcStageMask = srcStageMask,
			.srcAccessMask = srcAccessMask,
			.dstStageMask = dstStageMask,
			.dstAccessMask = dstAccessMask,
			.oldLayout = oldLayout,
			.newLayout = newLayout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = {.aspectMask = image_aspect_flags,
								 .baseMipLevel = 0,
								 .levelCount = 1,
								 .baseArrayLayer = 0,
								 .layerCount = 1}};
		vk::DependencyInfo dependencyInfo = {.dependencyFlags = {},
											 .imageMemoryBarrierCount = 1,
											 .pImageMemoryBarriers = &barrier};
		commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
	}

	// =========================================================================
	// SECTION 14: LIFECYCLE — RECREATION & CLEANUP
	// The swapchain (and everything sized to it) must be recreated when the
	// window is resized. Everything else is destroyed once at shutdown.
	// =========================================================================

	// Called on window resize. Waits for the GPU to be idle, destroys all
	// swapchain-sized resources, then recreates them at the new size.
	void recreateSwapChain()
	{
		int width = 0, height = 0;

		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		device.waitIdle();

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		ImGui_ImplVulkan_SetMinImageCount(
			static_cast<uint32_t>(swapChainImages.size()));
		createDepthResources();
		createColorResources();
	}

	// Rebuilds the graphics pipeline and MSAA/depth attachments after a
	// runtime change to msaaSamples. Must be called with no frames in flight.
	void rebuildMsaa()
	{
		device.waitIdle();
		colorImageView = nullptr;
		colorImage = nullptr;
		colorImageMemory = nullptr;
		depthImageView = nullptr;
		depthImage = nullptr;
		depthImageMemory = nullptr;
		graphicsPipeline = nullptr;
		createDepthResources();
		createColorResources();
		createGraphicsPipeline();
	}

	// Destroys all resources that are tied to the swapchain extent.
	// Called both from recreateSwapChain() and from cleanup().
	void cleanupSwapChain()
	{
		colorImageView = nullptr;
		colorImage = nullptr;
		colorImageMemory = nullptr;
		depthImageView = nullptr;
		depthImage = nullptr;
		depthImageMemory = nullptr;
		swapChainImageViews.clear();
		swapChain = nullptr;
	}

	// Destroys all Vulkan objects in reverse dependency order (child objects
	// before parents). RAII handles call vkDestroy automatically when set to
	// nullptr, but order matters: e.g. the device must outlive everything
	// allocated from it, and the instance must outlive the device.
	void cleanup()
	{
		cleanupSwapChain();

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		inFlightFences.clear();
		renderFinishedSemaphores.clear();
		presentCompleteSemaphores.clear();
		commandBuffers.clear();
		commandPool = nullptr;
		graphicsPipeline = nullptr;
		pipelineLayout = nullptr;
		textureSampler = nullptr;
		textureImageView = nullptr;
		textureImage = nullptr;
		textureImageMemory = nullptr;
		descriptorSets.clear();
		descriptorPool = nullptr;
		imguiDescriptorPool = nullptr;
		uniformBuffers.clear();
		uniformBuffersMemory.clear();
		indexBuffer = nullptr;
		indexBufferMemory = nullptr;
		vertexBuffer = nullptr;
		vertexBufferMemory = nullptr;
		descriptorSetLayout = nullptr;
		queue = nullptr;
		device = nullptr;
		surface = nullptr;
		debugMessenger = nullptr;
		instance = nullptr;

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void drawImGui()
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::GetIO().DisplaySize = ImVec2(
			static_cast<float>(swapChainExtent.width),
			static_cast<float>(swapChainExtent.height));

		ImGui::Begin("Info");

		// Performance
		ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
		ImGui::Text("Frame time: %.3f ms", 1000.0f / ImGui::GetIO().Framerate);

		ImGui::Separator();

		// Display info
		ImGui::Text("Resolution: %ux%u", swapChainExtent.width,
					swapChainExtent.height);
		ImGui::Text("Device: %s", physicalDevice.getProperties().deviceName.data());
		ImGui::Text("Mip levels: %u", mipLevels);

		ImGui::Separator();

		// V-Sync toggle: eFifo = on, eMailbox = off
		bool vsync = (pendingPresentMode == vk::PresentModeKHR::eFifo);
		if (ImGui::Checkbox("V-Sync", &vsync))
			pendingPresentMode =
				vsync ? vk::PresentModeKHR::eFifo : vk::PresentModeKHR::eMailbox;

		// MSAA combo — lists all power-of-two counts up to hardware max
		static const vk::SampleCountFlagBits kSampleCounts[] = {
			vk::SampleCountFlagBits::e1,
			vk::SampleCountFlagBits::e2,
			vk::SampleCountFlagBits::e4,
			vk::SampleCountFlagBits::e8,
			vk::SampleCountFlagBits::e16,
			vk::SampleCountFlagBits::e32,
			vk::SampleCountFlagBits::e64,
		};
		if (ImGui::BeginCombo("MSAA", vk::to_string(pendingMsaaSamples).c_str()))
		{
			for (auto count : kSampleCounts)
			{
				if (!(supportedMsaaSamples & count))
					continue;
				bool selected = (count == pendingMsaaSamples);
				if (ImGui::Selectable(vk::to_string(count).c_str(), selected))
					pendingMsaaSamples = count;
				if (selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		ImGui::End();

		ImGui::Render();
	}

	void imGuiInit()
	{
		// ImGui needs its own descriptor pool
		vk::DescriptorPoolSize pool_size(vk::DescriptorType::eCombinedImageSampler,
										 1);
		vk::DescriptorPoolCreateInfo pool_info = {
			.sType = vk::StructureType::eDescriptorPoolCreateInfo,
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = 1,
			.poolSizeCount = 1,
			.pPoolSizes = &pool_size};
		imguiDescriptorPool = vk::raii::DescriptorPool(device, pool_info);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		float xscale, yscale;
		glfwGetWindowContentScale(window, &xscale, &yscale);
		ImGui::GetIO().Fonts->AddFontDefault();
		ImGui::GetStyle().ScaleAllSizes(xscale);
		ImGui_ImplGlfw_InitForVulkan(window, true);

		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = *instance;
		init_info.PhysicalDevice = *physicalDevice;
		init_info.Device = *device;
		init_info.Queue = *queue;
		init_info.DescriptorPool = *imguiDescriptorPool;
		init_info.MinImageCount = 2;
		init_info.ImageCount = swapChainImages.size();

		init_info.UseDynamicRendering = true;
		init_info.PipelineInfoMain.PipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = (VkFormat *)&swapChainSurfaceFormat.format};
		init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		ImGui_ImplVulkan_Init(&init_info);
	}
};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
