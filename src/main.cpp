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

#include <nlohmann/json.hpp>

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

#include "Device.hpp"
#include "Swapchain.hpp"

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

const std::string TEXTURE_PATH = "textures/viking_room.png";
const std::string SCENE_PATH   = "scenes/scene.json";

/* This is used to make the drawing process more efficient by avoiding the CPU
being idle while the GPU renders frame N. By having multiple frames in flight,
while the CPU records commands for frame N+1, the GPU renders frame N.*/
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

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

struct Renderable {
    vk::raii::Buffer         vertexBuffer       = nullptr;
    vk::raii::DeviceMemory   vertexBufferMemory = nullptr;
    vk::raii::Buffer         indexBuffer        = nullptr;
    vk::raii::DeviceMemory   indexBufferMemory  = nullptr;
    uint32_t                 indexCount         = 0;

    glm::mat4                modelMatrix        = glm::mat4(1.0f);
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
	// Owns Context → Instance → DebugMessenger → Surface → PhysicalDevice →
	// Device → Queue. Constructed in initVulkan, destroyed last in cleanup.
	std::unique_ptr<Device> vulkanDevice;

	// --- Swapchain ---
	// The swapchain is a ring buffer of images we render into and present.
	std::unique_ptr<Swapchain> swapchain;

	// --- Graphics pipeline ---
	// The pipeline encodes the full state of the GPU rendering stages: shaders,
	// rasterization, blending, depth test, etc.
	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline graphicsPipeline = nullptr;

	vk::PresentModeKHR pendingPresentMode = vk::PresentModeKHR::eMailbox;

	// --- Command recording ---
	// One pool owns the memory; one command buffer per frame in flight.
	vk::raii::CommandPool commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;
	uint32_t frameIndex = 0;

	// --- Render attachments (recreated alongside the swapchain on resize) ---
	// MSAA: we render into a multisampled color image and resolve into the
	// swapchain image. The depth image tests fragment depth for occlusion.
	// msaaSamples: currently active count. pendingMsaaSamples: user-requested
	// change (applied at the start of the next frame). max and supported counts
	// are queried from vulkanDevice.
	vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
	vk::SampleCountFlagBits pendingMsaaSamples = vk::SampleCountFlagBits::e1;
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
	// One Renderable per object; each owns its own GPU buffers and model matrix.
	std::vector<Renderable> renderables;

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

	// Data layout of the uniform buffer as the shader sees it.
	struct UniformBufferObject
	{
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

	void initVulkan()
	{
#ifdef NDEBUG
		std::vector<const char *> layers = {};
#else
		std::vector<const char *> layers = {"VK_LAYER_KHRONOS_validation"};
#endif
		vulkanDevice = std::make_unique<Device>(
			window,
			std::vector<const char *>{vk::KHRSwapchainExtensionName},
			layers);
		msaaSamples = vulkanDevice->getMsaaSamples();
		pendingMsaaSamples = msaaSamples;
		swapchain = std::make_unique<Swapchain>(*vulkanDevice, window, pendingPresentMode);
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createColorResources();
		createDepthResources();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		loadScene();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
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
		descriptorSetLayout = vk::raii::DescriptorSetLayout(vulkanDevice->getLogicalDevice(), layoutInfo);
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
							  .width = static_cast<float>(swapchain->getExtent().width),
							  .height = static_cast<float>(swapchain->getExtent().height),
							  .minDepth = 0.0f,
							  .maxDepth = 1.0f};
		vk::Rect2D scissor{.offset = vk::Offset2D{0, 0}, .extent = swapchain->getExtent()};
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
		vk::PushConstantRange pushConstantRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex,
			.offset = 0,
			.size = sizeof(glm::mat4)};
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = 1,
			.pSetLayouts = &*descriptorSetLayout,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pushConstantRange};

		pipelineLayout = vk::raii::PipelineLayout(vulkanDevice->getLogicalDevice(), pipelineLayoutInfo);
		vk::Format depthFormat = findDepthFormat();
		vk::PipelineDepthStencilStateCreateInfo depthStencil{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False};
		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapchain->getSurfaceFormat().format,
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
		graphicsPipeline = vk::raii::Pipeline(vulkanDevice->getLogicalDevice(), nullptr, pipelineInfo);
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
		vk::raii::ShaderModule shaderModule{vulkanDevice->getLogicalDevice(), createInfo};
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
			vk::FormatProperties props = vulkanDevice->getPhysicalDevice().getFormatProperties(format);
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
	// buffers and then submitted to a vulkanDevice->getGraphicsQueue(). The command pool is the allocator
	// for these buffers — it is tied to a specific queue family.
	// =========================================================================

	// Creates the command pool for our graphics queue family. The
	// eResetCommandBuffer flag lets us re-record each buffer every frame without
	// needing to free and reallocate.
	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = vulkanDevice->getGraphicsIndex()};
		commandPool = vk::raii::CommandPool(vulkanDevice->getLogicalDevice(), poolInfo);
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

		commandBuffers = vk::raii::CommandBuffers(vulkanDevice->getLogicalDevice(), allocInfo);
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
		vk::Format colorFormat = swapchain->getSurfaceFormat().format;

		createImage(swapchain->getExtent().width, swapchain->getExtent().height, 1, msaaSamples,
					colorFormat, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransientAttachment |
						vk::ImageUsageFlagBits::eColorAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage,
					colorImageMemory);
		colorImageView = vulkanDevice->createImageView(colorImage, colorFormat,
													   vk::ImageAspectFlagBits::eColor, 1);
	}

	// Creates the depth image and its view. The format is chosen at runtime
	// based on what the GPU supports (D32, D32S8, or D24S8).
	// The depth image must have the same MSAA sample count as the color image.
	void createDepthResources()
	{
		vk::Format depthFormat = findDepthFormat();
		createImage(swapchain->getExtent().width, swapchain->getExtent().height, 1, msaaSamples,
					depthFormat, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eDepthStencilAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
					depthImageMemory);
		depthImageView = vulkanDevice->createImageView(depthImage, depthFormat,
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

		image = vk::raii::Image(vulkanDevice->getLogicalDevice(), imageInfo);

		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex =
				vulkanDevice->findMemoryType(memRequirements.memoryTypeBits, properties)};
		imageMemory = vk::raii::DeviceMemory(vulkanDevice->getLogicalDevice(), allocInfo);
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
		buffer = vk::raii::Buffer(vulkanDevice->getLogicalDevice(), bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo memoryAllocateInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex =
				vulkanDevice->findMemoryType(memRequirements.memoryTypeBits, properties)};
		bufferMemory = vk::raii::DeviceMemory(vulkanDevice->getLogicalDevice(), memoryAllocateInfo);
		buffer.bindMemory(*bufferMemory, 0);
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
				std::move(vk::raii::CommandBuffers(vulkanDevice->getLogicalDevice(), allocInfo).front()));

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
		vulkanDevice->getGraphicsQueue().submit(submitInfo, nullptr);
		vulkanDevice->getGraphicsQueue().waitIdle();
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
			vulkanDevice->getPhysicalDevice().getFormatProperties(imageFormat);
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
			vulkanDevice->createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb,
										  vk::ImageAspectFlagBits::eColor, mipLevels);
	}

	// Creates the sampler that describes how the fragment shader reads the
	// texture: linear filtering (smooth), repeat wrapping, anisotropic
	// filtering up to the hardware maximum, and full mip range.
	void createTextureSampler()
	{
		vk::PhysicalDeviceProperties properties = vulkanDevice->getPhysicalDevice().getProperties();
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
		textureSampler = vk::raii::Sampler(vulkanDevice->getLogicalDevice(), samplerInfo);
	}

	// =========================================================================
	// SECTION 9b: PROCEDURAL GEOMETRY
	// Hard-coded vertex/index data for built-in mesh types.
	// Each mesh is defined in local object space; the model matrix (push
	// constant) places it in world space at draw time.
	// =========================================================================

	static std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeCube()
	{
		std::vector<Vertex> verts = {
			// +Z face (front)
			{{-0.5f, -0.5f,  0.5f}, {1,1,1}, {0,0}},
			{{ 0.5f, -0.5f,  0.5f}, {1,1,1}, {1,0}},
			{{ 0.5f,  0.5f,  0.5f}, {1,1,1}, {1,1}},
			{{-0.5f,  0.5f,  0.5f}, {1,1,1}, {0,1}},
			// -Z face (back)
			{{ 0.5f, -0.5f, -0.5f}, {1,1,1}, {0,0}},
			{{-0.5f, -0.5f, -0.5f}, {1,1,1}, {1,0}},
			{{-0.5f,  0.5f, -0.5f}, {1,1,1}, {1,1}},
			{{ 0.5f,  0.5f, -0.5f}, {1,1,1}, {0,1}},
			// +X face (right)
			{{ 0.5f, -0.5f,  0.5f}, {1,1,1}, {0,0}},
			{{ 0.5f, -0.5f, -0.5f}, {1,1,1}, {1,0}},
			{{ 0.5f,  0.5f, -0.5f}, {1,1,1}, {1,1}},
			{{ 0.5f,  0.5f,  0.5f}, {1,1,1}, {0,1}},
			// -X face (left)
			{{-0.5f, -0.5f, -0.5f}, {1,1,1}, {0,0}},
			{{-0.5f, -0.5f,  0.5f}, {1,1,1}, {1,0}},
			{{-0.5f,  0.5f,  0.5f}, {1,1,1}, {1,1}},
			{{-0.5f,  0.5f, -0.5f}, {1,1,1}, {0,1}},
			// +Y face (top)
			{{-0.5f,  0.5f,  0.5f}, {1,1,1}, {0,0}},
			{{ 0.5f,  0.5f,  0.5f}, {1,1,1}, {1,0}},
			{{ 0.5f,  0.5f, -0.5f}, {1,1,1}, {1,1}},
			{{-0.5f,  0.5f, -0.5f}, {1,1,1}, {0,1}},
			// -Y face (bottom)
			{{-0.5f, -0.5f, -0.5f}, {1,1,1}, {0,0}},
			{{ 0.5f, -0.5f, -0.5f}, {1,1,1}, {1,0}},
			{{ 0.5f, -0.5f,  0.5f}, {1,1,1}, {1,1}},
			{{-0.5f, -0.5f,  0.5f}, {1,1,1}, {0,1}},
		};
		std::vector<uint32_t> idxs;
		for (uint32_t f = 0; f < 6; ++f) {
			uint32_t b = f * 4;
			idxs.insert(idxs.end(), {b,b+1,b+2, b,b+2,b+3});
		}
		return {verts, idxs};
	}

	static std::pair<std::vector<Vertex>, std::vector<uint32_t>> makePlane()
	{
		std::vector<Vertex> verts = {
			{{-2.0f, -2.0f, 0.0f}, {1,0.5,1}, {0,0}},
			{{ 2.0f, -2.0f, 0.0f}, {1,0.5,1}, {1,0}},
			{{ 2.0f,  2.0f, 0.0f}, {1,0.5,1}, {1,1}},
			{{-2.0f,  2.0f, 0.0f}, {1,0.5,1}, {0,1}},
		};
		std::vector<uint32_t> idxs = {0,1,2, 0,2,3};
		return {verts, idxs};
	}

	// =========================================================================
	// SECTION 10: SCENE BUFFERS
	// Uploads vertex, index, and uniform data to the GPU. Vertex and index
	// buffers are immutable (uploaded once via a staging buffer). Uniform
	// buffers are persistently mapped and updated every frame.
	// =========================================================================

	// Uploads geometry into a Renderable's GPU buffers via staging buffers.
	void uploadRenderable(Renderable &r, const std::vector<Vertex> &verts,
						  const std::vector<uint32_t> &idxs)
	{
		// Vertex buffer
		{
			vk::DeviceSize size = sizeof(verts[0]) * verts.size();
			vk::raii::Buffer staging({});
			vk::raii::DeviceMemory stagingMem({});
			createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
						 vk::MemoryPropertyFlagBits::eHostVisible |
							 vk::MemoryPropertyFlagBits::eHostCoherent,
						 staging, stagingMem);
			void *data = stagingMem.mapMemory(0, size);
			memcpy(data, verts.data(), size);
			stagingMem.unmapMemory();
			createBuffer(size,
						 vk::BufferUsageFlagBits::eTransferDst |
							 vk::BufferUsageFlagBits::eVertexBuffer,
						 vk::MemoryPropertyFlagBits::eDeviceLocal,
						 r.vertexBuffer, r.vertexBufferMemory);
			copyBuffer(staging, r.vertexBuffer, size);
		}
		// Index buffer
		{
			vk::DeviceSize size = sizeof(idxs[0]) * idxs.size();
			vk::raii::Buffer staging({});
			vk::raii::DeviceMemory stagingMem({});
			createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
						 vk::MemoryPropertyFlagBits::eHostVisible |
							 vk::MemoryPropertyFlagBits::eHostCoherent,
						 staging, stagingMem);
			void *data = stagingMem.mapMemory(0, size);
			memcpy(data, idxs.data(), size);
			stagingMem.unmapMemory();
			createBuffer(size,
						 vk::BufferUsageFlagBits::eTransferDst |
							 vk::BufferUsageFlagBits::eIndexBuffer,
						 vk::MemoryPropertyFlagBits::eDeviceLocal,
						 r.indexBuffer, r.indexBufferMemory);
			copyBuffer(staging, r.indexBuffer, size);
		}
		r.indexCount = static_cast<uint32_t>(idxs.size());
	}

	// Reads scene.json, instantiates a Renderable for each listed object.
	void loadScene()
	{
		std::ifstream f(SCENE_PATH);
		if (!f.is_open())
			throw std::runtime_error("failed to open scene file: " + SCENE_PATH);

		nlohmann::json scene = nlohmann::json::parse(f);

		for (const auto &obj : scene["objects"])
		{
			std::string mesh = obj["mesh"];
			glm::vec3 pos = {obj["position"][0], obj["position"][1], obj["position"][2]};

			auto [verts, idxs] = (mesh == "cube") ? makeCube() : makePlane();

			Renderable r;
			r.modelMatrix = glm::translate(glm::mat4(1.0f), pos);
			uploadRenderable(r, verts, idxs);
			renderables.push_back(std::move(r));
		}
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
		descriptorPool = vk::raii::DescriptorPool(vulkanDevice->getLogicalDevice(), poolInfo);
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
		descriptorSets = vulkanDevice->getLogicalDevice().allocateDescriptorSets(allocInfo);

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
			vulkanDevice->getLogicalDevice().updateDescriptorSets(descriptorWrites, {});
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

		for (size_t i = 0; i < swapchain->getImages().size(); i++)
		{
			renderFinishedSemaphores.emplace_back(vulkanDevice->getLogicalDevice(), vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			presentCompleteSemaphores.emplace_back(vulkanDevice->getLogicalDevice(), vk::SemaphoreCreateInfo());
			inFlightFences.emplace_back(
				vulkanDevice->getLogicalDevice(),
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

		vulkanDevice->getLogicalDevice().waitIdle();
	}

	/* Architecture of the drawFrame function:
																																																																																																																																	CPU GPU | |
																																																																																																																																	waitForFences
	   ◄──────────────── fence signaled (prev frame done) | | acquireNextImage | |
	   | reset
	   + record               │ (GPU still busy on other slot) | | vulkanDevice->getGraphicsQueue().submit
	   ─────────────────► starts rendering |          wait on semaphore |
																																																																																																																																	vulkanDevice->getGraphicsQueue().present
	   ◄─────────────── renderFinished semaphore signaled | | frameIndex++ |
	*/
	void drawFrame()
	{
		if (pendingMsaaSamples != msaaSamples)
		{
			msaaSamples = pendingMsaaSamples;
			rebuildMsaa();
		}

		if (pendingPresentMode != swapchain->getPresentMode())
			recreateSwapChain();

		// Wait for the previous frame to finish rendering (CPU)
		auto fenceResult =
			vulkanDevice->getLogicalDevice().waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}

		// Ask the swapchain for the next available framebuffer (image) to render
		// to.
		auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(
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
		vulkanDevice->getLogicalDevice().resetFences(*inFlightFences[frameIndex]);

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
		vulkanDevice->getGraphicsQueue().submit(submitInfo, *inFlightFences[frameIndex]);

		// Present the image, only after render has finished.
		const vk::PresentInfoKHR presentInfoKHR{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
			.swapchainCount = 1,
			.pSwapchains = &*swapchain->getSwapChain(),
			.pImageIndices = &imageIndex};

		result = vulkanDevice->getGraphicsQueue().presentKHR(presentInfoKHR);

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
		ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
						  glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f),
									static_cast<float>(swapchain->getExtent().width) /
										static_cast<float>(swapchain->getExtent().height),
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
			swapchain->getImages()[imageIndex], vk::ImageLayout::eUndefined,
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
												  *swapchain->getImageViews()[imageIndex],
											  .resolveImageLayout =
												  vk::ImageLayout::
													  eColorAttachmentOptimal,
											  .loadOp =
												  vk::AttachmentLoadOp::eClear,
											  .storeOp =
												  vk::AttachmentStoreOp::eDontCare,
											  .clearValue = clearColor}
				: vk::RenderingAttachmentInfo{
					  .imageView = *swapchain->getImageViews()[imageIndex],
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
			.renderArea = {vk::Offset2D{0, 0}, swapchain->getExtent()},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachmentInfo,
			.pDepthAttachment = &depthAttachmentInfo};

		// Begin rendering
		commandBuffer.beginRendering(renderingInfo);

		// Bind pipeline
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
								   *graphicsPipeline);

		commandBuffer.setViewport(
			0,
			vk::Viewport{0.0f, 0.0f, static_cast<float>(swapchain->getExtent().width),
						 static_cast<float>(swapchain->getExtent().height), 0.0f, 1.0f});
		commandBuffer.setScissor(0,
								 vk::Rect2D{vk::Offset2D{0, 0}, swapchain->getExtent()});

		commandBuffers[frameIndex].bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
			*descriptorSets[frameIndex], nullptr);

		for (const auto &r : renderables)
		{
			commandBuffer.pushConstants2(
				vk::PushConstantsInfo{}
					.setLayout(*pipelineLayout)
					.setStageFlags(vk::ShaderStageFlagBits::eVertex)
					.setOffset(0)
					.setSize(sizeof(glm::mat4))
					.setPValues(&r.modelMatrix));
			commandBuffer.bindVertexBuffers(0, *r.vertexBuffer, {0});
			commandBuffer.bindIndexBuffer(*r.indexBuffer, 0, vk::IndexType::eUint32);
			commandBuffer.drawIndexed(r.indexCount, 1, 0, 0, 0);
		}

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
			.imageView = *swapchain->getImageViews()[imageIndex],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eLoad, // load, don't clear — preserve
												   // your scene
			.storeOp = vk::AttachmentStoreOp::eStore};

		vk::RenderingInfo imguiRenderingInfo{
			.renderArea = {vk::Offset2D{0, 0}, swapchain->getExtent()},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &imguiColorAttachment};

		commandBuffer.beginRendering(imguiRenderingInfo);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);
		commandBuffer.endRendering();

		// Transition image back to present source after rendering
		transition_image_layout(
			swapchain->getImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
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
		vulkanDevice->getLogicalDevice().waitIdle();
		cleanupSwapChain();
		swapchain->recreate(window, pendingPresentMode);
		ImGui_ImplVulkan_SetMinImageCount(
			static_cast<uint32_t>(swapchain->getImages().size()));
		createDepthResources();
		createColorResources();
	}

	// Rebuilds the graphics pipeline and MSAA/depth attachments after a
	// runtime change to msaaSamples. Must be called with no frames in flight.
	void rebuildMsaa()
	{
		vulkanDevice->getLogicalDevice().waitIdle();
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

	// Destroys Application-owned resources tied to the swapchain extent.
	// The swapchain itself (image views, handle) is managed by the Swapchain object.
	void cleanupSwapChain()
	{
		colorImageView = nullptr;
		colorImage = nullptr;
		colorImageMemory = nullptr;
		depthImageView = nullptr;
		depthImage = nullptr;
		depthImageMemory = nullptr;
	}

	// Destroys all Vulkan objects in reverse dependency order (child objects
	// before parents). RAII handles call vkDestroy automatically when set to
	// nullptr, but order matters: e.g. the device must outlive everything
	// allocated from it, and the instance must outlive the vulkanDevice->getLogicalDevice().
	void cleanup()
	{
		cleanupSwapChain();

		renderables.clear();

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
		descriptorSetLayout = nullptr;
		swapchain.reset();
		vulkanDevice.reset();

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void drawImGui()
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::GetIO().DisplaySize = ImVec2(
			static_cast<float>(swapchain->getExtent().width),
			static_cast<float>(swapchain->getExtent().height));

		ImGui::Begin("Info");

		// Performance
		ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
		ImGui::Text("Frame time: %.3f ms", 1000.0f / ImGui::GetIO().Framerate);

		ImGui::Separator();

		// Display info
		ImGui::Text("Resolution: %ux%u", swapchain->getExtent().width,
					swapchain->getExtent().height);
		ImGui::Text("Device: %s", vulkanDevice->getPhysicalDevice().getProperties().deviceName.data());
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
				if (!(vulkanDevice->getSupportedMsaaSamples() & count))
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
		// ImGui needs its own descriptor pool.
		// imgui 1.92+ requires at least IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE (8)
		// descriptors per font atlas; 1 was too small and caused silent allocation failures.
		vk::DescriptorPoolSize pool_size(vk::DescriptorType::eCombinedImageSampler,
										 IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE);
		vk::DescriptorPoolCreateInfo pool_info = {
			.sType = vk::StructureType::eDescriptorPoolCreateInfo,
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE,
			.poolSizeCount = 1,
			.pPoolSizes = &pool_size};
		imguiDescriptorPool = vk::raii::DescriptorPool(vulkanDevice->getLogicalDevice(), pool_info);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		float xscale, yscale;
		glfwGetWindowContentScale(window, &xscale, &yscale);
		ImGui::GetIO().Fonts->AddFontDefault();
		ImGui::GetStyle().ScaleAllSizes(xscale);
		ImGui_ImplGlfw_InitForVulkan(window, true);

		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = *vulkanDevice->getInstance();
		init_info.PhysicalDevice = *vulkanDevice->getPhysicalDevice();
		init_info.Device = *vulkanDevice->getLogicalDevice();
		init_info.Queue = *vulkanDevice->getGraphicsQueue();
		init_info.DescriptorPool = *imguiDescriptorPool;
		init_info.MinImageCount = 2;
		init_info.ImageCount = swapchain->getImages().size();

		init_info.UseDynamicRendering = true;
		init_info.PipelineInfoMain.PipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = (VkFormat *)&swapchain->getSurfaceFormat().format};
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
