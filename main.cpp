#include <algorithm>
#include <array>
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

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

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow *window = nullptr;
	vk::raii::Context context;
	vk::raii::Instance instance = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR surface = nullptr;
	vk::raii::PhysicalDevice physicalDevice = nullptr;
	vk::raii::Device device = nullptr;
	vk::raii::Queue queue = nullptr;
	uint32_t graphicsIndex = ~0;
	uint32_t frameIndex = 0;
	vk::raii::SwapchainKHR swapChain = nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::SurfaceFormatKHR swapChainSurfaceFormat;
	vk::Extent2D swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline graphicsPipeline = nullptr;
	vk::raii::CommandPool commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;
	vk::raii::Buffer vertexBuffer = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;

	// A Vector of Semaphores to signal when frames have been presented
	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;

	// A Vector of Semaphores to signal when frames have been rendered
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;

	// A Fence used to wait for the previous frame to finish, on the CPU.
	std::vector<vk::raii::Fence> inFlightFences;

	bool framebufferResized = false;
	struct Vertex
	{
		glm::vec2 pos;
		glm::vec3 color;

		static vk::VertexInputBindingDescription getBindingDescription()
		{
			return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
		}

		static std::array<vk::VertexInputAttributeDescription, 2>
		getAttributeDescriptions()
		{
			return {vk::VertexInputAttributeDescription(
						0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
					vk::VertexInputAttributeDescription(
						1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))};
		}
	};
	const std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
										  {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
										  {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

	std::vector<const char *> requiredDeviceExtension = {
		vk::KHRSwapchainExtensionName};

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow *window, int width,
										  int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication *>(
			glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createGraphicsPipeline();
		createCommandPool();
		createVertexBuffer();
		createCommandBuffers();
		createSyncObjects();
	}

	void createImageViews()
	{
		assert(swapChainImageViews.empty());

		vk::ImageViewCreateInfo imageViewCreateInfo{
			.viewType = vk::ImageViewType::e2D,
			.format = swapChainSurfaceFormat.format,
			.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

		for (auto &image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

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
			.frontFace = vk::FrontFace::eClockwise,
			.depthBiasEnable = vk::False,
			.depthBiasSlopeFactor = 1.0f,
			.lineWidth = 1.0f};
		vk::PipelineMultisampleStateCreateInfo multisampling{
			.rasterizationSamples = vk::SampleCountFlagBits::e1,
			.sampleShadingEnable = vk::False};

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
			.setLayoutCount = 0, .pushConstantRangeCount = 0};

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapChainSurfaceFormat.format};
		vk::GraphicsPipelineCreateInfo pipelineInfo{
			.pNext = &pipelineRenderingCreateInfo,
			.stageCount = 2,
			.pStages = shaderStages,
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = pipelineLayout,
			.renderPass = nullptr};
		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = graphicsIndex};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	void createCommandBuffers()
	{
		commandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool = commandPool,
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = MAX_FRAMES_IN_FLIGHT};

		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	void createVertexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
		// create temp variable from where to save the staging buffer, create it with the function
		vk::raii::Buffer stagingBuffer(nullptr);
		vk::raii::DeviceMemory stagingBufferMemory(nullptr);

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, vertices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,  vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);
		
		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory &bufferMemory)
	{
		vk::BufferCreateInfo bufferInfo{
			.size = size,
			.usage = usage,
			.sharingMode = vk::SharingMode::eExclusive};
		buffer = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements =
			buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo memoryAllocateInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
		bufferMemory = vk::raii::DeviceMemory(device, memoryAllocateInfo);
		buffer.bindMemory(*bufferMemory, 0);
	}

	void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size){
				vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
		vk::raii::CommandBuffer       commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
		commandCopyBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
		commandCopyBuffer.end();
		queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
		queue.waitIdle();
	}

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

	void recordCommandBuffer(uint32_t imageIndex)
	{
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});

		// Transition image to color attachment optimal before rendering
		transition_image_layout(
			imageIndex, vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal, {},	   // srcAccessMask
			vk::AccessFlagBits2::eColorAttachmentWrite,		   // dstAccessMask
			vk::PipelineStageFlagBits2::eTopOfPipe,			   // srcStageMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput // dstStageMask
		);

		// Set up rendering attachment
		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo{
			.imageView = *swapChainImageViews[imageIndex],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor};

		// Set up rendering info
		vk::RenderingInfo renderingInfo{
			.renderArea = {vk::Offset2D{0, 0}, swapChainExtent},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachmentInfo};

		// Begin rendering
		commandBuffer.beginRendering(renderingInfo);

		// Bind pipeline
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
								   *graphicsPipeline);

		commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
		// Set dynamic viewport and scissor
		commandBuffer.setViewport(
			0,
			vk::Viewport{0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
						 static_cast<float>(swapChainExtent.height), 0.0f, 1.0f});
		commandBuffer.setScissor(0,
								 vk::Rect2D{vk::Offset2D{0, 0}, swapChainExtent});

		// Draw the triangle
		commandBuffer.draw(3, 1, 0, 0);

		// End rendering
		commandBuffer.endRendering();

		// Transition image back to present source after rendering
		transition_image_layout(
			imageIndex, vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,			// srcAccessMask
			{},													// dstAccessMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStageMask
			vk::PipelineStageFlagBits2::eBottomOfPipe			// dstStageMask
		);

		commandBuffer.end();
	}

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

	void transition_image_layout(uint32_t imageIndex, vk::ImageLayout oldLayout,
								 vk::ImageLayout newLayout,
								 vk::AccessFlags2 srcAccessMask,
								 vk::AccessFlags2 dstAccessMask,
								 vk::PipelineStageFlags2 srcStageMask,
								 vk::PipelineStageFlags2 dstStageMask)
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
			.image = swapChainImages[imageIndex],
			.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
								 .baseMipLevel = 0,
								 .levelCount = 1,
								 .baseArrayLayer = 0,
								 .layerCount = 1}};
		vk::DependencyInfo dependencyInfo = {.dependencyFlags = {},
											 .imageMemoryBarrierCount = 1,
											 .pImageMemoryBarriers = &barrier};
		commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
	}

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
					CPU                          GPU
					|                            |
					waitForFences ◄──────────────── fence signaled (prev frame
	   done) |                            | acquireNextImage             | | |
					reset + record               │ (GPU still busy on other slot)
					|                            |
					queue.submit ─────────────────► starts rendering
					|          wait on semaphore |
					queue.present ◄─────────────── renderFinished semaphore
	   signaled |                            | frameIndex++                 |
	*/
	void drawFrame()
	{
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

		// Record the next rendering command
		recordCommandBuffer(imageIndex);

		// A flag that allows free execution of all other pipeline stages until
		// Color Attachment output.
		vk::PipelineStageFlags waitDestinationStageMask(
			vk::PipelineStageFlagBits::eColorAttachmentOutput);

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

	void cleanup()
	{
		cleanupSwapChain();

		inFlightFences.clear();
		renderFinishedSemaphores.clear();
		presentCompleteSemaphores.clear();
		commandBuffers.clear();
		commandPool = nullptr;
		graphicsPipeline = nullptr;
		pipelineLayout = nullptr;
		queue = nullptr;
		device = nullptr;
		surface = nullptr;
		debugMessenger = nullptr;
		instance = nullptr;

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void cleanupSwapChain()
	{
		swapChainImageViews.clear();
		swapChain = nullptr;
	}

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

	void createSurface()
	{
		VkSurfaceKHR _surface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice)
	{
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
			   supportsAllRequiredExtensions && supportsRequiredFeatures;
	}

	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> physicalDevices =
			instance.enumeratePhysicalDevices();
		auto const devIter =
			std::ranges::find_if(physicalDevices, [&](auto const &physicalDevice)
								 { return isDeviceSuitable(physicalDevice); });
		if (devIter == physicalDevices.end())
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
		physicalDevice = *devIter;
	}

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
				{}, // vk::PhysicalDeviceFeatures2
				{.shaderDrawParameters =
					 true},					// vk::PhysicalDeviceVulkan11Features
				{.dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
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
		vk::PresentModeKHR presentMode =
			chooseSwapPresentMode(availablePresentModes);

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

	[[nodiscard]] vk::raii::ShaderModule
	createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{
			.codeSize = code.size() * sizeof(char),
			.pCode = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule shaderModule{device, createInfo};
		return shaderModule;
	}

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
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}