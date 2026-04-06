#include "VulkanRenderer.hpp"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <limits>

#include <nlohmann/json.hpp>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
const std::string TEXTURE_PATH = "textures/viking_room.png";
const std::string SCENE_PATH = "scenes/scene.json";

vk::VertexInputBindingDescription VulkanRenderer::getVertexBindingDescription()
{
	return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
}

std::array<vk::VertexInputAttributeDescription, 4> VulkanRenderer::getVertexAttributeDescriptions()
{
	return {
		vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
		vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
		vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)),
		vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal))};
}

void VulkanRenderer::run()
{
	initWindow();
	initVulkan();
	imGuiInit();
	mainLoop();
	cleanup();
}

void VulkanRenderer::initWindow()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(WIDTH, HEIGHT, "blaz-engine", nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

// GLFW calls this when the window is resized. We can't recreate the
// swapchain here directly (we're outside the render loop), so we just set a
// flag that drawFrame checks at the end of each frame.
void VulkanRenderer::framebufferResizeCallback(GLFWwindow *window, int width,
											   int height)
{
	auto app = reinterpret_cast<VulkanRenderer *>(
		glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}

void VulkanRenderer::initVulkan()
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
	createShadowPipeline();
	createCommandPool();
	createColorResources();
	createDepthResources();
	createShadowMapResources();
	createShadowMapSampler();
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

void VulkanRenderer::createDescriptorSetLayout()
{
	std::array bindings = {vk::DescriptorSetLayoutBinding(
							   0, vk::DescriptorType::eUniformBuffer, 1,
							   vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
						   vk::DescriptorSetLayoutBinding(
							   1, vk::DescriptorType::eCombinedImageSampler, 1,
							   vk::ShaderStageFlagBits::eFragment, nullptr),
						   vk::DescriptorSetLayoutBinding(
							   2, vk::DescriptorType::eCombinedImageSampler, 1,
							   vk::ShaderStageFlagBits::eFragment, nullptr)};

	vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.bindingCount = bindings.size(), .pBindings = bindings.data()};
	descriptorSetLayout = vk::raii::DescriptorSetLayout(vulkanDevice->getLogicalDevice(), layoutInfo);
}

void VulkanRenderer::createGraphicsPipeline()
{
	vk::raii::ShaderModule shaderModule =
		createShaderModule(readFile("shaders/scene.spv"));
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
	auto bindingDescription = getVertexBindingDescription();
	auto attributeDescriptions = getVertexAttributeDescriptions();
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
	vk::Format depthFormat = vulkanDevice->findDepthFormat();
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

void VulkanRenderer::createShadowPipeline()
{
	vk::raii::ShaderModule shaderModule =
		createShaderModule(readFile("shaders/shadow.spv"));

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eVertex,
		.module = shaderModule,
		.pName = "vertMain"};

	auto bindingDescription = getVertexBindingDescription();

	vk::VertexInputAttributeDescription posAttr{
		0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)};

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = 1,
		.pVertexAttributeDescriptions = &posAttr};

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		.topology = vk::PrimitiveTopology::eTriangleList};

	vk::Viewport viewport{.x = 0.0f,
						  .y = 0.0f,
						  .width = static_cast<float>(SHADOW_MAP_SIZE),
						  .height = static_cast<float>(SHADOW_MAP_SIZE),
						  .minDepth = 0.0f,
						  .maxDepth = 1.0f};
	vk::Rect2D scissor{.offset = vk::Offset2D{0, 0}, .extent = vk::Extent2D{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1,
													  .pViewports = &viewport,
													  .scissorCount = 1,
													  .pScissors = &scissor};
	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eFront,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = vk::True,
		.depthBiasConstantFactor = 1.25f,
		.depthBiasSlopeFactor = 1.75f,
		.lineWidth = 1.0f};
	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = vk::SampleCountFlagBits::e1};
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable = vk::True,
		.depthWriteEnable = vk::True,
		.depthCompareOp = vk::CompareOp::eLessOrEqual,
	};
	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.attachmentCount = 0};

	vk::Format depthFormat = vulkanDevice->findDepthFormat();
	vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
		.colorAttachmentCount = 0,
		.depthAttachmentFormat = depthFormat};

	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.pNext = &pipelineRenderingCreateInfo,
		.stageCount = 1,
		.pStages = &vertShaderStageInfo,
		.pVertexInputState = &vertexInputInfo,
		.pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlending,
		.layout = pipelineLayout};

	shadowPipeline = vk::raii::Pipeline(vulkanDevice->getLogicalDevice(), nullptr, pipelineInfo);
}

std::vector<char> VulkanRenderer::readFile(const std::string &filename)
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
VulkanRenderer::createShaderModule(const std::vector<char> &code)
{
	vk::ShaderModuleCreateInfo createInfo{
		.codeSize = code.size() * sizeof(char),
		.pCode = reinterpret_cast<const uint32_t *>(code.data())};
	vk::raii::ShaderModule shaderModule{vulkanDevice->getLogicalDevice(), createInfo};
	return shaderModule;
}

bool VulkanRenderer::hasStencilComponent(vk::Format format)
{
	return format == vk::Format::eD32SfloatS8Uint ||
		   format == vk::Format::eD24UnormS8Uint;
}

// eResetCommandBuffer lets us re-record each buffer every frame without freeing it.
void VulkanRenderer::createCommandPool()
{
	vk::CommandPoolCreateInfo poolInfo{
		.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		.queueFamilyIndex = vulkanDevice->getGraphicsIndex()};
	commandPool = vk::raii::CommandPool(vulkanDevice->getLogicalDevice(), poolInfo);
}

void VulkanRenderer::createCommandBuffers()
{
	commandBuffers.clear();
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = MAX_FRAMES_IN_FLIGHT};

	commandBuffers = vk::raii::CommandBuffers(vulkanDevice->getLogicalDevice(), allocInfo);
}

// MSAA color image: GPU renders here at multiple samples/pixel, then resolves
// (averages) into the swapchain image. eTransient because its contents never
// leave the GPU — consumed immediately by the resolve step.
void VulkanRenderer::createColorResources()
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

// Depth image format is chosen at runtime (D32, D32S8, or D24S8); must match
// the MSAA sample count of the color image.
void VulkanRenderer::createDepthResources()
{
	vk::Format depthFormat = vulkanDevice->findDepthFormat();
	createImage(swapchain->getExtent().width, swapchain->getExtent().height, 1, msaaSamples,
				depthFormat, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eDepthStencilAttachment,
				vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
				depthImageMemory);
	depthImageView = vulkanDevice->createImageView(depthImage, depthFormat,
												   vk::ImageAspectFlagBits::eDepth, 1);
}

void VulkanRenderer::createShadowMapResources()
{
	vk::Format depthFormat = vulkanDevice->findDepthFormat();
	createImage(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1, vk::SampleCountFlagBits::e1,
				depthFormat, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eDepthStencilAttachment |
					vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal, shadowMapImage,
				shadowMapImageMemory);
	shadowMapImageView = vulkanDevice->createImageView(
		shadowMapImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

void VulkanRenderer::createShadowMapSampler()
{
	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW = vk::SamplerAddressMode::eClampToEdge,
		.mipLodBias = 0.0f,
		.anisotropyEnable = vk::False,
		.maxAnisotropy = 1.0f,
		.compareEnable = vk::True,
		.compareOp = vk::CompareOp::eLessOrEqual,
		.minLod = 0.0f,
		.maxLod = 0.0f,
		.borderColor = vk::BorderColor::eFloatOpaqueWhite,
		.unnormalizedCoordinates = vk::False};
	shadowMapSampler =
		vk::raii::Sampler(vulkanDevice->getLogicalDevice(), samplerInfo);
}

void VulkanRenderer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
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

void VulkanRenderer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
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

// One-time-submit command buffer for load-time upload operations.
std::unique_ptr<vk::raii::CommandBuffer> VulkanRenderer::beginSingleTimeCommands()
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

// Ends, submits, and blocks until the GPU has finished executing the commands.
void VulkanRenderer::endSingleTimeCommands(vk::raii::CommandBuffer &commandBuffer)
{
	commandBuffer.end();

	vk::SubmitInfo submitInfo{.commandBufferCount = 1,
							  .pCommandBuffers = &*commandBuffer};
	vulkanDevice->getGraphicsQueue().submit(submitInfo, nullptr);
	vulkanDevice->getGraphicsQueue().waitIdle();
}

void VulkanRenderer::copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
								vk::DeviceSize size)
{
	auto commandCopyBuffer = beginSingleTimeCommands();
	commandCopyBuffer->copyBuffer(srcBuffer, dstBuffer,
								  vk::BufferCopy(0, 0, size));
	endSingleTimeCommands(*commandCopyBuffer);
}

void VulkanRenderer::copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image,
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
	endSingleTimeCommands(*commandBuffer);
}

// Load-time image layout transition (allocates its own one-time command buffer).
void VulkanRenderer::transitionImageLayout(const vk::raii::Image &image,
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

void VulkanRenderer::createTextureImage()
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
void VulkanRenderer::generateMipmaps(vk::raii::Image &image, vk::Format imageFormat,
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
	// Last mip level was only ever a blit destination, so the loop never transitioned it.
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

void VulkanRenderer::createTextureImageView()
{
	textureImageView =
		vulkanDevice->createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb,
									  vk::ImageAspectFlagBits::eColor, mipLevels);
}

void VulkanRenderer::createTextureSampler()
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

void VulkanRenderer::uploadRenderable(Renderable &r, const std::vector<Vertex> &verts,
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

void VulkanRenderer::loadScene()
{
	std::ifstream f(SCENE_PATH);
	if (!f.is_open())
		throw std::runtime_error("failed to open scene file: " + SCENE_PATH);

	nlohmann::json scene = nlohmann::json::parse(f);

	for (const auto &obj : scene["objects"])
	{
		std::string mesh = obj["mesh"];
		glm::vec3 pos = {obj["position"][0], obj["position"][1], obj["position"][2]};
		glm::vec3 color = obj.contains("color")
							  ? glm::vec3{obj["color"][0], obj["color"][1], obj["color"][2]}
							  : glm::vec3{1.0f, 1.0f, 1.0f};
		float size = obj.value("size", 1.0f);

		auto [verts, idxs] = (mesh == "cube") ? makeCube(color, size) : makePlane(color, size);

		Renderable r;
		r.modelMatrix = glm::translate(glm::mat4(1.0f), pos);
		uploadRenderable(r, verts, idxs);
		renderables.push_back(std::move(r));
	}
}

// Host-visible, persistently mapped — CPU writes UBO data each frame via the
// mapped pointer, no explicit map/unmap needed.
void VulkanRenderer::createUniformBuffers()
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

void VulkanRenderer::createDescriptorPool()
{
	std::array poolSize{
		vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
							   MAX_FRAMES_IN_FLIGHT),
		vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,
							   MAX_FRAMES_IN_FLIGHT * 2)}; // binding 1 (texture) + binding 2 (shadow map)
	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = MAX_FRAMES_IN_FLIGHT,
		.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
		.pPoolSizes = poolSize.data()};
	descriptorPool = vk::raii::DescriptorPool(vulkanDevice->getLogicalDevice(), poolInfo);
}

void VulkanRenderer::createDescriptorSets()
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
		vk::DescriptorImageInfo shadowMapInfo{
			.sampler = shadowMapSampler,
			.imageView = shadowMapImageView,
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
								   .pImageInfo = &imageInfo},
			vk::WriteDescriptorSet{.dstSet = descriptorSets[i],
								   .dstBinding = 2,
								   .dstArrayElement = 0,
								   .descriptorCount = 1,
								   .descriptorType =
									   vk::DescriptorType::eCombinedImageSampler,
								   .pImageInfo = &shadowMapInfo}};
		vulkanDevice->getLogicalDevice().updateDescriptorSets(descriptorWrites, {});
	}
}

// presentComplete: signals when the display is done reading the swapchain image.
// renderFinished: signals when rendering is done and the image can be presented.
// inFlightFence: CPU blocks here to avoid overwriting resources still in use.
void VulkanRenderer::createSyncObjects()
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

void VulkanRenderer::mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		drawFrame();
	}

	vulkanDevice->getLogicalDevice().waitIdle();
}

void VulkanRenderer::drawFrame()
{
	if (pendingMsaaSamples != msaaSamples)
	{
		msaaSamples = pendingMsaaSamples;
		rebuildMsaa();
	}

	if (pendingPresentMode != swapchain->getPresentMode())
		recreateSwapChain();

	auto fenceResult =
		vulkanDevice->getLogicalDevice().waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
	if (fenceResult != vk::Result::eSuccess)
	{
		throw std::runtime_error("failed to wait for fence!");
	}

	auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(
		UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

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

	recordCommandBuffer(imageIndex);

	vk::PipelineStageFlags waitDestinationStageMask(
		vk::PipelineStageFlagBits::eColorAttachmentOutput);

	updateUniformBuffer(frameIndex);

	const vk::SubmitInfo submitInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
		.pWaitDstStageMask = &waitDestinationStageMask,
		.commandBufferCount = 1,
		.pCommandBuffers = &*commandBuffers[frameIndex],
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]};

	vulkanDevice->getGraphicsQueue().submit(submitInfo, *inFlightFences[frameIndex]);

	const vk::PresentInfoKHR presentInfoKHR{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
		.swapchainCount = 1,
		.pSwapchains = &*swapchain->getSwapChain(),
		.pImageIndices = &imageIndex};

	result = vulkanDevice->getGraphicsQueue().presentKHR(presentInfoKHR);

	if ((result == vk::Result::eSuboptimalKHR) ||
		(result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
	{
		framebufferResized = false;
		recreateSwapChain();
	}
	else
	{
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

	frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanRenderer::updateUniformBuffer(uint32_t currentImage)
{
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(
					 currentTime - startTime)
					 .count();

	UniformBufferObject ubo{};
	ubo.view = camera.getViewMatrix();
	ubo.proj = camera.getProjectionMatrix(
		static_cast<float>(swapchain->getExtent().width) /
		static_cast<float>(swapchain->getExtent().height));

	float dt = time - prevTime;
	prevTime = time;
	if (lightOrbit)
	{
		lightAngle += (dt * 0.5f) * glm::two_pi<float>() / 10.0f; // one orbit every 10 seconds
		lightAngle = lightAngle > glm::two_pi<float>() ? lightAngle - glm::two_pi<float>() : lightAngle;
	}

	glm::vec3 lightPos = glm::vec3(
		glm::cos(lightAngle) * 8.0f,
		glm::sin(lightAngle) * 8.0f,
		6.0f);
	glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 lightProj = glm::ortho(-10.f, 10.f, -10.f, 10.f, 0.1f, 50.f);
	ubo.lightSpaceMatrix = lightProj * lightView;
	ubo.lightDir = glm::vec4(glm::normalize(lightPos), 0.0f);
	ubo.cameraPos = glm::vec4(camera.position, 0.0f);
	ubo.materialParams = glm::vec4(ambient, specularStrength, shininess, 0.0f);

	memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

// Shadow pass → image layout transitions → main color pass → ImGui pass → present transition.
void VulkanRenderer::recordCommandBuffer(uint32_t imageIndex)
{
	auto &commandBuffer = commandBuffers[frameIndex];
	commandBuffer.begin({});

	// ---- Shadow pass ----
	// Transition shadow map to depth attachment for writing
	transition_image_layout(*shadowMapImage,
							vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
							vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
							vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
							vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
							vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
							vk::ImageAspectFlagBits::eDepth);

	vk::ClearValue shadowClear = vk::ClearDepthStencilValue(1.0f, 0);
	vk::RenderingAttachmentInfo shadowDepthAttachment{
		.imageView = *shadowMapImageView,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore, // must store — read by main pass
		.clearValue = shadowClear};
	vk::RenderingInfo shadowRenderingInfo{
		.renderArea = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}},
		.layerCount = 1,
		.pDepthAttachment = &shadowDepthAttachment};

	commandBuffer.beginRendering(shadowRenderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowPipeline);
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
	commandBuffer.endRendering();

	// Barrier: shadow map depth attachment → shader read for main pass
	transition_image_layout(*shadowMapImage,
							vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
							vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
							vk::AccessFlagBits2::eShaderRead,
							vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
							vk::PipelineStageFlagBits2::eFragmentShader,
							vk::ImageAspectFlagBits::eDepth);
	// ---- End shadow pass ----

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

	vk::RenderingInfo renderingInfo{
		.renderArea = {vk::Offset2D{0, 0}, swapchain->getExtent()},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo};

	commandBuffer.beginRendering(renderingInfo);
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

	commandBuffer.endRendering();

	// Barrier before ImGui pass so it sees finished color writes.
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

	// ImGui pass: renders directly into the swapchain image, loading rather
	// than clearing to preserve the scene rendered above.
	vk::RenderingAttachmentInfo imguiColorAttachment{
		.imageView = *swapchain->getImageViews()[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eLoad,
		.storeOp = vk::AttachmentStoreOp::eStore};

	vk::RenderingInfo imguiRenderingInfo{
		.renderArea = {vk::Offset2D{0, 0}, swapchain->getExtent()},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &imguiColorAttachment};

	commandBuffer.beginRendering(imguiRenderingInfo);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);
	commandBuffer.endRendering();

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

// In-frame layout transition — records into the current frame's command buffer
// rather than allocating its own (cf. transitionImageLayout).
void VulkanRenderer::transition_image_layout(vk::Image image, vk::ImageLayout oldLayout,
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

void VulkanRenderer::recreateSwapChain()
{
	vulkanDevice->getLogicalDevice().waitIdle();
	cleanupSwapChain();
	swapchain->recreate(window, pendingPresentMode);
	ImGui_ImplVulkan_SetMinImageCount(
		static_cast<uint32_t>(swapchain->getImages().size()));
	createDepthResources();
	createColorResources();
}

// Must be called with no frames in flight (GPU idle).
void VulkanRenderer::rebuildMsaa()
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

// Destroys resources tied to the swapchain extent. The swapchain itself is
// managed by the Swapchain object.
void VulkanRenderer::cleanupSwapChain()
{
	colorImageView = nullptr;
	colorImage = nullptr;
	colorImageMemory = nullptr;
	depthImageView = nullptr;
	depthImage = nullptr;
	depthImageMemory = nullptr;
}

// RAII handles call vkDestroy automatically when set to nullptr, but order
// matters: child objects must be destroyed before their parents.
void VulkanRenderer::cleanup()
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
	shadowPipeline = nullptr;
	graphicsPipeline = nullptr;
	pipelineLayout = nullptr;
	shadowMapSampler = nullptr;
	shadowMapImageView = nullptr;
	shadowMapImage = nullptr;
	shadowMapImageMemory = nullptr;
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

void VulkanRenderer::drawImGui()
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

	ImGui::Separator();

	// Light controls
	ImGui::Checkbox("Orbit light", &lightOrbit);
	if (!lightOrbit)
		ImGui::SliderAngle("Light angle", &lightAngle, 0.0f, 360.0f);

	ImGui::Separator();

	// Camera position (always looks at origin)
	camera.drawImGui();

	ImGui::Separator();

	// Material / lighting parameters
	ImGui::SliderFloat("Ambient", &ambient, 0.0f, 1.0f);
	ImGui::SliderFloat("Specular strength", &specularStrength, 0.0f, 1.0f);
	ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);

	ImGui::End();

	ImGui::Render();
}

void VulkanRenderer::imGuiInit()
{
	// imgui 1.92+ requires at least IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE
	// descriptors per font atlas; allocating fewer causes silent allocation failures.
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
