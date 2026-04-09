#include "VulkanRenderer.hpp"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>

#include <nlohmann/json.hpp>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
const std::string SCENE_PATH = "scenes/scene.json";

vk::VertexInputBindingDescription VulkanRenderer::getVertexBindingDescription()
{
	return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
}

std::array<vk::VertexInputAttributeDescription, 5> VulkanRenderer::getVertexAttributeDescriptions()
{
	return {
		vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat,  offsetof(Vertex, pos)),
		vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat,  offsetof(Vertex, color)),
		vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat,     offsetof(Vertex, texCoord)),
		vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat,  offsetof(Vertex, normal)),
		vk::VertexInputAttributeDescription(4, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent))};
}

void VulkanRenderer::run()
{
	initWindow();
	initVulkan();
	imGuiInit();
	try
	{
		mainLoop();
	}
	catch (...)
	{
		// Ensure Vulkan objects are destroyed in the correct order even when
		// mainLoop exits via an exception, before the exception propagates.
		cleanup();
		throw;
	}
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
	// Sync pendingPresentMode to the mode the swapchain actually chose.
	// If the hardware/compositor doesn't support the preferred mode, the
	// swapchain falls back silently. Without this, drawFrame would call
	// recreateSwapChain every single frame.
	pendingPresentMode = swapchain->getPresentMode();

	// --- Descriptor layouts (must exist before any pipeline is created) ---
	createDescriptorSetLayout();
	createSkyDescriptorSetLayout();

	// --- Pipelines ---
	createGraphicsPipeline();
	createShadowPipeline();
	createSkyPipeline();

	// --- Command infrastructure ---
	createCommandPool();

	// --- Render attachments ---
	createColorResources();
	createDepthResources();
	createShadowMapResources();
	createShadowMapSampler();

	// --- Scene and GPU data ---
	loadScene();
	createUniformBuffers();

	// --- Descriptors ---
	createDescriptorPool();
	createDescriptorSets();
	createSkyDescriptorSets();   // must come after createUniformBuffers

	// --- Synchronisation ---
	createCommandBuffers();
	createSyncObjects();
}

void VulkanRenderer::createDescriptorSetLayout()
{
	std::array bindings = {
		vk::DescriptorSetLayoutBinding(
			0, vk::DescriptorType::eUniformBuffer, 1,
			vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
		// Binding 1: bindless texture array. PARTIALLY_BOUND means we only
		// need to fill slots we actually use — the rest can stay empty.
		vk::DescriptorSetLayoutBinding(
			1, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES,
			vk::ShaderStageFlagBits::eFragment, nullptr),
		vk::DescriptorSetLayoutBinding(
			2, vk::DescriptorType::eCombinedImageSampler, 1,
			vk::ShaderStageFlagBits::eFragment, nullptr)};

	std::array<vk::DescriptorBindingFlags, 3> bindingFlags = {
		vk::DescriptorBindingFlags{},                          // binding 0: UBO, no special flags
		vk::DescriptorBindingFlagBits::ePartiallyBound,        // binding 1: texture array
		vk::DescriptorBindingFlags{}};                         // binding 2: shadow map

	vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
		.bindingCount = static_cast<uint32_t>(bindingFlags.size()),
		.pBindingFlags = bindingFlags.data()};

	vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.pNext = &bindingFlagsInfo,
		.bindingCount = static_cast<uint32_t>(bindings.size()),
		.pBindings = bindings.data()};
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
		.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		.offset = 0,
		.size = sizeof(PushConstants)};
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

// ─── Sky pipeline ──────────────────────────────────────────────────────────

void VulkanRenderer::createSkyDescriptorSetLayout()
{
	// Sky only needs the UBO (binding 0) to access invProj / invViewRot / lightDir.
	vk::DescriptorSetLayoutBinding uboBinding{
		.binding         = 0,
		.descriptorType  = vk::DescriptorType::eUniformBuffer,
		.descriptorCount = 1,
		.stageFlags      = vk::ShaderStageFlagBits::eFragment};
	vk::DescriptorSetLayoutCreateInfo info{.bindingCount = 1, .pBindings = &uboBinding};
	skyDescriptorSetLayout = vk::raii::DescriptorSetLayout(vulkanDevice->getLogicalDevice(), info);
}

void VulkanRenderer::createSkyPipeline()
{
	vk::raii::ShaderModule skyShader = createShaderModule(readFile("shaders/sky.spv"));
	vk::PipelineShaderStageCreateInfo stages[2] = {
		{.stage = vk::ShaderStageFlagBits::eVertex,   .module = skyShader, .pName = "skyVert"},
		{.stage = vk::ShaderStageFlagBits::eFragment, .module = skyShader, .pName = "skyFrag"}};

	// Sky push constants: SkyPushConstants (64 bytes), fragment stage only.
	vk::PushConstantRange pushRange{
		.stageFlags = vk::ShaderStageFlagBits::eFragment,
		.offset     = 0,
		.size       = sizeof(SkyPushConstants)};
	vk::PipelineLayoutCreateInfo layoutInfo{
		.setLayoutCount         = 1,
		.pSetLayouts            = &*skyDescriptorSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &pushRange};
	skyPipelineLayout = vk::raii::PipelineLayout(vulkanDevice->getLogicalDevice(), layoutInfo);

	// No vertex input — fullscreen triangle is generated from SV_VertexID.
	vk::PipelineVertexInputStateCreateInfo vertexInput{};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		.topology = vk::PrimitiveTopology::eTriangleList};

	std::vector<vk::DynamicState> dynStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynState{
		.dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
		.pDynamicStates    = dynStates.data()};
	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode    = vk::CullModeFlagBits::eNone,   // fullscreen tri, no culling needed
		.lineWidth   = 1.0f};
	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = msaaSamples};

	vk::PipelineColorBlendAttachmentState colorBlendAtt{
		.blendEnable    = vk::False,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.attachmentCount = 1, .pAttachments = &colorBlendAtt};

	// Depth: test against scene geometry (sky at depth 1.0 passes where nothing was drawn),
	// but don't write depth so scene geometry isn't overwritten.
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable  = vk::True,
		.depthWriteEnable = vk::False,
		.depthCompareOp   = vk::CompareOp::eLessOrEqual};

	vk::Format depthFormat = vulkanDevice->findDepthFormat();
	vk::PipelineRenderingCreateInfo renderingInfo{
		.colorAttachmentCount    = 1,
		.pColorAttachmentFormats = &swapchain->getSurfaceFormat().format,
		.depthAttachmentFormat   = depthFormat};

	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.pNext               = &renderingInfo,
		.stageCount          = 2,
		.pStages             = stages,
		.pVertexInputState   = &vertexInput,
		.pInputAssemblyState = &inputAssembly,
		.pViewportState      = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState   = &multisampling,
		.pDepthStencilState  = &depthStencil,
		.pColorBlendState    = &colorBlending,
		.pDynamicState       = &dynState,
		.layout              = skyPipelineLayout};
	skyPipeline = vk::raii::Pipeline(vulkanDevice->getLogicalDevice(), nullptr, pipelineInfo);
}

void VulkanRenderer::createSkyDescriptorSets()
{
	vk::DescriptorPoolSize uboPoolSize{
		.type            = vk::DescriptorType::eUniformBuffer,
		.descriptorCount = MAX_FRAMES_IN_FLIGHT};
	vk::DescriptorPoolCreateInfo poolInfo{
		.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets       = MAX_FRAMES_IN_FLIGHT,
		.poolSizeCount = 1,
		.pPoolSizes    = &uboPoolSize};
	skyDescriptorPool = vk::raii::DescriptorPool(vulkanDevice->getLogicalDevice(), poolInfo);

	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *skyDescriptorSetLayout);
	skyDescriptorSets = vulkanDevice->getLogicalDevice().allocateDescriptorSets(
		vk::DescriptorSetAllocateInfo{
			.descriptorPool     = skyDescriptorPool,
			.descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
			.pSetLayouts        = layouts.data()});

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorBufferInfo buf{uniformBuffers[i], 0, sizeof(UniformBufferObject)};
		vulkanDevice->getLogicalDevice().updateDescriptorSets(
			vk::WriteDescriptorSet{
				.dstSet          = skyDescriptorSets[i],
				.dstBinding      = 0,
				.descriptorCount = 1,
				.descriptorType  = vk::DescriptorType::eUniformBuffer,
				.pBufferInfo     = &buf},
			{});
	}
}

// ─── Utility helpers ───────────────────────────────────────────────────────

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
		.addressModeU = vk::SamplerAddressMode::eClampToBorder,
		.addressModeV = vk::SamplerAddressMode::eClampToBorder,
		.addressModeW = vk::SamplerAddressMode::eClampToBorder,
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

	// Use locals so that if allocateMemory throws, the new VkImage is destroyed
	// by its local RAII destructor and the output params are left unchanged.
	vk::raii::Image newImage(vulkanDevice->getLogicalDevice(), imageInfo);
	vk::MemoryRequirements memRequirements = newImage.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex =
			vulkanDevice->findMemoryType(memRequirements.memoryTypeBits, properties)};
	vk::raii::DeviceMemory newMemory(vulkanDevice->getLogicalDevice(), allocInfo);
	newImage.bindMemory(newMemory, 0);

	image = std::move(newImage);
	imageMemory = std::move(newMemory);
}

void VulkanRenderer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
								  vk::MemoryPropertyFlags properties,
								  vk::raii::Buffer &buffer,
								  vk::raii::DeviceMemory &bufferMemory)
{
	vk::BufferCreateInfo bufferInfo{.size = size,
									.usage = usage,
									.sharingMode = vk::SharingMode::eExclusive};
	vk::raii::Buffer newBuffer(vulkanDevice->getLogicalDevice(), bufferInfo);
	vk::MemoryRequirements memRequirements = newBuffer.getMemoryRequirements();
	vk::MemoryAllocateInfo memoryAllocateInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex =
			vulkanDevice->findMemoryType(memRequirements.memoryTypeBits, properties)};
	vk::raii::DeviceMemory newMemory(vulkanDevice->getLogicalDevice(), memoryAllocateInfo);
	newBuffer.bindMemory(*newMemory, 0);

	buffer = std::move(newBuffer);
	bufferMemory = std::move(newMemory);
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
		.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0,
							 1}};

	vk::DependencyInfo dependencyInfo{.imageMemoryBarrierCount = 1,
									  .pImageMemoryBarriers = &barrier};
	commandBuffer->pipelineBarrier2(dependencyInfo);

	endSingleTimeCommands(*commandBuffer);
}

// Loads a texture from disk, uploads it to the GPU, creates a view and sampler,
// appends it to the global textures array, and returns its index. Returns the
// cached index immediately if the same path was already loaded.
uint32_t VulkanRenderer::loadTexture(const std::string &path)
{
	auto it = textureCache.find(path);
	if (it != textureCache.end())
		return it->second;

	int texWidth, texHeight, texChannels;
	stbi_uc *pixels = stbi_load(path.c_str(), &texWidth, &texHeight,
								&texChannels, STBI_rgb_alpha);
	if (!pixels)
		throw std::runtime_error("failed to load texture: " + path);

	uint32_t mipLevels = static_cast<uint32_t>(
		std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
	vk::DeviceSize imageSize = texWidth * texHeight * 4;

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

	Texture tex;
	createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
				vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eTransferSrc |
					vk::ImageUsageFlagBits::eTransferDst |
					vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal, tex.image, tex.memory);
	transitionImageLayout(tex.image, vk::ImageLayout::eUndefined,
						  vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(stagingBuffer, tex.image,
					  static_cast<uint32_t>(texWidth),
					  static_cast<uint32_t>(texHeight));
	generateMipmaps(tex.image, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);

	tex.view = vulkanDevice->createImageView(*tex.image, vk::Format::eR8G8B8A8Srgb,
											 vk::ImageAspectFlagBits::eColor, mipLevels);

	vk::PhysicalDeviceProperties props = vulkanDevice->getPhysicalDevice().getProperties();
	vk::SamplerCreateInfo samplerInfo{
		.magFilter               = vk::Filter::eLinear,
		.minFilter               = vk::Filter::eLinear,
		.mipmapMode              = vk::SamplerMipmapMode::eLinear,
		.addressModeU            = vk::SamplerAddressMode::eRepeat,
		.addressModeV            = vk::SamplerAddressMode::eRepeat,
		.addressModeW            = vk::SamplerAddressMode::eRepeat,
		.anisotropyEnable        = vk::True,
		.maxAnisotropy           = props.limits.maxSamplerAnisotropy,
		.compareOp               = vk::CompareOp::eAlways,
		.maxLod                  = vk::LodClampNone,
		.borderColor             = vk::BorderColor::eIntOpaqueBlack,
		.unnormalizedCoordinates = vk::False};
	tex.sampler = vk::raii::Sampler(vulkanDevice->getLogicalDevice(), samplerInfo);

	uint32_t index = static_cast<uint32_t>(textures.size());
	textures.push_back(std::move(tex));
	textureCache[path] = index;
	return index;
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

// Rebuilds a model matrix from the decomposed transform stored in Renderable.
// Called once at load time and again whenever ImGui sliders change a value.
static glm::mat4 buildModelMatrix(glm::vec3 pos, glm::vec3 rotDeg, float scale)
{
	glm::mat4 m = glm::translate(glm::mat4(1.0f), pos);
	m = glm::rotate(m, glm::radians(rotDeg.x), glm::vec3(1, 0, 0));
	m = glm::rotate(m, glm::radians(rotDeg.y), glm::vec3(0, 1, 0));
	m = glm::rotate(m, glm::radians(rotDeg.z), glm::vec3(0, 0, 1));
	return glm::scale(m, glm::vec3(scale));
}

void VulkanRenderer::loadScene()
{
	std::ifstream f(SCENE_PATH);
	if (!f.is_open())
		throw std::runtime_error("failed to open scene file: " + SCENE_PATH);

	nlohmann::json scene = nlohmann::json::parse(f);

	// Optional skybox settings
	if (scene.contains("skybox"))
	{
		const auto &s = scene["skybox"];
		if (s.contains("horizonColor"))
			skyPush.horizonColor = {s["horizonColor"][0], s["horizonColor"][1], s["horizonColor"][2], 1.0f};
		if (s.contains("zenithColor"))
			skyPush.zenithColor  = {s["zenithColor"][0],  s["zenithColor"][1],  s["zenithColor"][2],  1.0f};
		if (s.contains("groundColor"))
			skyPush.groundColor  = {s["groundColor"][0],  s["groundColor"][1],  s["groundColor"][2],  1.0f};
		if (s.contains("sunColor"))
			skyPush.sunParams    = {s["sunColor"][0], s["sunColor"][1], s["sunColor"][2],
			                        s.value("sunSize", 0.9998f)};
	}

	// Optional point lights (overrides the compiled-in defaults when present)
	if (scene.contains("pointLights"))
	{
		pointLights.clear();
		for (const auto &pl : scene["pointLights"])
		{
			PointLightData d;
			if (pl.contains("position"))
				d.position  = {pl["position"][0], pl["position"][1], pl["position"][2]};
			if (pl.contains("color"))
				d.color     = {pl["color"][0], pl["color"][1], pl["color"][2]};
			d.intensity     = pl.value("intensity", 3.0f);
			d.radius        = pl.value("radius",    8.0f);
			d.enabled       = pl.value("enabled",   true);
			pointLights.push_back(d);
		}
	}

	for (const auto &obj : scene["objects"])
	{
		std::string mesh = obj["mesh"];
		glm::vec3 pos = {obj["position"][0], obj["position"][1], obj["position"][2]};
		glm::vec3 color = obj.contains("color")
							  ? glm::vec3{obj["color"][0], obj["color"][1], obj["color"][2]}
							  : glm::vec3{1.0f, 1.0f, 1.0f};
		float size = obj.value("size", 1.0f);
		std::optional<std::string> texturePath = obj.contains("texture")
			? std::optional<std::string>{obj["texture"].get<std::string>()}
			: std::nullopt;
		std::optional<std::string> specularMapPath = obj.contains("specularMap")
			? std::optional<std::string>{obj["specularMap"].get<std::string>()}
			: std::nullopt;
		std::optional<std::string> normalMapPath = obj.contains("normalMap")
			? std::optional<std::string>{obj["normalMap"].get<std::string>()}
			: std::nullopt;
		std::optional<std::string> heightMapPath = obj.contains("heightMap")
			? std::optional<std::string>{obj["heightMap"].get<std::string>()}
			: std::nullopt;

		// Optional rotation (XYZ Euler angles, degrees) and uniform scale.
		glm::vec3 rotDeg = obj.contains("rotation")
							   ? glm::vec3{obj["rotation"][0], obj["rotation"][1], obj["rotation"][2]}
							   : glm::vec3{0.0f};
		float scale = obj.value("scale", 1.0f);

		std::pair<std::vector<Vertex>, std::vector<uint32_t>> meshData;
		if (mesh == "cube")
			meshData = makeCube(color, size);
		else if (mesh == "plane")
			meshData = makePlane(color, size);
		else
			meshData = loadOBJ(mesh);

		Renderable r;
		r.label       = mesh;
		r.position    = pos;
		r.rotationDeg = rotDeg;
		r.scale       = scale;
		r.modelMatrix = buildModelMatrix(pos, rotDeg, scale);
		r.textureIndex     = texturePath     ? loadTexture(*texturePath)     : 0xFFFFu;
		r.specularMapIndex = specularMapPath ? loadTexture(*specularMapPath) : 0xFFFFu;
		r.normalMapIndex   = normalMapPath   ? loadTexture(*normalMapPath)   : 0xFFFFu;
		r.heightMapIndex   = heightMapPath   ? loadTexture(*heightMapPath)   : 0xFFFFu;
		uploadRenderable(r, meshData.first, meshData.second);
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
	// Binding 1 holds up to MAX_TEXTURES samplers per set.
	// Binding 2 holds 1 sampler (shadow map) per set.
	std::array poolSize{
		vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
							   MAX_FRAMES_IN_FLIGHT),
		vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,
							   MAX_FRAMES_IN_FLIGHT * (MAX_TEXTURES + 1))};
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

	// Build one image-info entry per loaded texture. This vector is the same
	// for every frame-in-flight since the texture data doesn't change.
	std::vector<vk::DescriptorImageInfo> textureInfos;
	for (const auto &tex : textures)
	{
		textureInfos.push_back({
			.sampler     = *tex.sampler,
			.imageView   = *tex.view,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal});
	}

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorBufferInfo bufferInfo{.buffer = uniformBuffers[i],
											.offset = 0,
											.range = sizeof(UniformBufferObject)};
		vk::DescriptorImageInfo shadowMapInfo{
			.sampler     = *shadowMapSampler,
			.imageView   = *shadowMapImageView,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

		std::vector<vk::WriteDescriptorSet> descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet          = descriptorSets[i],
				.dstBinding      = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType  = vk::DescriptorType::eUniformBuffer,
				.pBufferInfo     = &bufferInfo},
			vk::WriteDescriptorSet{
				.dstSet          = descriptorSets[i],
				.dstBinding      = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType  = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo      = &shadowMapInfo}};
		if (!textureInfos.empty())
		{
			descriptorWrites.push_back(vk::WriteDescriptorSet{
				.dstSet          = descriptorSets[i],
				.dstBinding      = 1,
				.dstArrayElement = 0,
				.descriptorCount = static_cast<uint32_t>(textureInfos.size()),
				.descriptorType  = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo      = textureInfos.data()});
		}
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
	auto lastTime = std::chrono::steady_clock::now();

	// Prime last mouse position so the first frame has zero delta.
	double mx0, my0;
	glfwGetCursorPos(window, &mx0, &my0);
	lastMousePos = {static_cast<float>(mx0), static_cast<float>(my0)};

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		// --- Delta time --------------------------------------------------
		auto now = std::chrono::steady_clock::now();
		float dt = std::chrono::duration<float>(now - lastTime).count();
		lastTime = now;

		// --- Camera fly mode (hold right mouse button) -------------------
		double mx, my;
		glfwGetCursorPos(window, &mx, &my);

		bool rightHeld    = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
		bool imguiWantsMouse = ImGui::GetIO().WantCaptureMouse;

		if (rightHeld && !imguiWantsMouse)
		{
			if (!cameraMode)
			{
				// First frame of camera mode: hide cursor and reset delta to
				// avoid a sudden jump from wherever the cursor was.
				cameraMode   = true;
				lastMousePos = {static_cast<float>(mx), static_cast<float>(my)};
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			glm::vec2 delta = {static_cast<float>(mx) - lastMousePos.x,
							   static_cast<float>(my) - lastMousePos.y};
			camera.processInput(window, dt, delta);
		}
		else if (cameraMode)
		{
			cameraMode = false;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

		lastMousePos = {static_cast<float>(mx), static_cast<float>(my)};
		// -----------------------------------------------------------------

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
	ubo.materialParams = glm::vec4(ambient, specularStrength, shininess, tonemapping ? exposure : 0.0f);

	int activeLights = 0;
	for (const auto &pl : pointLights)
	{
		if (!pl.enabled || activeLights >= MAX_POINT_LIGHTS) continue;
		ubo.pointLightPos[activeLights]   = glm::vec4(pl.position, pl.intensity);
		ubo.pointLightColor[activeLights] = glm::vec4(pl.color,    pl.radius);
		activeLights++;
	}
	ubo.lightCounts  = glm::vec4(static_cast<float>(activeLights), 0.0f, 0.0f, 0.0f);
	ubo.shadowParams = glm::vec4(shadowBiasMin, shadowBiasMax, 0.0f, 0.0f);
	ubo.pomParams    = glm::vec4(pomDepthScale, pomMinSteps, pomMaxSteps, 0.0f);

	// Sky ray reconstruction matrices.
	ubo.invProj    = glm::inverse(ubo.proj);
	// Rotation-only view: zero the translation column then invert via transpose
	// (valid because the rotation part of an orthogonal matrix satisfies R^-1 = R^T).
	glm::mat4 viewRot = ubo.view;
	viewRot[3]         = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	ubo.invViewRot     = glm::transpose(viewRot);

	memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

// Shadow pass → image layout transitions → main color pass → ImGui pass → present transition.
void VulkanRenderer::recordCommandBuffer(uint32_t imageIndex)
{
	auto &commandBuffer = commandBuffers[frameIndex];
	commandBuffer.begin({});

	// ---- Shadow pass ----
	// Transition shadow map to depth attachment for writing
	recordImageBarrier(*shadowMapImage,
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
		PushConstants pc{r.modelMatrix, r.textureIndex, r.specularMapIndex, r.normalMapIndex, r.heightMapIndex};
		commandBuffer.pushConstants2(
			vk::PushConstantsInfo{}
				.setLayout(*pipelineLayout)
				.setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
				.setOffset(0)
				.setSize(sizeof(PushConstants))
				.setPValues(&pc));
		commandBuffer.bindVertexBuffers(0, *r.vertexBuffer, {0});
		commandBuffer.bindIndexBuffer(*r.indexBuffer, 0, vk::IndexType::eUint32);
		commandBuffer.drawIndexed(r.indexCount, 1, 0, 0, 0);
	}
	commandBuffer.endRendering();

	// Barrier: shadow map depth attachment → shader read for main pass
	recordImageBarrier(*shadowMapImage,
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
		recordImageBarrier(
			*colorImage, vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal, {},		// srcAccessMask
			vk::AccessFlagBits2::eColorAttachmentWrite,			// dstAccessMask
			vk::PipelineStageFlagBits2::eTopOfPipe,				// srcStageMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStageMask
			vk::ImageAspectFlagBits::eColor);
	}

	// Transition swapchain image to color attachment optimal (resolve target,
	// or direct render target when no MSAA)
	recordImageBarrier(
		swapchain->getImages()[imageIndex], vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal, {},		// srcAccessMask
		vk::AccessFlagBits2::eColorAttachmentWrite,			// dstAccessMask
		vk::PipelineStageFlagBits2::eTopOfPipe,				// srcStageMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStageMask
		vk::ImageAspectFlagBits::eColor);

	recordImageBarrier(*depthImage, vk::ImageLayout::eUndefined,
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
		PushConstants pc{r.modelMatrix, r.textureIndex, r.specularMapIndex, r.normalMapIndex, r.heightMapIndex};
		commandBuffer.pushConstants2(
			vk::PushConstantsInfo{}
				.setLayout(*pipelineLayout)
				.setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
				.setOffset(0)
				.setSize(sizeof(PushConstants))
				.setPValues(&pc));
		commandBuffer.bindVertexBuffers(0, *r.vertexBuffer, {0});
		commandBuffer.bindIndexBuffer(*r.indexBuffer, 0, vk::IndexType::eUint32);
		commandBuffer.drawIndexed(r.indexCount, 1, 0, 0, 0);
	}

	// ---- Sky pass (within the same render pass, after scene geometry) ----
	// The sky fullscreen triangle sits at depth 1.0. With eLessOrEqual + no depth write,
	// it only fills pixels where the depth buffer still holds the clear value of 1.0
	// (i.e., where no geometry was drawn). This means no extra barrier is needed between
	// the scene and sky draws.
	if (skyEnabled)
	{
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *skyPipeline);
		commandBuffer.setViewport(
			0, vk::Viewport{0.0f, 0.0f,
			                static_cast<float>(swapchain->getExtent().width),
			                static_cast<float>(swapchain->getExtent().height), 0.0f, 1.0f});
		commandBuffer.setScissor(0, vk::Rect2D{{0, 0}, swapchain->getExtent()});
		commandBuffers[frameIndex].bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, skyPipelineLayout, 0,
			*skyDescriptorSets[frameIndex], nullptr);
		commandBuffer.pushConstants2(
			vk::PushConstantsInfo{}
				.setLayout(*skyPipelineLayout)
				.setStageFlags(vk::ShaderStageFlagBits::eFragment)
				.setOffset(0)
				.setSize(sizeof(SkyPushConstants))
				.setPValues(&skyPush));
		commandBuffer.draw(3, 1, 0, 0);   // fullscreen triangle, no vertex buffer
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

	recordImageBarrier(
		swapchain->getImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,			// srcAccessMask
		{},													// dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStageMask
		vk::PipelineStageFlagBits2::eBottomOfPipe,			// dstStageMask
		vk::ImageAspectFlagBits::eColor);

	commandBuffer.end();
}

// Records an image memory barrier into the current frame's command buffer.
// Unlike transitionImageLayout() (which allocates its own one-shot command buffer),
// this is called during recordCommandBuffer() for in-flight transitions.
void VulkanRenderer::recordImageBarrier(vk::Image image, vk::ImageLayout oldLayout,
										vk::ImageLayout newLayout,
										vk::AccessFlags2 srcAccessMask,
										vk::AccessFlags2 dstAccessMask,
										vk::PipelineStageFlags2 srcStageMask,
										vk::PipelineStageFlags2 dstStageMask,
										vk::ImageAspectFlags aspectFlags)
{
	vk::ImageMemoryBarrier2 barrier{
		.srcStageMask        = srcStageMask,
		.srcAccessMask       = srcAccessMask,
		.dstStageMask        = dstStageMask,
		.dstAccessMask       = dstAccessMask,
		.oldLayout           = oldLayout,
		.newLayout           = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image               = image,
		.subresourceRange    = {aspectFlags, 0, 1, 0, 1}};
	commandBuffers[frameIndex].pipelineBarrier2(
		vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
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
	skyPipeline      = nullptr;
	createDepthResources();
	createColorResources();
	createGraphicsPipeline();
	createSkyPipeline();
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
	skyDescriptorSets.clear();
	skyDescriptorPool      = nullptr;
	skyPipeline            = nullptr;
	skyPipelineLayout      = nullptr;
	skyDescriptorSetLayout = nullptr;
	shadowPipeline   = nullptr;
	graphicsPipeline = nullptr;
	pipelineLayout   = nullptr;
	shadowMapSampler = nullptr;
	shadowMapImageView = nullptr;
	shadowMapImage = nullptr;
	shadowMapImageMemory = nullptr;
	textures.clear();
	textureCache.clear();
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
	ImGui::Text("Textures loaded: %u", static_cast<uint32_t>(textures.size()));

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

	// Shadow bias sliders.
	// biasMin = bias when face is lit head-on (low acne risk, low peter-pan risk).
	// biasMax = bias at grazing angles (high acne risk without it, peter-pan if too high).
	ImGui::SliderFloat("Shadow bias min", &shadowBiasMin, 0.0f, 0.01f, "%.4f");
	ImGui::SliderFloat("Shadow bias max", &shadowBiasMax, 0.0f, 0.02f, "%.4f");

	ImGui::Separator();

	// POM
	if (ImGui::CollapsingHeader("Parallax (POM)"))
	{
		ImGui::SliderFloat("Depth scale", &pomDepthScale, 0.001f, 0.2f,  "%.3f");
		ImGui::SliderFloat("Min steps",   &pomMinSteps,   4.0f,   16.0f, "%.0f");
		ImGui::SliderFloat("Max steps",   &pomMaxSteps,   8.0f,   64.0f, "%.0f");
		ImGui::TextDisabled("Assign 'heightMap' in scene.json to activate POM per object.");
	}

	ImGui::Separator();

	// Camera
	if (ImGui::CollapsingHeader("Camera"))
		camera.drawImGui();

	ImGui::Separator();

	// Material / lighting parameters
	ImGui::Checkbox("Tonemapping (ACES)", &tonemapping);
	if (tonemapping)
		ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f);
	ImGui::SliderFloat("Ambient",          &ambient,          0.0f,  1.0f);
	ImGui::SliderFloat("Specular strength",&specularStrength, 0.0f,  1.0f);
	ImGui::SliderFloat("Shininess",        &shininess,        1.0f, 256.0f);

	ImGui::Separator();

	// Sky
	if (ImGui::CollapsingHeader("Sky"))
	{
		ImGui::Checkbox("Enabled##sky", &skyEnabled);
		if (skyEnabled)
		{
			ImGui::ColorEdit3("Horizon",  &skyPush.horizonColor.x);
			ImGui::ColorEdit3("Zenith",   &skyPush.zenithColor.x);
			ImGui::ColorEdit3("Ground",   &skyPush.groundColor.x);
			ImGui::ColorEdit3("Sun color",&skyPush.sunParams.x);
			// Sun size as degrees (more intuitive than cosine)
			float sunDeg = glm::degrees(std::acos(skyPush.sunParams.w));
			if (ImGui::SliderFloat("Sun size (deg)", &sunDeg, 0.1f, 10.0f))
				skyPush.sunParams.w = std::cos(glm::radians(sunDeg));
		}
	}

	ImGui::Separator();

	// Point lights
	if (ImGui::CollapsingHeader("Point Lights"))
	{
		bool atMax = static_cast<int>(pointLights.size()) >= MAX_POINT_LIGHTS;
		if (atMax) ImGui::BeginDisabled();
		if (ImGui::Button("Add Light"))
			pointLights.push_back(PointLightData{});
		if (atMax)
		{
			ImGui::EndDisabled();
			ImGui::SameLine();
			ImGui::TextDisabled("(max %d)", MAX_POINT_LIGHTS);
		}

		int removeIdx = -1;
		for (int i = 0; i < static_cast<int>(pointLights.size()); i++)
		{
			auto &pl = pointLights[i];
			ImGui::PushID(i);
			char label[32];
			snprintf(label, sizeof(label), "Light %d", i);
			if (ImGui::TreeNode(label))
			{
				ImGui::Checkbox("Enabled",      &pl.enabled);
				ImGui::DragFloat3("Position",   &pl.position.x,  0.05f);
				ImGui::ColorEdit3("Color",      &pl.color.x);
				ImGui::SliderFloat("Intensity", &pl.intensity,   0.0f, 20.0f);
				ImGui::SliderFloat("Radius",    &pl.radius,      0.5f, 50.0f);
				if (ImGui::Button("Remove"))
					removeIdx = i;
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		if (removeIdx >= 0)
			pointLights.erase(pointLights.begin() + removeIdx);
	}

	ImGui::Separator();

	// Per-object transform editors
	if (ImGui::CollapsingHeader("Objects"))
	{
		for (size_t i = 0; i < renderables.size(); i++)
		{
			auto &r = renderables[i];
			ImGui::PushID(static_cast<int>(i));
			// Show a tree node per object. Use the mesh label as display name.
			if (ImGui::TreeNode(r.label.c_str()))
			{
				bool changed = false;
				// DragFloat lets the user drag to adjust or ctrl+click to type a value.
				changed |= ImGui::DragFloat3("Position",   &r.position.x,    0.05f);
				changed |= ImGui::DragFloat3("Rotation",   &r.rotationDeg.x, 1.0f, -180.0f, 180.0f);
				changed |= ImGui::DragFloat ("Scale",      &r.scale,         0.01f,   0.01f, 100.0f);
				if (changed)
					r.modelMatrix = buildModelMatrix(r.position, r.rotationDeg, r.scale);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}

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
