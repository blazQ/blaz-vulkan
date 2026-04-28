#include "VulkanRenderer.hpp"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>

#include <nlohmann/json.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "Renderable.hpp"

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
const std::string  SCENE_PATH = "scenes/scene-descriptor.json";

// ─── Vertex format ─────────────────────────────────────────────────────────

vk::VertexInputBindingDescription VulkanRenderer::getVertexBindingDescription()
{
    return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
}

std::array<vk::VertexInputAttributeDescription, 5>
VulkanRenderer::getVertexAttributeDescriptions()
{
    return {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat,    offsetof(Vertex, pos)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat,    offsetof(Vertex, color)),
        vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat,       offsetof(Vertex, texCoord)),
        vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat,    offsetof(Vertex, normal)),
        vk::VertexInputAttributeDescription(4, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent))};
}

// ─── Top-level lifecycle ────────────────────────────────────────────────────

void VulkanRenderer::run()
{
    basePath_ = std::filesystem::path(ASSET_DIR);
    initWindow();
    initVulkan();
    imGuiInit();
    try
    {
        mainLoop();
    }
    catch (...)
    {
        cleanup();
        throw;
    }
    cleanup();
}

void VulkanRenderer::initWindow()
{
    if (getenv("ENABLE_VULKAN_RENDERDOC_CAPTURE"))
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "blaz-engine", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

// GLFW calls this from outside the render loop — just set a flag for drawFrame.
void VulkanRenderer::framebufferResizeCallback(GLFWwindow* window, int, int)
{
    auto app = reinterpret_cast<VulkanRenderer*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void VulkanRenderer::initVulkan()
{
#ifdef NDEBUG
    std::vector<const char*> layers = {};
#else
    std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
#endif
    vulkanDevice = std::make_unique<Device>(
        window,
        std::vector<const char*>{vk::KHRSwapchainExtensionName},
        layers);
    msaaSamples                    = vulkanDevice->getMsaaSamples();
    settings_.pendingMsaaSamples   = msaaSamples;

    swapchain = std::make_unique<Swapchain>(*vulkanDevice, window, settings_.pendingPresentMode);
    // Sync to the mode the swapchain actually chose (hardware may fall back).
    settings_.pendingPresentMode = swapchain->getPresentMode();

    textureManager_.init(*vulkanDevice);

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
    createNormalResources();
    createShadowMapResources();
    createShadowMapSampler();

    // --- Scene and GPU data ---
    loadScene();
    createUniformBuffers();

    // --- Descriptors ---
    createDescriptorPool();
    createDescriptorSets();
    createSkyDescriptorSets();

    // --- Synchronisation ---
    createCommandBuffers();
    createSyncObjects();
}

// ─── Descriptor set layout ──────────────────────────────────────────────────

void VulkanRenderer::createDescriptorSetLayout()
{
    std::array bindings = {
        vk::DescriptorSetLayoutBinding(
            0, vk::DescriptorType::eUniformBuffer, 1,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
        // Binding 1: bindless texture array. PARTIALLY_BOUND lets us leave
        // unused slots empty — only accessed slots need to be written.
        vk::DescriptorSetLayoutBinding(
            1, vk::DescriptorType::eCombinedImageSampler, TextureManager::MAX_TEXTURES,
            vk::ShaderStageFlagBits::eFragment, nullptr),
        vk::DescriptorSetLayoutBinding(
            2, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment, nullptr)};

    std::array<vk::DescriptorBindingFlags, 3> bindingFlags = {
        vk::DescriptorBindingFlags{},
        vk::DescriptorBindingFlagBits::ePartiallyBound,
        vk::DescriptorBindingFlags{}};

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .bindingCount  = static_cast<uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data()};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .pNext        = &bindingFlagsInfo,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data()};
    descriptorSetLayout = vk::raii::DescriptorSetLayout(
        vulkanDevice->getLogicalDevice(), layoutInfo);
}

// ─── Pipelines ──────────────────────────────────────────────────────────────

std::vector<char> VulkanRenderer::readFile(const std::filesystem::path& path)
{
    std::ifstream file(basePath_ / path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("failed to open file '" + path.string() + "'");
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

[[nodiscard]] vk::raii::ShaderModule
VulkanRenderer::createShaderModule(const std::vector<char>& code)
{
    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size() * sizeof(char),
        .pCode    = reinterpret_cast<const uint32_t*>(code.data())};
    return vk::raii::ShaderModule{vulkanDevice->getLogicalDevice(), createInfo};
}

void VulkanRenderer::createGraphicsPipeline()
{
    vk::raii::ShaderModule shaderModule =
        createShaderModule(readFile("shaders/scene.spv"));
    vk::PipelineShaderStageCreateInfo stages[] = {
        {.stage = vk::ShaderStageFlagBits::eVertex,   .module = shaderModule, .pName = "vertMain"},
        {.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"}};

    auto bindingDesc  = getVertexBindingDescription();
    auto attributeDesc = getVertexAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &bindingDesc,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDesc.size()),
        .pVertexAttributeDescriptions    = attributeDesc.data()};

    std::vector dynStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
        .pDynamicStates    = dynStates.data()};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode  = vk::PolygonMode::eFill,
        .cullMode     = vk::CullModeFlagBits::eBack,
        .frontFace    = vk::FrontFace::eCounterClockwise,
        .lineWidth    = 1.0f};
    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = msaaSamples};
    vk::PipelineColorBlendAttachmentState colorBlendAtt{
        .blendEnable    = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    std::array<vk::PipelineColorBlendAttachmentState, 2> blendAtts = {colorBlendAtt, colorBlendAtt};
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable  = vk::False,
        .attachmentCount= uint32_t(blendAtts.size()),
        .pAttachments   = blendAtts.data()};
    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset     = 0,
        .size       = sizeof(PushConstants)};
    vk::PipelineLayoutCreateInfo layoutInfo{
        .setLayoutCount        = 1,
        .pSetLayouts           = &*descriptorSetLayout,
        .pushConstantRangeCount= 1,
        .pPushConstantRanges   = &pushConstantRange};
    pipelineLayout = vk::raii::PipelineLayout(vulkanDevice->getLogicalDevice(), layoutInfo);

    vk::Format depthFmt = vulkanDevice->findDepthFormat();
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable  = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp   = vk::CompareOp::eLess};
    
    std::array<vk::Format, 2> colorFormats = {
        swapchain->getSurfaceFormat().format,
        vk::Format::eR8G8B8A8Unorm};

    vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount    = static_cast<uint32_t>(colorFormats.size()),
        .pColorAttachmentFormats = colorFormats.data(),
        .depthAttachmentFormat   = depthFmt};

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pNext             = &renderingInfo,
        .stageCount        = 2,
        .pStages           = stages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState    = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState= &depthStencil,
        .pColorBlendState  = &colorBlending,
        .pDynamicState     = &dynamicState,
        .layout            = pipelineLayout};
    graphicsPipeline = vk::raii::Pipeline(vulkanDevice->getLogicalDevice(), nullptr, pipelineInfo);
}

void VulkanRenderer::createShadowPipeline()
{
    vk::raii::ShaderModule shaderModule =
        createShaderModule(readFile("shaders/shadow.spv"));
    vk::PipelineShaderStageCreateInfo vertStage{
        .stage  = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule,
        .pName  = "vertMain"};

    auto bindingDesc = getVertexBindingDescription();
    vk::VertexInputAttributeDescription posAttr{
        0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)};
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &bindingDesc,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions    = &posAttr};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};
    vk::Viewport viewport{0.0f, 0.0f,
        static_cast<float>(SHADOW_MAP_SIZE), static_cast<float>(SHADOW_MAP_SIZE),
        0.0f, 1.0f};
    vk::Rect2D scissor{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1, .pViewports = &viewport,
        .scissorCount  = 1, .pScissors  = &scissor};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode             = vk::PolygonMode::eFill,
        .cullMode                = vk::CullModeFlagBits::eFront,
        .frontFace               = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable         = vk::True,
        .depthBiasConstantFactor = 1.25f,
        .depthBiasSlopeFactor    = 1.75f,
        .lineWidth               = 1.0f};
    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1};
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable  = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp   = vk::CompareOp::eLessOrEqual};
    vk::PipelineColorBlendStateCreateInfo colorBlending{.attachmentCount = 0};

    vk::Format depthFmt = vulkanDevice->findDepthFormat();
    vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount = 0,
        .depthAttachmentFormat= depthFmt};

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pNext               = &renderingInfo,
        .stageCount          = 1,
        .pStages             = &vertStage,
        .pVertexInputState   = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState   = &multisampling,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlending,
        .layout              = pipelineLayout};
    shadowPipeline = vk::raii::Pipeline(vulkanDevice->getLogicalDevice(), nullptr, pipelineInfo);
}

// ─── Sky pipeline ────────────────────────────────────────────────────────────

void VulkanRenderer::createSkyDescriptorSetLayout()
{
    vk::DescriptorSetLayoutBinding uboBinding{
        .binding         = 0,
        .descriptorType  = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags      = vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo info{
        .bindingCount = 1, .pBindings = &uboBinding};
    skyDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        vulkanDevice->getLogicalDevice(), info);
}

void VulkanRenderer::createSkyPipeline()
{
    vk::raii::ShaderModule skyShader = createShaderModule(readFile("shaders/sky.spv"));
    vk::PipelineShaderStageCreateInfo stages[2] = {
        {.stage = vk::ShaderStageFlagBits::eVertex,   .module = skyShader, .pName = "skyVert"},
        {.stage = vk::ShaderStageFlagBits::eFragment, .module = skyShader, .pName = "skyFrag"}};

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

    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList};
    std::vector<vk::DynamicState> dynStates = {
        vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynState{
        .dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
        .pDynamicStates    = dynStates.data()};
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1, .scissorCount = 1};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eNone,
        .lineWidth   = 1.0f};
    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = msaaSamples};
    vk::PipelineColorBlendAttachmentState colorBlendAtt{
        .blendEnable    = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    std::array<vk::PipelineColorBlendAttachmentState, 2> blendAtts = {colorBlendAtt, colorBlendAtt};
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .attachmentCount = uint32_t(blendAtts.size()), .pAttachments = blendAtts.data()};
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable  = vk::True,
        .depthWriteEnable = vk::False,
        .depthCompareOp   = vk::CompareOp::eLessOrEqual};

    vk::Format depthFmt = vulkanDevice->findDepthFormat();
    std::array<vk::Format, 2> colorFormats = {
        swapchain->getSurfaceFormat().format,
        vk::Format::eR8G8B8A8Unorm};
    vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount    = uint32_t(colorFormats.size()),
        .pColorAttachmentFormats = colorFormats.data(),
        .depthAttachmentFormat   = depthFmt};

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

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo buf{uniformBuffers[i], 0, sizeof(UniformBufferObject)};
        vulkanDevice->getLogicalDevice().updateDescriptorSets(
            vk::WriteDescriptorSet{
                .dstSet         = skyDescriptorSets[i],
                .dstBinding     = 0,
                .descriptorCount= 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo    = &buf},
            {});
    }
}

// ─── Command infrastructure ─────────────────────────────────────────────────

// eResetCommandBuffer: allows re-recording each buffer every frame.
void VulkanRenderer::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo{
        .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = vulkanDevice->getGraphicsIndex()};
    commandPool = vk::raii::CommandPool(vulkanDevice->getLogicalDevice(), poolInfo);
}

void VulkanRenderer::createCommandBuffers()
{
    commandBuffers.clear();
    commandBuffers = vk::raii::CommandBuffers(
        vulkanDevice->getLogicalDevice(),
        vk::CommandBufferAllocateInfo{
            .commandPool        = commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT});
}

// ─── Render attachments ──────────────────────────────────────────────────────

// MSAA color image: GPU renders here at multiple samples/pixel, then resolves
// into the swapchain image. eTransient: contents never leave the GPU.
void VulkanRenderer::createColorResources()
{
    vk::Format colorFormat = swapchain->getSurfaceFormat().format;
    vulkanDevice->createImage(
        swapchain->getExtent().width, swapchain->getExtent().height,
        1, msaaSamples, colorFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        colorImage, colorImageMemory);
    colorImageView = vulkanDevice->createImageView(
        colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
}

void VulkanRenderer::createDepthResources()
{
    vk::Format depthFormat = vulkanDevice->findDepthFormat();
    vulkanDevice->createImage(
        swapchain->getExtent().width, swapchain->getExtent().height,
        1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        depthImage, depthImageMemory);
    depthImageView = vulkanDevice->createImageView(
        depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

void VulkanRenderer::createShadowMapResources()
{
    vk::Format depthFormat = vulkanDevice->findDepthFormat();
    vulkanDevice->createImage(
        SHADOW_MAP_SIZE, SHADOW_MAP_SIZE,
        1, vk::SampleCountFlagBits::e1, depthFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        shadowMapImage, shadowMapImageMemory);
    shadowMapImageView = vulkanDevice->createImageView(
        shadowMapImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

void VulkanRenderer::createNormalResources()
{
    vk::Format normalFormat = vk::Format::eR8G8B8A8Unorm;
    bool msaaEnabled = msaaSamples != vk::SampleCountFlagBits::e1;

    vulkanDevice->createImage(
        swapchain->getExtent().width, swapchain->getExtent().height,
        1, msaaSamples, normalFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment |
        (msaaEnabled ? vk::ImageUsageFlagBits{} : vk::ImageUsageFlagBits::eSampled),
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        normalImage, normalImageMemory);
    normalImageView = vulkanDevice->createImageView(
        normalImage, normalFormat, vk::ImageAspectFlagBits::eColor, 1);

    if (msaaEnabled)
    {
        vulkanDevice->createImage(
            swapchain->getExtent().width, swapchain->getExtent().height,
            1, vk::SampleCountFlagBits::e1, normalFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            normalResolveImage, normalResolveImageMemory);
        normalResolveImageView = vulkanDevice->createImageView(
            normalResolveImage, normalFormat, vk::ImageAspectFlagBits::eColor, 1);
    }
}

void VulkanRenderer::createShadowMapSampler()
{
    vk::SamplerCreateInfo samplerInfo{
        .magFilter               = vk::Filter::eLinear,
        .minFilter               = vk::Filter::eLinear,
        .mipmapMode              = vk::SamplerMipmapMode::eLinear,
        .addressModeU            = vk::SamplerAddressMode::eClampToBorder,
        .addressModeV            = vk::SamplerAddressMode::eClampToBorder,
        .addressModeW            = vk::SamplerAddressMode::eClampToBorder,
        .anisotropyEnable        = vk::False,
        .maxAnisotropy           = 1.0f,
        .compareEnable           = vk::True,
        .compareOp               = vk::CompareOp::eLessOrEqual,
        .borderColor             = vk::BorderColor::eFloatOpaqueWhite,
        .unnormalizedCoordinates = vk::False};
    shadowMapSampler = vk::raii::Sampler(vulkanDevice->getLogicalDevice(), samplerInfo);
}

// ─── Scene loading ───────────────────────────────────────────────────────────

void VulkanRenderer::uploadRenderable(Renderable& r, const std::vector<Vertex>& verts,
                                       const std::vector<uint32_t>& idxs)
{
    // Vertex buffer
    {
        vk::DeviceSize size = sizeof(verts[0]) * verts.size();
        vk::raii::Buffer       staging({});
        vk::raii::DeviceMemory stagingMem({});
        vulkanDevice->createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            staging, stagingMem);
        void* data = stagingMem.mapMemory(0, size);
        memcpy(data, verts.data(), size);
        stagingMem.unmapMemory();
        vulkanDevice->createBuffer(size,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            r.vertexBuffer, r.vertexBufferMemory);
        vulkanDevice->copyBuffer(staging, r.vertexBuffer, size);
    }
    // Index buffer
    {
        vk::DeviceSize size = sizeof(idxs[0]) * idxs.size();
        vk::raii::Buffer       staging({});
        vk::raii::DeviceMemory stagingMem({});
        vulkanDevice->createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            staging, stagingMem);
        void* data = stagingMem.mapMemory(0, size);
        memcpy(data, idxs.data(), size);
        stagingMem.unmapMemory();
        vulkanDevice->createBuffer(size,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            r.indexBuffer, r.indexBufferMemory);
        vulkanDevice->copyBuffer(staging, r.indexBuffer, size);
    }
    r.indexCount = static_cast<uint32_t>(idxs.size());
}

void VulkanRenderer::loadScene()
{
    auto scenePath = basePath_ / SCENE_PATH;
    std::ifstream f(scenePath);
    if (!f.is_open())
        throw std::runtime_error("failed to open scene file: " + scenePath.string());

    nlohmann::json scene = nlohmann::json::parse(f);

    // Optional skybox settings
    if (scene.contains("skybox"))
    {
        const auto& s = scene["skybox"];
        if (s.contains("horizonColor"))
            settings_.skyPush.horizonColor = {s["horizonColor"][0], s["horizonColor"][1], s["horizonColor"][2], 1.0f};
        if (s.contains("zenithColor"))
            settings_.skyPush.zenithColor  = {s["zenithColor"][0],  s["zenithColor"][1],  s["zenithColor"][2],  1.0f};
        if (s.contains("groundColor"))
            settings_.skyPush.groundColor  = {s["groundColor"][0],  s["groundColor"][1],  s["groundColor"][2],  1.0f};
        if (s.contains("sunColor"))
            settings_.skyPush.sunParams = {s["sunColor"][0], s["sunColor"][1], s["sunColor"][2],
                                            s.value("sunSize", 0.9998f)};
    }

    // Optional point lights (overrides compiled-in defaults when present)
    if (scene.contains("pointLights"))
    {
        settings_.pointLights.clear();
        for (const auto& pl : scene["pointLights"])
        {
            PointLightData d;
            if (pl.contains("position"))
                d.position = {pl["position"][0], pl["position"][1], pl["position"][2]};
            if (pl.contains("color"))
                d.color = {pl["color"][0], pl["color"][1], pl["color"][2]};
            d.intensity = pl.value("intensity", 3.0f);
            d.radius    = pl.value("radius",    8.0f);
            d.enabled   = pl.value("enabled",   true);
            settings_.pointLights.push_back(d);
        }
    }

    auto assetPath = [&](const std::string& rel) { return basePath_ / rel; };

    for (const auto& obj : scene["objects"])
    {
        std::string mesh  = obj["mesh"];
        glm::vec3   pos   = {obj["position"][0], obj["position"][1], obj["position"][2]};
        glm::vec3   color = obj.contains("color")
            ? glm::vec3{obj["color"][0], obj["color"][1], obj["color"][2]}
            : glm::vec3{1.0f};
        float size = obj.value("size", 1.0f);

        auto optStr = [&](const char* key) -> std::optional<std::string> {
            return obj.contains(key)
                ? std::optional<std::string>{obj[key].get<std::string>()}
                : std::nullopt;
        };

        glm::vec3 rotDeg = obj.contains("rotation")
            ? glm::vec3{obj["rotation"][0], obj["rotation"][1], obj["rotation"][2]}
            : glm::vec3{0.0f};
        float scale = obj.value("scale", 1.0f);

        std::pair<std::vector<Vertex>, std::vector<uint32_t>> meshData;

        if (mesh == "cube")
            meshData = makeCube(color, size);
        else if (mesh == "plane")
            meshData = makePlane(color, size);
        else if (mesh == "sphere")
        {
            uint32_t sectors = obj.value("sectors", 32u);
            uint32_t stacks  = obj.value("stacks",  16u);
            meshData = makeSphere(color, size, sectors, stacks);
        }
        else
        {
            auto ext = std::filesystem::path(mesh).extension();
            if (ext != ".gltf" && ext != ".glb")
                throw std::runtime_error("unsupported mesh format '" + mesh + "' — only .gltf/.glb are supported");

            bool yUpToZUp = obj.value("yUpToZUp", false);
            glm::mat4 parentTransform = buildModelMatrix(pos, rotDeg, scale);
            for (auto& prim : loadGLTF(assetPath(mesh), yUpToZUp))
            {
                auto resolveSlot = [&](const GltfImage& img, bool linear) -> uint32_t {
                    if (!img.path.empty())   return textureManager_.load(img.path, linear);
                    if (!img.bytes.empty())  return textureManager_.loadFromMemory(img.bytes, linear);
                    return 0xFFFFu;
                };
                Renderable r;
                r.label                  = mesh;
                r.position               = pos;
                r.rotationDeg            = rotDeg;
                r.scale                  = scale;
                r.modelMatrix            = parentTransform * prim.transform;
                r.textureIndex           = resolveSlot(prim.baseColor,         false);
                r.normalMapIndex         = resolveSlot(prim.normalMap,         true);
                r.metallicRoughnessIndex = resolveSlot(prim.metallicRoughness, true);
                r.heightMapIndex         = 0xFFFFu;
                uploadRenderable(r, prim.vertices, prim.indices);
                renderables.push_back(std::move(r));
            }
            continue;
        }

        Renderable r;
        r.label                  = mesh;
        r.position               = pos;
        r.rotationDeg            = rotDeg;
        r.scale                  = scale;
        r.modelMatrix            = buildModelMatrix(pos, rotDeg, scale);
        r.textureIndex           = optStr("texture")    ? textureManager_.load(assetPath(*optStr("texture")))    : 0xFFFFu;
        r.metallicRoughnessIndex = optStr("specularMap")? textureManager_.load(assetPath(*optStr("specularMap"))): 0xFFFFu;
        r.normalMapIndex         = optStr("normalMap")  ? textureManager_.load(assetPath(*optStr("normalMap")),  true) : 0xFFFFu;
        r.heightMapIndex         = optStr("heightMap")  ? textureManager_.load(assetPath(*optStr("heightMap")),  true) : 0xFFFFu;
        uploadRenderable(r, meshData.first, meshData.second);
        renderables.push_back(std::move(r));
    }
    std::cout << "Loaded " << textureManager_.count() << " unique textures\n";
}

// ─── Uniform buffers ─────────────────────────────────────────────────────────

// Host-visible, persistently mapped — CPU writes UBO data each frame via the
// mapped pointer; no explicit map/unmap per frame needed.
void VulkanRenderer::createUniformBuffers()
{
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        vk::raii::Buffer       buf({});
        vk::raii::DeviceMemory bufMem({});
        vulkanDevice->createBuffer(
            bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            buf, bufMem);
        uniformBuffers.emplace_back(std::move(buf));
        uniformBuffersMemory.emplace_back(std::move(bufMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
    }
}

// ─── Descriptor pool + sets ──────────────────────────────────────────────────

void VulkanRenderer::createDescriptorPool()
{
    std::array poolSizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                               MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,
                               MAX_FRAMES_IN_FLIGHT * (TextureManager::MAX_TEXTURES + 1))};
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets       = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes    = poolSizes.data()};
    descriptorPool = vk::raii::DescriptorPool(vulkanDevice->getLogicalDevice(), poolInfo);
}

void VulkanRenderer::createDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
    descriptorSets.clear();
    descriptorSets = vulkanDevice->getLogicalDevice().allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{
            .descriptorPool     = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts        = layouts.data()});

    auto textureInfos = textureManager_.descriptorImageInfos();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo bufferInfo{uniformBuffers[i], 0, sizeof(UniformBufferObject)};
        vk::DescriptorImageInfo  shadowInfo{*shadowMapSampler, *shadowMapImageView,
                                            vk::ImageLayout::eShaderReadOnlyOptimal};

        std::vector<vk::WriteDescriptorSet> writes{
            vk::WriteDescriptorSet{
                .dstSet          = descriptorSets[i],
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo     = &bufferInfo},
            vk::WriteDescriptorSet{
                .dstSet          = descriptorSets[i],
                .dstBinding      = 2,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo      = &shadowInfo}};

        if (!textureInfos.empty())
        {
            writes.push_back(vk::WriteDescriptorSet{
                .dstSet          = descriptorSets[i],
                .dstBinding      = 1,
                .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo      = textureInfos.data()});
        }
        vulkanDevice->getLogicalDevice().updateDescriptorSets(writes, {});
    }
}

// ─── Synchronisation ─────────────────────────────────────────────────────────

// presentComplete: signals when the display is done reading the swapchain image.
// renderFinished:  signals when rendering is done and the image can be presented.
// inFlightFence:   CPU blocks here to avoid overwriting resources still in use.
void VulkanRenderer::createSyncObjects()
{
    assert(presentCompleteSemaphores.empty() &&
           renderFinishedSemaphores.empty() && inFlightFences.empty());

    for (size_t i = 0; i < swapchain->getImages().size(); ++i)
        renderFinishedSemaphores.emplace_back(
            vulkanDevice->getLogicalDevice(), vk::SemaphoreCreateInfo());

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        presentCompleteSemaphores.emplace_back(
            vulkanDevice->getLogicalDevice(), vk::SemaphoreCreateInfo());
        inFlightFences.emplace_back(
            vulkanDevice->getLogicalDevice(),
            vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
}

// ─── Main loop ───────────────────────────────────────────────────────────────

void VulkanRenderer::mainLoop()
{
    auto lastTime = std::chrono::steady_clock::now();

    double mx0, my0;
    glfwGetCursorPos(window, &mx0, &my0);
    lastMousePos = {static_cast<float>(mx0), static_cast<float>(my0)};

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        auto  now = std::chrono::steady_clock::now();
        float dt  = std::chrono::duration<float>(now - lastTime).count();
        lastTime  = now;

        // Camera fly mode: hold right mouse button
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        bool rightHeld      = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        bool imguiWantsMouse= ImGui::GetIO().WantCaptureMouse;

        if (rightHeld && !imguiWantsMouse)
        {
            if (!cameraMode)
            {
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

        drawFrame();
    }
    vulkanDevice->getLogicalDevice().waitIdle();
}

void VulkanRenderer::drawFrame()
{
    if (settings_.pendingMsaaSamples != msaaSamples)
    {
        msaaSamples = settings_.pendingMsaaSamples;
        rebuildMsaa();
    }
    if (settings_.pendingPresentMode != swapchain->getPresentMode())
        recreateSwapChain();

    auto fenceResult = vulkanDevice->getLogicalDevice().waitForFences(
        *inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
        throw std::runtime_error("failed to wait for fence");

    auto [result, imageIndex] = swapchain->getSwapChain().acquireNextImage(
        UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }
    else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("failed to acquire swap chain image");

    vulkanDevice->getLogicalDevice().resetFences(*inFlightFences[frameIndex]);
    commandBuffers[frameIndex].reset();

    // Build the ImGui widget tree for this frame.
    imguiLayer_.beginFrame();
    sceneEditor_.draw(settings_, renderables, camera, {
        swapchain->getExtent(),
        vulkanDevice->getPhysicalDevice().getProperties().deviceName.data(),
        static_cast<uint32_t>(textureManager_.count()),
        vulkanDevice->getSupportedMsaaSamples()});
    imguiLayer_.endFrame();

    updateUniformBuffer(frameIndex);
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vulkanDevice->getGraphicsQueue().submit(
        vk::SubmitInfo{
            .waitSemaphoreCount   = 1,
            .pWaitSemaphores      = &*presentCompleteSemaphores[frameIndex],
            .pWaitDstStageMask    = &waitStage,
            .commandBufferCount   = 1,
            .pCommandBuffers      = &*commandBuffers[frameIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores    = &*renderFinishedSemaphores[imageIndex]},
        *inFlightFences[frameIndex]);

    result = vulkanDevice->getGraphicsQueue().presentKHR(
        vk::PresentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &*renderFinishedSemaphores[imageIndex],
            .swapchainCount     = 1,
            .pSwapchains        = &*swapchain->getSwapChain(),
            .pImageIndices      = &imageIndex});

    if (result == vk::Result::eSuboptimalKHR ||
        result == vk::Result::eErrorOutOfDateKHR || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to present swap chain image");
    }

    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ─── Per-frame updates ───────────────────────────────────────────────────────

void VulkanRenderer::updateUniformBuffer(uint32_t currentImage)
{
    static auto startTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
        std::chrono::high_resolution_clock::now() - startTime).count();

    UniformBufferObject ubo{};
    float aspect = static_cast<float>(swapchain->getExtent().width) /
                   static_cast<float>(swapchain->getExtent().height);
    ubo.view = camera.getViewMatrix();
    ubo.proj = camera.getProjectionMatrix(aspect);

    float dt  = time - prevTime;
    prevTime  = time;
    if (settings_.lightOrbit)
    {
        settings_.lightAngle += (dt * 0.5f) * glm::two_pi<float>() / 10.0f;
        if (settings_.lightAngle > glm::two_pi<float>())
            settings_.lightAngle -= glm::two_pi<float>();
    }

    glm::vec3 lightPos{glm::cos(settings_.lightAngle) * 8.0f,
                       glm::sin(settings_.lightAngle) * 8.0f, 6.0f};

    // Camera-fitted shadow projection: unproject camera NDC corners to world
    // space, then fit the light's ortho frustum tightly around them.
    glm::mat4 camProj = glm::perspective(glm::radians(75.0f), aspect, 0.5f, settings_.shadowFar);
    camProj[1][1] *= -1; // Vulkan Y-flip
    glm::mat4 invCamVP = glm::inverse(camProj * camera.getViewMatrix());

    std::array<glm::vec3, 8> corners;
    int ci = 0;
    for (float x : {-1.0f, 1.0f})
        for (float y : {-1.0f, 1.0f})
            for (float z : {0.0f, 1.0f}) // Vulkan [0,1] depth range
            {
                glm::vec4 pt = invCamVP * glm::vec4(x, y, z, 1.0f);
                corners[ci++] = glm::vec3(pt) / pt.w;
            }

    glm::vec3 frustumCenter{};
    for (auto& c : corners) frustumCenter += c;
    frustumCenter /= 8.0f;

    glm::vec3 lightDir  = glm::normalize(lightPos);
    glm::mat4 lightView = glm::lookAt(
        frustumCenter + lightDir * settings_.shadowFar, // light is upstream of the frustum
        frustumCenter,
        glm::vec3(0.0f, 0.0f, 1.0f));

    float minX = std::numeric_limits<float>::max(),  maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max(),  maxY = std::numeric_limits<float>::lowest();
    float minZ = std::numeric_limits<float>::max(),  maxZ = std::numeric_limits<float>::lowest();
    for (auto& c : corners)
    {
        glm::vec4 lc = lightView * glm::vec4(c, 1.0f);
        minX = std::min(minX, lc.x); maxX = std::max(maxX, lc.x);
        minY = std::min(minY, lc.y); maxY = std::max(maxY, lc.y);
        minZ = std::min(minZ, lc.z); maxZ = std::max(maxZ, lc.z);
    }

    // glm's lookAt puts -Z forward, so near = -maxZ, far = -minZ.
    // Add a margin to catch shadow casters behind the view frustum.
    float nearClip = std::max(0.01f, -maxZ);
    float farClip  = -minZ + (-minZ - (-maxZ)) * 0.5f;
    ubo.lightSpaceMatrix = glm::ortho(minX, maxX, minY, maxY, nearClip, farClip) * lightView;

    ubo.lightDir      = glm::vec4(glm::normalize(lightPos), 0.0f);
    ubo.cameraPos     = glm::vec4(camera.position, 0.0f);
    ubo.materialParams = glm::vec4(settings_.ambient, settings_.defaultRoughness,
                                    settings_.defaultMetallic,
                                    settings_.tonemapping ? settings_.exposure : 0.0f);

    int activeLights = 0;
    for (const auto& pl : settings_.pointLights)
    {
        if (!pl.enabled || activeLights >= RenderSettings::MAX_POINT_LIGHTS) continue;
        ubo.pointLightPos  [activeLights] = glm::vec4(pl.position, pl.intensity);
        ubo.pointLightColor[activeLights] = glm::vec4(pl.color,    pl.radius);
        ++activeLights;
    }
    ubo.lightCounts  = glm::vec4(static_cast<float>(activeLights), 0, 0, 0);
    ubo.shadowParams = glm::vec4(settings_.shadowBiasMin, settings_.shadowBiasMax, 0, 0);
    ubo.pomParams    = glm::vec4(settings_.pomDepthScale, settings_.pomMinSteps,
                                  settings_.pomMaxSteps, 0);

    float effectiveDensity = settings_.fogEnabled ? settings_.fogDensity : 0.0f;
    if (settings_.fogSyncSky)
        settings_.fogColor = glm::vec3(settings_.skyPush.horizonColor);
    ubo.fogParams = glm::vec4(effectiveDensity, settings_.fogHeightFalloff,
                               settings_.fogMaxOpacity, 0);
    ubo.fogColor  = glm::vec4(settings_.fogColor, 1.0f);

    ubo.invProj = glm::inverse(ubo.proj);
    glm::mat4 viewRot = ubo.view;
    viewRot[3]    = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    ubo.invViewRot = glm::transpose(viewRot);

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

// ─── Command buffer recording ────────────────────────────────────────────────

void VulkanRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& cmd = commandBuffers[frameIndex];
    cmd.begin({});

    // ---- Shadow pass ----
    recordImageBarrier(*shadowMapImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);

    vk::ClearValue shadowClear = vk::ClearDepthStencilValue(1.0f, 0);
    vk::RenderingAttachmentInfo shadowDepthAtt{
        .imageView   = *shadowMapImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp      = vk::AttachmentLoadOp::eClear,
        .storeOp     = vk::AttachmentStoreOp::eStore,
        .clearValue  = shadowClear};
    cmd.beginRendering(vk::RenderingInfo{
        .renderArea      = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}},
        .layerCount      = 1,
        .pDepthAttachment= &shadowDepthAtt});

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowPipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);
    for (const auto& r : renderables)
    {
        PushConstants pc{r.modelMatrix, r.textureIndex, r.metallicRoughnessIndex,
                         r.normalMapIndex, r.heightMapIndex};
        cmd.pushConstants2(vk::PushConstantsInfo{}
            .setLayout(*pipelineLayout)
            .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
            .setOffset(0).setSize(sizeof(PushConstants)).setPValues(&pc));
        cmd.bindVertexBuffers(0, *r.vertexBuffer, {0});
        cmd.bindIndexBuffer(*r.indexBuffer, 0, vk::IndexType::eUint32);
        cmd.drawIndexed(r.indexCount, 1, 0, 0, 0);
    }
    cmd.endRendering();

    // Shadow map → shader-readable
    recordImageBarrier(*shadowMapImage,
        vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eDepth);

    // ---- Main color pass ----
    bool msaaEnabled = msaaSamples != vk::SampleCountFlagBits::e1;

    if (msaaEnabled)
        recordImageBarrier(*colorImage,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
            {}, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

    recordImageBarrier(swapchain->getImages()[imageIndex],
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);

    recordImageBarrier(*normalImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    if (msaaEnabled)
        recordImageBarrier(*normalResolveImage,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
            {}, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);
    
    recordImageBarrier(*depthImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);

    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderingAttachmentInfo colorAtt =
        msaaEnabled
            ? vk::RenderingAttachmentInfo{
                .imageView        = *colorImageView,
                .imageLayout      = vk::ImageLayout::eColorAttachmentOptimal,
                .resolveMode      = vk::ResolveModeFlagBits::eAverage,
                .resolveImageView = *swapchain->getImageViews()[imageIndex],
                .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp           = vk::AttachmentLoadOp::eClear,
                .storeOp          = vk::AttachmentStoreOp::eDontCare,
                .clearValue       = clearColor}
            : vk::RenderingAttachmentInfo{
                .imageView   = *swapchain->getImageViews()[imageIndex],
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp      = vk::AttachmentLoadOp::eClear,
                .storeOp     = vk::AttachmentStoreOp::eStore,
                .clearValue  = clearColor};

    vk::RenderingAttachmentInfo depthAtt{
        .imageView   = *depthImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp      = vk::AttachmentLoadOp::eClear,
        .storeOp     = vk::AttachmentStoreOp::eDontCare,
        .clearValue  = clearDepth};

    vk::RenderingAttachmentInfo normalAtt =
        msaaEnabled
            ? vk::RenderingAttachmentInfo{
                .imageView          = *normalImageView,
                .imageLayout        = vk::ImageLayout::eColorAttachmentOptimal,
                .resolveMode        = vk::ResolveModeFlagBits::eAverage,
                .resolveImageView   = *normalResolveImageView,
                .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp             = vk::AttachmentLoadOp::eClear,
                .storeOp            = vk::AttachmentStoreOp::eDontCare,
                .clearValue         = clearColor}
            : vk::RenderingAttachmentInfo{
                .imageView   = *normalImageView,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp      = vk::AttachmentLoadOp::eClear,
                .storeOp     = vk::AttachmentStoreOp::eStore,
                .clearValue  = clearColor};

    std::array<vk::RenderingAttachmentInfo, 2> attachments = {colorAtt, normalAtt};
    
    cmd.beginRendering(vk::RenderingInfo{
        .renderArea           = {{0, 0}, swapchain->getExtent()},
        .layerCount           = 1,
        .colorAttachmentCount = uint32_t(attachments.size()),
        .pColorAttachments    = attachments.data(),
        .pDepthAttachment     = &depthAtt});

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    cmd.setViewport(0, vk::Viewport{0.0f, 0.0f,
        static_cast<float>(swapchain->getExtent().width),
        static_cast<float>(swapchain->getExtent().height), 0.0f, 1.0f});
    cmd.setScissor(0, vk::Rect2D{{0, 0}, swapchain->getExtent()});
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);

    for (const auto& r : renderables)
    {
        PushConstants pc{r.modelMatrix, r.textureIndex, r.metallicRoughnessIndex,
                         r.normalMapIndex, r.heightMapIndex};
        cmd.pushConstants2(vk::PushConstantsInfo{}
            .setLayout(*pipelineLayout)
            .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
            .setOffset(0).setSize(sizeof(PushConstants)).setPValues(&pc));
        cmd.bindVertexBuffers(0, *r.vertexBuffer, {0});
        cmd.bindIndexBuffer(*r.indexBuffer, 0, vk::IndexType::eUint32);
        cmd.drawIndexed(r.indexCount, 1, 0, 0, 0);
    }

    // Sky pass: fullscreen triangle at depth 1.0, fills pixels where no geometry was drawn.
    if (settings_.skyEnabled)
    {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *skyPipeline);
        cmd.setViewport(0, vk::Viewport{0.0f, 0.0f,
            static_cast<float>(swapchain->getExtent().width),
            static_cast<float>(swapchain->getExtent().height), 0.0f, 1.0f});
        cmd.setScissor(0, vk::Rect2D{{0, 0}, swapchain->getExtent()});
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            skyPipelineLayout, 0, *skyDescriptorSets[frameIndex], nullptr);
        cmd.pushConstants2(vk::PushConstantsInfo{}
            .setLayout(*skyPipelineLayout)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment)
            .setOffset(0).setSize(sizeof(SkyPushConstants)).setPValues(&settings_.skyPush));
        cmd.draw(3, 1, 0, 0);
    }
    cmd.endRendering();

    // Barrier before ImGui pass so it sees finished color writes.
    vk::MemoryBarrier2 memBarrier{
        .srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                         vk::AccessFlagBits2::eColorAttachmentWrite};
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1, .pMemoryBarriers = &memBarrier});

    // ImGui pass — beginRendering/endRendering handled inside ImGuiLayer.
    imguiLayer_.renderDrawData(cmd,
        *swapchain->getImageViews()[imageIndex],
        swapchain->getExtent());

    // Transition swapchain image to present layout.
    recordImageBarrier(swapchain->getImages()[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite, {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);

    // Transition normal image (or its resolve) to shader-readable for the SSAO pass.
    recordImageBarrier(msaaEnabled ? *normalResolveImage : *normalImage,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);

    cmd.end();
}

void VulkanRenderer::recordImageBarrier(vk::Image image,
                                         vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                         vk::AccessFlags2 srcAccess, vk::AccessFlags2 dstAccess,
                                         vk::PipelineStageFlags2 srcStage,
                                         vk::PipelineStageFlags2 dstStage,
                                         vk::ImageAspectFlags aspectFlags)
{
    vk::ImageMemoryBarrier2 barrier{
        .srcStageMask        = srcStage,
        .srcAccessMask       = srcAccess,
        .dstStageMask        = dstStage,
        .dstAccessMask       = dstAccess,
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {aspectFlags, 0, 1, 0, 1}};
    commandBuffers[frameIndex].pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

void VulkanRenderer::recreateSwapChain()
{
    vulkanDevice->getLogicalDevice().waitIdle();
    cleanupSwapChain();
    swapchain->recreate(window, settings_.pendingPresentMode);
    ImGui_ImplVulkan_SetMinImageCount(
        static_cast<uint32_t>(swapchain->getImages().size()));
    createDepthResources();
    createColorResources();
    createNormalResources();
}

void VulkanRenderer::rebuildMsaa()
{
    vulkanDevice->getLogicalDevice().waitIdle();
    colorImageView  = nullptr;
    colorImage      = nullptr;
    colorImageMemory= nullptr;
    depthImageView  = nullptr;
    depthImage      = nullptr;
    depthImageMemory= nullptr;
    graphicsPipeline= nullptr;
    skyPipeline     = nullptr;
    createDepthResources();
    createColorResources();
    createGraphicsPipeline();
    createSkyPipeline();
}

void VulkanRenderer::cleanupSwapChain()
{
    normalResolveImageView   = nullptr;
    normalResolveImage       = nullptr;
    normalResolveImageMemory = nullptr;
    normalImageView          = nullptr;
    normalImage              = nullptr;
    normalImageMemory        = nullptr;
    colorImageView  = nullptr;
    colorImage      = nullptr;
    colorImageMemory= nullptr;
    depthImageView  = nullptr;
    depthImage      = nullptr;
    depthImageMemory= nullptr;
}

// RAII handles call vkDestroy automatically when set to nullptr, but order
// matters: child objects must be destroyed before their parents.
void VulkanRenderer::cleanup()
{
    cleanupSwapChain();
    renderables.clear();

    // ImGui must be shut down before the device is destroyed.
    imguiLayer_.shutdown();

    inFlightFences.clear();
    renderFinishedSemaphores.clear();
    presentCompleteSemaphores.clear();
    commandBuffers.clear();
    commandPool         = nullptr;
    skyDescriptorSets.clear();
    skyDescriptorPool   = nullptr;
    skyPipeline         = nullptr;
    skyPipelineLayout   = nullptr;
    skyDescriptorSetLayout = nullptr;
    shadowPipeline      = nullptr;
    graphicsPipeline    = nullptr;
    pipelineLayout      = nullptr;
    shadowMapSampler    = nullptr;
    shadowMapImageView  = nullptr;
    shadowMapImage      = nullptr;
    shadowMapImageMemory= nullptr;
    textureManager_     = TextureManager{};  // releases all GPU textures
    descriptorSets.clear();
    descriptorPool      = nullptr;
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    descriptorSetLayout = nullptr;
    swapchain.reset();
    vulkanDevice.reset();

    glfwDestroyWindow(window);
    glfwTerminate();
}

// ─── ImGui ───────────────────────────────────────────────────────────────────

void VulkanRenderer::imGuiInit()
{
    imguiLayer_.init(*vulkanDevice, *swapchain, window);
}
