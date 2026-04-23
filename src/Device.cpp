#include "Device.hpp"
#include <vulkan/vulkan_profiles.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

Device::Device(GLFWwindow *window, std::vector<const char *> requiredDeviceExtensions, std::vector<const char *> validationLayers)
    : requiredDeviceExtensions(requiredDeviceExtensions), validationLayers(validationLayers)
{
    createInstance();
    setupDebugMessenger();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
    createUploadCommandPool();
}

vk::raii::Instance &Device::getInstance() { return instance; }
vk::raii::SurfaceKHR &Device::getSurface() { return surface; }
vk::raii::PhysicalDevice &Device::getPhysicalDevice() { return physicalDevice; }
vk::raii::Device &Device::getLogicalDevice() { return device; }
vk::raii::Queue &Device::getGraphicsQueue() { return queue; }
uint32_t Device::getGraphicsIndex() const { return graphicsIndex; }
vk::SampleCountFlagBits Device::getMsaaSamples() const { return msaaSamples; }
vk::SampleCountFlagBits Device::getMaxMsaaSamples() const { return maxMsaaSamples; }
vk::SampleCountFlags Device::getSupportedMsaaSamples() const { return supportedMsaaSamples; }

void Device::createInstance()
{
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion14};

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(
        validationLayers, [&layerProperties](auto const &requiredLayer)
        { return std::ranges::none_of(
              layerProperties, [requiredLayer](auto const &layerProperty)
              { return strcmp(layerProperty.layerName, requiredLayer) == 0; }); });
    if (unsupportedLayerIt != validationLayers.end())
    {
        throw std::runtime_error("failed to create instance: unsupported layer: " +
                                 std::string(*unsupportedLayerIt));
    }

    // Get the required extensions.
    auto requiredExtensions = getRequiredExtensions();

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
        throw std::runtime_error("failed to create instance: unsupported extension: " +
                                 std::string(*unsupportedPropertyIt));
    }

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.data(),
        .enabledExtensionCount =
            static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data()};

    instance = vk::raii::Instance(context, createInfo);
}

void Device::setupDebugMessenger()
{
    if (validationLayers.empty())
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

void Device::createSurface(GLFWwindow *window)
{
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
    {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
}

void Device::pickPhysicalDevice()
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
            break;
        }
    }
    if (physicalDevice == nullptr)
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void Device::createLogicalDevice()
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
        throw std::runtime_error("failed to find a graphics + present queue!");
    }

    graphicsIndex = queueIndex;

    // query for Vulkan 1.3 features
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan12Features,
                       vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
        featureChain = {
            {.features = {.samplerAnisotropy =
                              true}}, // vk::PhysicalDeviceFeatures2
            {.shaderDrawParameters =
                 true}, // vk::PhysicalDeviceVulkan11Features
            {.shaderSampledImageArrayNonUniformIndexing = true,
             .descriptorBindingPartiallyBound = true,
             .runtimeDescriptorArray = true}, // vk::PhysicalDeviceVulkan12Features
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
            static_cast<uint32_t>(requiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = requiredDeviceExtensions.data()};

    constexpr VpProfileProperties profile{VP_KHR_ROADMAP_2022_NAME,
                                          VP_KHR_ROADMAP_2022_SPEC_VERSION};
    VpDeviceCreateInfo vpDeviceCreateInfo{
        .pCreateInfo             = reinterpret_cast<const VkDeviceCreateInfo *>(&deviceCreateInfo),
        .enabledFullProfileCount = 1,
        .pEnabledFullProfiles    = &profile};
    VkDevice rawDevice;
    if (vpCreateDevice(*physicalDevice, &vpDeviceCreateInfo, nullptr, &rawDevice) != VK_SUCCESS)
        throw std::runtime_error("failed to create logical device with VP_KHR_ROADMAP_2022 profile!");
    device = vk::raii::Device(physicalDevice, rawDevice);
    queue = vk::raii::Queue(device, queueIndex, 0);
}

bool Device::isDeviceSuitable(const vk::raii::PhysicalDevice &device)
{
    // Check if the device is a discrete GPU.
    bool isDiscreteGPU = device.getProperties().deviceType ==
                         vk::PhysicalDeviceType::eDiscreteGpu;

    // Check if any queue family supports graphics
    auto queueFamilies = device.getQueueFamilyProperties();
    bool supportsGraphics =
        std::ranges::any_of(queueFamilies, [](auto const &qfp)
                            { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

    // Check that app-specific extensions (e.g. VK_KHR_swapchain) are present —
    // these are not part of the profile.
    auto availableDeviceExtensions =
        device.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions = std::ranges::all_of(
        requiredDeviceExtensions,
        [&availableDeviceExtensions](auto const &ext)
        {
            return std::ranges::any_of(
                availableDeviceExtensions,
                [ext](auto const &availableDeviceExtension)
                {
                    return strcmp(availableDeviceExtension.extensionName,
                                  ext) == 0;
                });
        });

    // VP_KHR_ROADMAP_2022 guarantees Vulkan 1.3, dynamicRendering,
    // synchronization2, runtimeDescriptorArray, descriptorBindingPartiallyBound,
    // and the other features we need — replacing the manual feature checks above.
    constexpr VpProfileProperties profile{VP_KHR_ROADMAP_2022_NAME,
                                          VP_KHR_ROADMAP_2022_SPEC_VERSION};
    VkBool32 profileSupported = VK_FALSE;
    vpGetPhysicalDeviceProfileSupport(*instance, *device, &profile, &profileSupported);

    return profileSupported && supportsGraphics &&
           supportsAllRequiredExtensions && isDiscreteGPU;
}

vk::SampleCountFlagBits Device::getMaxUsableSampleCount()
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

uint32_t Device::findMemoryType(uint32_t typeFilter,
                                vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
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

vk::Format Device::findSupportedFormat(const std::vector<vk::Format> &candidates,
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

vk::Format Device::findDepthFormat()
{
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

vk::raii::ImageView Device::createImageView(vk::Image image, vk::Format format,
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

std::vector<const char *> Device::getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (!validationLayers.empty())
    {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Device::debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return vk::False;
}

// ─── GPU upload utilities ──────────────────────────────────────────────────

// eTransient: hint to the driver that command buffers will be short-lived.
void Device::createUploadCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eTransient,
        .queueFamilyIndex = graphicsIndex};
    uploadCommandPool_ = vk::raii::CommandPool(device, poolInfo);
}

void Device::createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                          vk::SampleCountFlagBits numSamples, vk::Format format,
                          vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                          vk::MemoryPropertyFlags properties,
                          vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory)
{
    vk::ImageCreateInfo imageInfo{
        .imageType   = vk::ImageType::e2D,
        .format      = format,
        .extent      = {width, height, 1},
        .mipLevels   = mipLevels,
        .arrayLayers = 1,
        .samples     = numSamples,
        .tiling      = tiling,
        .usage       = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    // Use locals so that if allocateMemory throws, the new VkImage is destroyed
    // by its local RAII destructor and the output params are left unchanged.
    vk::raii::Image newImage(device, imageInfo);
    vk::MemoryRequirements memReqs = newImage.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize  = memReqs.size,
        .memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties)};
    vk::raii::DeviceMemory newMemory(device, allocInfo);
    newImage.bindMemory(newMemory, 0);

    image       = std::move(newImage);
    imageMemory = std::move(newMemory);
}

void Device::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                           vk::MemoryPropertyFlags properties,
                           vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory)
{
    vk::BufferCreateInfo bufferInfo{
        .size        = size,
        .usage       = usage,
        .sharingMode = vk::SharingMode::eExclusive};
    vk::raii::Buffer newBuffer(device, bufferInfo);
    vk::MemoryRequirements memReqs = newBuffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize  = memReqs.size,
        .memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties)};
    vk::raii::DeviceMemory newMemory(device, allocInfo);
    newBuffer.bindMemory(*newMemory, 0);

    buffer       = std::move(newBuffer);
    bufferMemory = std::move(newMemory);
}

std::unique_ptr<vk::raii::CommandBuffer> Device::beginSingleTimeCommands()
{
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool        = uploadCommandPool_,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1};
    auto cmd = std::make_unique<vk::raii::CommandBuffer>(
        std::move(vk::raii::CommandBuffers(device, allocInfo).front()));
    cmd->begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return cmd;
}

void Device::endSingleTimeCommands(vk::raii::CommandBuffer& cmd)
{
    cmd.end();
    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd};
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

void Device::copyBuffer(vk::raii::Buffer& src, vk::raii::Buffer& dst, vk::DeviceSize size)
{
    auto cmd = beginSingleTimeCommands();
    cmd->copyBuffer(src, dst, vk::BufferCopy(0, 0, size));
    endSingleTimeCommands(*cmd);
}

void Device::copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image,
                                uint32_t width, uint32_t height)
{
    auto cmd = beginSingleTimeCommands();
    vk::BufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageOffset       = {0, 0, 0},
        .imageExtent       = {width, height, 1}};
    cmd->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
    endSingleTimeCommands(*cmd);
}

void Device::transitionImageLayout(const vk::raii::Image& image,
                                    vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    auto cmd = beginSingleTimeCommands();

    vk::AccessFlags2        srcAccessMask, dstAccessMask;
    vk::PipelineStageFlags2 srcStageMask,  dstStageMask;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe;
        dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
        dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
        srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader;
        dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    }
    else
    {
        throw std::runtime_error("unsupported layout transition");
    }

    vk::ImageMemoryBarrier2 barrier{
        .srcStageMask        = srcStageMask,
        .srcAccessMask       = srcAccessMask,
        .dstStageMask        = dstStageMask,
        .dstAccessMask       = dstAccessMask,
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = *image,
        .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0,
                                vk::RemainingMipLevels, 0, 1}};

    cmd->pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    endSingleTimeCommands(*cmd);
}