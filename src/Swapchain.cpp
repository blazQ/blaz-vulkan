#include "Swapchain.hpp"
#include <algorithm>
#include <cassert>
#include <limits>

Swapchain::Swapchain(Device& device, GLFWwindow* window, vk::PresentModeKHR preferredMode)
    : device(device), window(window)
{
    createSwapChain(preferredMode);
    createImageViews();
}

vk::raii::SwapchainKHR& Swapchain::getSwapChain() { return swapChain; }
std::vector<vk::Image>& Swapchain::getImages() { return swapChainImages; }
std::vector<vk::raii::ImageView>& Swapchain::getImageViews() { return swapChainImageViews; }
const vk::SurfaceFormatKHR& Swapchain::getSurfaceFormat() const { return swapChainSurfaceFormat; }
const vk::Extent2D& Swapchain::getExtent() const { return swapChainExtent; }
vk::PresentModeKHR Swapchain::getPresentMode() const { return presentMode; }

void Swapchain::recreate(GLFWwindow* win, vk::PresentModeKHR preferredMode)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(win, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(win, &width, &height);
        glfwWaitEvents();
    }

    device.getLogicalDevice().waitIdle();

    swapChainImageViews.clear();
    swapChain = nullptr;

    window = win;
    createSwapChain(preferredMode);
    createImageViews();
}

void Swapchain::createSwapChain(vk::PresentModeKHR preferredMode)
{
    vk::SurfaceCapabilitiesKHR surfaceCapabilities =
        device.getPhysicalDevice().getSurfaceCapabilitiesKHR(*device.getSurface());
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats =
        device.getPhysicalDevice().getSurfaceFormatsKHR(*device.getSurface());
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

    std::vector<vk::PresentModeKHR> availablePresentModes =
        device.getPhysicalDevice().getSurfacePresentModesKHR(*device.getSurface());
    if (!std::ranges::any_of(availablePresentModes,
                              [preferredMode](auto m) { return m == preferredMode; }))
        preferredMode = chooseSwapPresentMode(availablePresentModes);
    presentMode = preferredMode;

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .surface = *device.getSurface(),
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

    swapChain = vk::raii::SwapchainKHR(device.getLogicalDevice(), swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
}

uint32_t Swapchain::chooseSwapMinImageCount(
    vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
{
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) &&
        (surfaceCapabilities.maxImageCount < minImageCount))
    {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

vk::SurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(
    std::vector<vk::SurfaceFormatKHR> const& availableFormats)
{
    assert(!availableFormats.empty());
    const auto formatIt =
        std::ranges::find_if(availableFormats, [](const auto& format)
                             { return format.format == vk::Format::eB8G8R8A8Srgb &&
                                      format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR Swapchain::chooseSwapPresentMode(
    std::vector<vk::PresentModeKHR> const& availablePresentModes)
{
    assert(std::ranges::any_of(availablePresentModes, [](auto m)
                               { return m == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(availablePresentModes,
                               [](const vk::PresentModeKHR value)
                               { return vk::PresentModeKHR::eMailbox == value; })
               ? vk::PresentModeKHR::eMailbox
               : vk::PresentModeKHR::eFifo;
}

vk::Extent2D Swapchain::chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
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

void Swapchain::createImageViews()
{
    swapChainImageViews.reserve(swapChainImages.size());
    for (uint32_t i = 0; i < swapChainImages.size(); i++)
    {
        swapChainImageViews.push_back(
            device.createImageView(swapChainImages[i], swapChainSurfaceFormat.format,
                            vk::ImageAspectFlagBits::eColor, 1));
    }
}