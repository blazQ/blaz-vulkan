#pragma once

#include "Device.hpp"
#include <vector>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class Swapchain
{
public:
    Swapchain(Device& device, GLFWwindow* window, vk::PresentModeKHR preferredMode);

    vk::raii::SwapchainKHR& getSwapChain();
    std::vector<vk::Image>& getImages();
    std::vector<vk::raii::ImageView>& getImageViews();
    const vk::SurfaceFormatKHR& getSurfaceFormat() const;
    const vk::Extent2D& getExtent() const;
    vk::PresentModeKHR getPresentMode() const;

    void recreate(GLFWwindow* window, vk::PresentModeKHR preferredMode);

private:
    Device& device;
    GLFWwindow* window;
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;
    vk::PresentModeKHR presentMode;

    void createSwapChain(vk::PresentModeKHR preferredMode);
    void createImageViews();

    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats);
    static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes);
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities);
};
