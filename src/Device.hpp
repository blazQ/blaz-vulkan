#pragma once

#include <vector>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class Device
{
public:
    Device(GLFWwindow *window, std::vector<const char *> requiredDeviceExtensions, std::vector<const char *> validationLayers);

    vk::raii::Instance &getInstance();
    vk::raii::SurfaceKHR &getSurface();
    vk::raii::PhysicalDevice &getPhysicalDevice();
    vk::raii::Device &getLogicalDevice();
    vk::raii::Queue &getGraphicsQueue();
    uint32_t getGraphicsIndex() const;

    vk::SampleCountFlagBits getMsaaSamples() const;
    vk::SampleCountFlagBits getMaxMsaaSamples() const;
    vk::SampleCountFlags getSupportedMsaaSamples() const;
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    vk::Format findDepthFormat();
    vk::raii::ImageView createImageView(vk::Image image, vk::Format format,
                                                   vk::ImageAspectFlags aspectFlags,
                                                   uint32_t mipLevels);

private:
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue queue = nullptr;
    uint32_t graphicsIndex;

    vk::SampleCountFlagBits msaaSamples;
    vk::SampleCountFlagBits maxMsaaSamples;
    vk::SampleCountFlags supportedMsaaSamples;
    std::vector<const char *> requiredDeviceExtensions;
    std::vector<const char *> validationLayers;

    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow *window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    bool isDeviceSuitable(const vk::raii::PhysicalDevice &device);

    vk::SampleCountFlagBits getMaxUsableSampleCount();
    std::vector<const char *> getRequiredExtensions();
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageType,
        const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData);
};