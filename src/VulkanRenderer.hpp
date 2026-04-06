#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Camera.hpp"
#include "Device.hpp"
#include "Scene.hpp"
#include "Swapchain.hpp"

class VulkanRenderer
{
public:
    void run();

private:
    // -------------------------------------------------------------------------
    // Nested types
    // -------------------------------------------------------------------------

    // GPU-side mesh instance: owns Vulkan buffers and a model matrix.
    struct Renderable
    {
        vk::raii::Buffer vertexBuffer = nullptr;
        vk::raii::DeviceMemory vertexBufferMemory = nullptr;
        vk::raii::Buffer indexBuffer = nullptr;
        vk::raii::DeviceMemory indexBufferMemory = nullptr;
        uint32_t indexCount = 0;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
    };

    // Data layout of the uniform buffer as the shader sees it.
    struct UniformBufferObject
    {
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 lightSpaceMatrix;
        glm::vec4 lightDir;
        glm::vec4 cameraPos;
        glm::vec4 materialParams; // x=ambient, y=specularStrength, z=shininess
    };

    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------

    // Window
    GLFWwindow *window = nullptr;
    bool framebufferResized = false;

    // Vulkan core
    std::unique_ptr<Device> vulkanDevice;
    std::unique_ptr<Swapchain> swapchain;

    // Pipelines
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline shadowPipeline = nullptr;
    vk::PresentModeKHR pendingPresentMode = vk::PresentModeKHR::eMailbox;

    // Command recording
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    uint32_t frameIndex = 0;

    // Render attachments (MSAA + depth)
    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
    vk::SampleCountFlagBits pendingMsaaSamples = vk::SampleCountFlagBits::e1;
    vk::raii::Image colorImage = nullptr;
    vk::raii::DeviceMemory colorImageMemory = nullptr;
    vk::raii::ImageView colorImageView = nullptr;
    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    // Texture
    uint32_t mipLevels = 0;
    vk::raii::Image textureImage = nullptr;
    vk::raii::DeviceMemory textureImageMemory = nullptr;
    vk::raii::ImageView textureImageView = nullptr;
    vk::raii::Sampler textureSampler = nullptr;

    // Shadow map
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;
    vk::raii::Image shadowMapImage = nullptr;
    vk::raii::DeviceMemory shadowMapImageMemory = nullptr;
    vk::raii::ImageView shadowMapImageView = nullptr;
    vk::raii::Sampler shadowMapSampler = nullptr;

    // Scene
    std::vector<Renderable> renderables;

    // Uniform buffers (one per frame in flight)
    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    // Camera
    Camera camera;

    // Lighting / material
    float ambient = 0.2f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;

    // Light animation
    float lightAngle = 0.0f;
    bool lightOrbit = true;
    float prevTime = 0.0f;

    // Descriptors
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    // Synchronization
    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    // ImGui
    vk::raii::DescriptorPool imguiDescriptorPool = nullptr;

    // -------------------------------------------------------------------------
    // Private methods
    // -------------------------------------------------------------------------

    // Vertex format helpers (kept here because they return Vulkan types)
    static vk::VertexInputBindingDescription getVertexBindingDescription();
    static std::array<vk::VertexInputAttributeDescription, 4> getVertexAttributeDescriptions();

    // Window
    void initWindow();
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    // Vulkan setup
    void initVulkan();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createShadowPipeline();
    static std::vector<char> readFile(const std::string &filename);
    vk::raii::ShaderModule createShaderModule(const std::vector<char> &code);
    static bool hasStencilComponent(vk::Format format);

    // Command infrastructure
    void createCommandPool();
    void createCommandBuffers();

    // Render attachments
    void createColorResources();
    void createDepthResources();
    void createShadowMapResources();
    void createShadowMapSampler();

    // GPU resource utilities
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                     vk::SampleCountFlagBits numSamples, vk::Format format,
                     vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                     vk::MemoryPropertyFlags properties,
                     vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory);
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags properties,
                      vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory);
    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();
    void endSingleTimeCommands(vk::raii::CommandBuffer &commandBuffer);
    void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size);
    void copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image,
                           uint32_t width, uint32_t height);
    void transitionImageLayout(const vk::raii::Image &image,
                               vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

    // Texture
    void createTextureImage();
    void generateMipmaps(vk::raii::Image &image, vk::Format imageFormat,
                         int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
    void createTextureImageView();
    void createTextureSampler();

    // Scene
    void uploadRenderable(Renderable &r, const std::vector<Vertex> &verts,
                          const std::vector<uint32_t> &idxs);
    void loadScene();
    void createUniformBuffers();

    // Descriptors
    void createDescriptorPool();
    void createDescriptorSets();

    // Synchronization
    void createSyncObjects();

    // Per-frame rendering
    void mainLoop();
    void drawFrame();
    void updateUniformBuffer(uint32_t currentImage);
    void recordCommandBuffer(uint32_t imageIndex);
    void transition_image_layout(vk::Image image,
                                 vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                 vk::AccessFlags2 srcAccessMask, vk::AccessFlags2 dstAccessMask,
                                 vk::PipelineStageFlags2 srcStageMask,
                                 vk::PipelineStageFlags2 dstStageMask,
                                 vk::ImageAspectFlags imageAspectFlags);

    // Lifecycle
    void recreateSwapChain();
    void rebuildMsaa();
    void cleanupSwapChain();
    void cleanup();

    // ImGui
    void drawImGui();
    void imGuiInit();
};
