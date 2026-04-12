#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

    // GPU-side texture: owns the image, its memory, a view, and a sampler.
    struct Texture
    {
        vk::raii::Image image = nullptr;
        vk::raii::DeviceMemory memory = nullptr;
        vk::raii::ImageView view = nullptr;
        vk::raii::Sampler sampler = nullptr;
    };

    // GPU-side mesh instance: owns Vulkan buffers, a model matrix, and a
    // texture index into the global bindless texture array.
    // The decomposed transform (position/rotationDeg/scale) is kept alongside
    // the baked matrix so the ImGui sliders can edit them at runtime.
    struct Renderable
    {
        vk::raii::Buffer vertexBuffer = nullptr;
        vk::raii::DeviceMemory vertexBufferMemory = nullptr;
        vk::raii::Buffer indexBuffer = nullptr;
        vk::raii::DeviceMemory indexBufferMemory = nullptr;
        uint32_t indexCount = 0;
        glm::mat4 modelMatrix      = glm::mat4(1.0f);
        uint32_t  textureIndex     = 0xFFFFu;
        uint32_t  specularMapIndex = 0xFFFFu;
        uint32_t  normalMapIndex   = 0xFFFFu;
        uint32_t  heightMapIndex   = 0xFFFFu;

        std::string   label;
        glm::vec3     position    = {0.0f, 0.0f, 0.0f};
        glm::vec3     rotationDeg = {0.0f, 0.0f, 0.0f}; // XYZ Euler, degrees
        float         scale       = 1.0f;
    };

    // Push constants sent per draw call: model matrix + which textures to use.
    struct PushConstants
    {
        glm::mat4 model;
        uint32_t  textureIndex;
        uint32_t  specularMapIndex; // 0xFFFF = no specular map, use global specStrength
        uint32_t  normalMapIndex;   // 0xFFFF = use geometric normal
        uint32_t  heightMapIndex;   // 0xFFFF = no height map, skip POM
    };

    // Data layout of the uniform buffer as the shader sees it.
    // All fields are vec4-aligned — do not insert scalars between them.
    // Must stay in sync with SkyUBO in sky.slang.
    struct UniformBufferObject
    {
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 lightSpaceMatrix;
        glm::vec4 lightDir;
        glm::vec4 cameraPos;
        glm::vec4 materialParams;        // x=ambient, y=specStrength, z=shininess, w=exposure
        glm::vec4 pointLightPos[4];      // xyz=world position, w=intensity
        glm::vec4 pointLightColor[4];    // xyz=color, w=radius (falloff distance)
        glm::vec4 lightCounts;           // x=number of active point lights
        glm::vec4 shadowParams;          // x=biasMin, y=biasMax (slope-scale shadow bias range)
        glm::vec4 pomParams;             // x=depthScale, y=minSteps, z=maxSteps
        // Fog
        glm::vec4 fogParams;             // x=density, y=heightFalloff, z=maxOpacity
        glm::vec4 fogColor;              // xyz=linear RGB fog color
        // Sky ray reconstruction (used by sky.slang)
        glm::mat4 invProj;               // inverse of proj
        glm::mat4 invViewRot;            // inverse of view rotation (translation zeroed)
    };

    // Sky appearance pushed per-frame. Must match SkyPush in sky.slang.
    struct SkyPushConstants
    {
        glm::vec4 horizonColor = {0.60f, 0.75f, 0.95f, 1.0f};
        glm::vec4 zenithColor  = {0.10f, 0.30f, 0.75f, 1.0f};
        glm::vec4 groundColor  = {0.20f, 0.15f, 0.10f, 1.0f};
        // xyz = sun color (HDR — can exceed 1.0), w = cos(half-angle of sun disk)
        glm::vec4 sunParams    = {1.5f,  1.3f,  1.0f,  0.9998f};
    };

    // CPU-side point light description, mirrored into the UBO each frame.
    struct PointLightData
    {
        glm::vec3 position  = {0.0f, 0.0f, 3.0f};
        float     intensity = 3.0f;
        glm::vec3 color     = {1.0f, 1.0f, 1.0f};
        float     radius    = 8.0f;
        bool      enabled   = false;
    };

    // -------------------------------------------------------------------------
    // Member variables
    // -------------------------------------------------------------------------

    // Base path: directory containing the executable. Used to resolve all
    // asset paths (shaders, scenes, textures) so the binary can be invoked
    // from any working directory.
    std::filesystem::path basePath_;

    // Window
    GLFWwindow *window = nullptr;
    bool framebufferResized = false;

    // Vulkan core
    std::unique_ptr<Device> vulkanDevice;
    std::unique_ptr<Swapchain> swapchain;

    // Scene pipeline
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline shadowPipeline = nullptr;

    // Sky pipeline
    vk::raii::DescriptorSetLayout skyDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout      skyPipelineLayout      = nullptr;
    vk::raii::Pipeline            skyPipeline            = nullptr;
    vk::raii::DescriptorPool      skyDescriptorPool      = nullptr;
    std::vector<vk::raii::DescriptorSet> skyDescriptorSets;
    SkyPushConstants skyPush;
    bool             skyEnabled = true;

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

    // Bindless texture array — all loaded textures live here.
    // Each Renderable holds an index into this vector.
    static constexpr uint32_t MAX_TEXTURES = 2048;
    std::vector<Texture> textures;
    std::unordered_map<std::string, uint32_t> textureCache;

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
    bool      cameraMode    = false;  // true while RMB is held and cursor is captured
    glm::vec2 lastMousePos  = {0.0f, 0.0f};

    // Lighting / material
    float ambient          = 0.2f;
    float specularStrength = 0.5f;
    float shininess        = 32.0f;
    float exposure         = 1.0f;
    bool  tonemapping      = true;

    // Shadow bias: lerp(biasMin, biasMax, NdotL) — more bias at grazing angles.
    // Too high → peter panning (shadow detaches from caster).
    // Too low  → shadow acne (surface self-shadows with noise).
    float shadowBiasMin = 0.0005f;
    float shadowBiasMax = 0.003f;
    float shadowOrthoSize = 20.0f;  // half-extent of the light ortho frustum
    float shadowNear      = 0.1f;
    float shadowFar       = 100.0f;

    // Parallax Occlusion Mapping
    float pomDepthScale = 0.05f;   // how tall the displacement appears (world units)
    float pomMinSteps   = 8.0f;    // steps when looking straight at the surface
    float pomMaxSteps   = 32.0f;   // steps at grazing angles

    // Fog
    bool      fogEnabled      = true;
    float     fogDensity      = 0.02f;         // higher = thicker, shorter visibility
    float     fogHeightFalloff= 0.3f;          // higher = fog thins out faster with altitude
    float     fogMaxOpacity   = 1.0f;          // clamp fog factor (< 1 keeps distant objects visible)
    bool      fogSyncSky      = true;          // keep fog color locked to sky horizon color
    glm::vec3 fogColor        = {0.60f, 0.75f, 0.95f};

    // Light animation
    float lightAngle = 0.0f;
    bool lightOrbit = true;
    float prevTime = 0.0f;

    // Point lights (no shadow casting).
    // Loaded from scene.json "pointLights" array; can also be added/removed at runtime via ImGui.
    static constexpr int MAX_POINT_LIGHTS = 4;
    std::vector<PointLightData> pointLights = {
        PointLightData{{-4.0f,  3.0f, 4.0f}, 4.0f, {1.0f, 0.85f, 0.5f},  8.0f, true},  // warm key
        PointLightData{{ 4.0f, -3.0f, 3.0f}, 3.0f, {0.4f, 0.6f,  1.0f}, 10.0f, true},  // cool fill
    };

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
    static std::array<vk::VertexInputAttributeDescription, 5> getVertexAttributeDescriptions();

    // Window
    void initWindow();
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    // Vulkan setup
    void initVulkan();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createShadowPipeline();
    void createSkyDescriptorSetLayout();
    void createSkyPipeline();
    void createSkyDescriptorSets();
    std::vector<char> readFile(const std::filesystem::path &path);
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
    uint32_t loadTexture(const std::filesystem::path &path, bool linearFormat = false);
    uint32_t loadTextureFromMemory(const std::vector<uint8_t> &bytes, bool linearFormat = false);
    void generateMipmaps(vk::raii::Image &image, vk::Format imageFormat,
                         int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

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
    // Records an image memory barrier into the current frame's command buffer.
    // Unlike transitionImageLayout() (which allocates its own one-shot command buffer),
    // this version is called during recordCommandBuffer() for in-flight transitions.
    void recordImageBarrier(vk::Image image,
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
