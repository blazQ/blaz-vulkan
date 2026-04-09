#include "Camera.hpp"

#include <cmath>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui.h"

glm::vec3 Camera::forward() const
{
    // Z-up spherical coordinates:
    //   yaw   = horizontal rotation around Z axis (0° = +X, 90° = +Y)
    //   pitch = elevation above the XY plane (positive = up, clamped to ±89°)
    float y = glm::radians(yaw);
    float p = glm::radians(pitch);
    return glm::normalize(glm::vec3{
        std::cos(y) * std::cos(p),
        std::sin(y) * std::cos(p),
        std::sin(p)
    });
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, position + forward(), glm::vec3(0.0f, 0.0f, 1.0f));
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const
{
    glm::mat4 proj = glm::perspective(glm::radians(75.0f), aspectRatio, 0.1f, 500.0f);
    proj[1][1] *= -1; // Vulkan Y-axis flip
    return proj;
}

void Camera::processInput(GLFWwindow *window, float dt, glm::vec2 mouseDelta)
{
    // Apply mouse look. Y delta is negated so dragging up = looking up.
    yaw   -= mouseDelta.x * sensitivity;
    pitch -= mouseDelta.y * sensitivity;
    pitch  = glm::clamp(pitch, -89.0f, 89.0f);

    // Derive right vector from forward × world-up (Z-up world).
    // Cross product with (0,0,1) gives the horizontal right vector.
    glm::vec3 fwd   = forward();
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 0.0f, 1.0f)));
    glm::vec3 up    = glm::vec3(0.0f, 0.0f, 1.0f);

    float v = speed * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) position += fwd   * v;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) position -= fwd   * v;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) position -= right * v;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) position += right * v;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) position += up    * v; // ascend
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) position -= up    * v; // descend
}

void Camera::drawImGui()
{
    ImGui::Text("Position: (%.2f, %.2f, %.2f)", position.x, position.y, position.z);
    ImGui::Text("Yaw: %.1f  Pitch: %.1f", yaw, pitch);
    ImGui::SliderFloat("Fly speed",    &speed,       0.5f, 50.0f);
    ImGui::SliderFloat("Mouse sensitivity", &sensitivity, 0.01f, 1.0f);
    ImGui::TextDisabled("Hold RMB to fly  |  WASD move  |  Q/E up/down");
}
