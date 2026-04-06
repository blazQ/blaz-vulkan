#include "Camera.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "imgui.h"

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const
{
    glm::mat4 proj = glm::perspective(glm::radians(75.0f), aspectRatio, 0.1f, 20.0f);
    proj[1][1] *= -1; // Vulkan Y-axis flip
    return proj;
}

void Camera::drawImGui()
{
    ImGui::SliderFloat("Camera X", &position.x, -10.0f, 10.0f);
    ImGui::SliderFloat("Camera Y", &position.y, -10.0f, 10.0f);
    ImGui::SliderFloat("Camera Z", &position.z, 0.1f, 10.0f);
}
