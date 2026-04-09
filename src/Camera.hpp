#pragma once
#include <glm/glm.hpp>

struct GLFWwindow; // forward declaration — avoids pulling GLFW into every TU

class Camera
{
public:
    glm::vec3 position  = {4.0f, 4.0f, 4.0f};
    float yaw            = -135.0f; // horizontal angle in degrees; -135° faces roughly toward origin
    float pitch          = -35.0f;  // vertical angle in degrees; negative = looking down

    float speed       = 5.0f;  // world units per second
    float sensitivity = 0.1f;  // degrees of rotation per pixel of mouse movement

    // Direction the camera is facing, derived from yaw and pitch.
    glm::vec3 forward() const;

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;

    // Apply mouse delta (pixels) and keyboard state to update position/orientation.
    // Call once per frame only when the camera is in fly mode.
    void processInput(GLFWwindow *window, float dt, glm::vec2 mouseDelta);

    void drawImGui();
};
