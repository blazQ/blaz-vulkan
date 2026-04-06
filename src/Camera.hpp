#pragma once
#include <glm/glm.hpp>

class Camera
{
public:
    glm::vec3 position = {4.0f, 4.0f, 4.0f};

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;
    void drawImGui();
};
