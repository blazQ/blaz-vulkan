#pragma once
#include <utility>
#include <vector>
#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;
};

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeCube(glm::vec3 color, float size);
std::pair<std::vector<Vertex>, std::vector<uint32_t>> makePlane(glm::vec3 color, float size);
