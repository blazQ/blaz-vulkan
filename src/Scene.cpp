#include "Scene.hpp"

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeCube(glm::vec3 color, float size)
{
    float h = size * 0.5f;
    std::vector<Vertex> verts = {
        // +Z face (front)
        {{-h, -h, h}, color, {0, 0}, {0, 0, 1}},
        {{h, -h, h}, color, {1, 0}, {0, 0, 1}},
        {{h, h, h}, color, {1, 1}, {0, 0, 1}},
        {{-h, h, h}, color, {0, 1}, {0, 0, 1}},
        // -Z face (back)
        {{h, -h, -h}, color, {0, 0}, {0, 0, -1}},
        {{-h, -h, -h}, color, {1, 0}, {0, 0, -1}},
        {{-h, h, -h}, color, {1, 1}, {0, 0, -1}},
        {{h, h, -h}, color, {0, 1}, {0, 0, -1}},
        // +X face (right)
        {{h, -h, h}, color, {0, 0}, {1, 0, 0}},
        {{h, -h, -h}, color, {1, 0}, {1, 0, 0}},
        {{h, h, -h}, color, {1, 1}, {1, 0, 0}},
        {{h, h, h}, color, {0, 1}, {1, 0, 0}},
        // -X face (left)
        {{-h, -h, -h}, color, {0, 0}, {-1, 0, 0}},
        {{-h, -h, h}, color, {1, 0}, {-1, 0, 0}},
        {{-h, h, h}, color, {1, 1}, {-1, 0, 0}},
        {{-h, h, -h}, color, {0, 1}, {-1, 0, 0}},
        // +Y face (top)
        {{-h, h, h}, color, {0, 0}, {0, 1, 0}},
        {{h, h, h}, color, {1, 0}, {0, 1, 0}},
        {{h, h, -h}, color, {1, 1}, {0, 1, 0}},
        {{-h, h, -h}, color, {0, 1}, {0, 1, 0}},
        // -Y face (bottom)
        {{-h, -h, -h}, color, {0, 0}, {0, -1, 0}},
        {{h, -h, -h}, color, {1, 0}, {0, -1, 0}},
        {{h, -h, h}, color, {1, 1}, {0, -1, 0}},
        {{-h, -h, h}, color, {0, 1}, {0, -1, 0}},
    };
    std::vector<uint32_t> idxs;
    for (uint32_t f = 0; f < 6; ++f)
    {
        uint32_t b = f * 4;
        idxs.insert(idxs.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    return {verts, idxs};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makePlane(glm::vec3 color, float size)
{
    float h = size * 0.5f;
    std::vector<Vertex> verts = {
        {{-h, -h, 0.0f}, color, {0, 0}, {0, 0, 1}},
        {{h, -h, 0.0f}, color, {1, 0}, {0, 0, 1}},
        {{h, h, 0.0f}, color, {1, 1}, {0, 0, 1}},
        {{-h, h, 0.0f}, color, {0, 1}, {0, 0, 1}},
    };
    std::vector<uint32_t> idxs = {0, 1, 2, 0, 2, 3};
    return {verts, idxs};
}
