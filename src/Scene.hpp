#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;
    glm::vec4 tangent; // xyz = tangent direction in world space, w = bitangent sign (+1 or -1)

    bool operator==(const Vertex &o) const
    {
        return pos == o.pos && color == o.color &&
               texCoord == o.texCoord && normal == o.normal &&
               tangent == o.tangent;
    }
};

struct GltfImage {
    std::string          path;  // non-empty = external file
    std::vector<uint8_t> bytes; // non-empty = embedded GLB image
};

struct GltfPrimitive {
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    glm::mat4             transform;
    GltfImage             baseColor;
    GltfImage             normalMap;
    GltfImage             metallicRoughness;
};



// Hash for Vertex — needed by loadOBJ to deduplicate vertices.
template <>
struct std::hash<Vertex>
{
    size_t operator()(const Vertex &v) const noexcept
    {
        // Combine hashes of every field. The magic constant is from Boost.
        auto h = [](size_t seed, size_t val) {
            return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        };
        auto fh = [](float f) { return std::hash<float>{}(f); };
        size_t s = 0;
        s = h(s, fh(v.pos.x));      s = h(s, fh(v.pos.y));      s = h(s, fh(v.pos.z));
        s = h(s, fh(v.color.x));    s = h(s, fh(v.color.y));    s = h(s, fh(v.color.z));
        s = h(s, fh(v.texCoord.x)); s = h(s, fh(v.texCoord.y));
        s = h(s, fh(v.normal.x));   s = h(s, fh(v.normal.y));   s = h(s, fh(v.normal.z));
        s = h(s, fh(v.tangent.x));  s = h(s, fh(v.tangent.y));  s = h(s, fh(v.tangent.z)); s = h(s, fh(v.tangent.w));
        return s;
    }
};

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeCube(glm::vec3 color, float size);
std::pair<std::vector<Vertex>, std::vector<uint32_t>> makePlane(glm::vec3 color, float size);
// size = radius. sectors = longitude divisions, stacks = latitude divisions.
// Normals point outward; tangents follow the direction of increasing longitude.
std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeSphere(glm::vec3 color, float radius,
                                                                   uint32_t sectors = 32,
                                                                   uint32_t stacks  = 16);
std::pair<std::vector<Vertex>, std::vector<uint32_t>> loadOBJ(const std::string &path, bool yUpToZUp = false);
std::vector<GltfPrimitive> loadGLTF(const std::string& path, bool yUpToZUp = false);