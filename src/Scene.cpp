#include "Scene.hpp"

#include <stdexcept>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeCube(glm::vec3 color, float size)
{
    float h = size * 0.5f;
    // Tangent = direction U increases along this face. w = bitangent sign (+1).
    // Bitangent = cross(normal, tangent) * w, reconstructed in the shader.
    std::vector<Vertex> verts = {
        // +Z face (front)  normal=(0,0,1)  U goes along +X → tangent=(1,0,0)
        {{-h, -h, h}, color, {0, 0}, {0, 0, 1}, {1, 0, 0, 1}},
        {{ h, -h, h}, color, {1, 0}, {0, 0, 1}, {1, 0, 0, 1}},
        {{ h,  h, h}, color, {1, 1}, {0, 0, 1}, {1, 0, 0, 1}},
        {{-h,  h, h}, color, {0, 1}, {0, 0, 1}, {1, 0, 0, 1}},
        // -Z face (back)   normal=(0,0,-1) U goes along -X → tangent=(-1,0,0)
        {{ h, -h, -h}, color, {0, 0}, {0, 0, -1}, {-1, 0, 0, 1}},
        {{-h, -h, -h}, color, {1, 0}, {0, 0, -1}, {-1, 0, 0, 1}},
        {{-h,  h, -h}, color, {1, 1}, {0, 0, -1}, {-1, 0, 0, 1}},
        {{ h,  h, -h}, color, {0, 1}, {0, 0, -1}, {-1, 0, 0, 1}},
        // +X face (right)  normal=(1,0,0)  U goes along -Z → tangent=(0,0,-1)
        {{h, -h,  h}, color, {0, 0}, {1, 0, 0}, {0, 0, -1, 1}},
        {{h, -h, -h}, color, {1, 0}, {1, 0, 0}, {0, 0, -1, 1}},
        {{h,  h, -h}, color, {1, 1}, {1, 0, 0}, {0, 0, -1, 1}},
        {{h,  h,  h}, color, {0, 1}, {1, 0, 0}, {0, 0, -1, 1}},
        // -X face (left)   normal=(-1,0,0) U goes along +Z → tangent=(0,0,1)
        {{-h, -h, -h}, color, {0, 0}, {-1, 0, 0}, {0, 0, 1, 1}},
        {{-h, -h,  h}, color, {1, 0}, {-1, 0, 0}, {0, 0, 1, 1}},
        {{-h,  h,  h}, color, {1, 1}, {-1, 0, 0}, {0, 0, 1, 1}},
        {{-h,  h, -h}, color, {0, 1}, {-1, 0, 0}, {0, 0, 1, 1}},
        // +Y face (top)    normal=(0,1,0)  U goes along +X → tangent=(1,0,0)
        {{-h, h,  h}, color, {0, 0}, {0, 1, 0}, {1, 0, 0, 1}},
        {{ h, h,  h}, color, {1, 0}, {0, 1, 0}, {1, 0, 0, 1}},
        {{ h, h, -h}, color, {1, 1}, {0, 1, 0}, {1, 0, 0, 1}},
        {{-h, h, -h}, color, {0, 1}, {0, 1, 0}, {1, 0, 0, 1}},
        // -Y face (bottom) normal=(0,-1,0) U goes along +X → tangent=(1,0,0)
        {{-h, -h, -h}, color, {0, 0}, {0, -1, 0}, {1, 0, 0, 1}},
        {{ h, -h, -h}, color, {1, 0}, {0, -1, 0}, {1, 0, 0, 1}},
        {{ h, -h,  h}, color, {1, 1}, {0, -1, 0}, {1, 0, 0, 1}},
        {{-h, -h,  h}, color, {0, 1}, {0, -1, 0}, {1, 0, 0, 1}},
    };
    std::vector<uint32_t> idxs;
    for (uint32_t f = 0; f < 6; ++f)
    {
        uint32_t b = f * 4;
        idxs.insert(idxs.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    return {verts, idxs};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> loadOBJ(const std::string &path)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str()))
        throw std::runtime_error("loadOBJ failed for '" + path + "': " + warn + err);

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::unordered_map<Vertex, uint32_t> uniqueVertices;

    for (const auto &shape : shapes)
    {
        for (const auto &index : shape.mesh.indices)
        {
            Vertex v{};
            v.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2],
            };
            v.color = {1.0f, 1.0f, 1.0f};
            if (index.texcoord_index >= 0)
            {
                v.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1], // flip Y
                };
            }
            if (index.normal_index >= 0)
            {
                v.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2],
                };
            }
            // tangent.w filled in during the tangent-generation pass below
            v.tangent = {0.0f, 0.0f, 0.0f, 1.0f};

            if (uniqueVertices.count(v) == 0)
            {
                uniqueVertices[v] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(v);
            }
            indices.push_back(uniqueVertices[v]);
        }
    }

    // Compute tangents per triangle and accumulate into vertices.
    // For each triangle we solve: deltaPos = deltaUV.x * T + deltaUV.y * B
    // Isolating T gives the tangent direction for that triangle. Vertices shared
    // by many triangles accumulate contributions that are normalised at the end.
    std::vector<glm::vec3> tangentAccum(vertices.size(), glm::vec3(0.0f));
    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        Vertex &v0 = vertices[indices[i]];
        Vertex &v1 = vertices[indices[i + 1]];
        Vertex &v2 = vertices[indices[i + 2]];

        glm::vec3 edge1 = v1.pos - v0.pos;
        glm::vec3 edge2 = v2.pos - v0.pos;
        glm::vec2 dUV1  = v1.texCoord - v0.texCoord;
        glm::vec2 dUV2  = v2.texCoord - v0.texCoord;

        float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
        if (std::abs(det) < 1e-6f) continue; // degenerate UV triangle, skip
        float inv = 1.0f / det;

        glm::vec3 tangent = inv * (dUV2.y * edge1 - dUV1.y * edge2);
        tangentAccum[indices[i]]     += tangent;
        tangentAccum[indices[i + 1]] += tangent;
        tangentAccum[indices[i + 2]] += tangent;
    }

    // Orthogonalise (Gram-Schmidt) and store with handedness sign in w.
    for (size_t i = 0; i < vertices.size(); i++)
    {
        glm::vec3 n = vertices[i].normal;
        glm::vec3 t = tangentAccum[i];
        // Remove the component of t that is parallel to n, then normalise.
        glm::vec3 ortho = glm::normalize(t - glm::dot(t, n) * n);
        // The sign tells the shader which way the bitangent points.
        // cross(n, t) should agree with the precomputed bitangent direction;
        // if they oppose each other the sign is -1 (mirrored UV island).
        // cross(n, t) gives the expected bitangent direction.
        // If the accumulated tangent agrees with it, sign = +1; mirrored UV → -1.
        float sign = (glm::dot(glm::cross(n, ortho), t) >= 0.0f) ? 1.0f : -1.0f;
        vertices[i].tangent = glm::vec4(ortho, sign);
    }

    return {vertices, indices};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makePlane(glm::vec3 color, float size)
{
    float h = size * 0.5f;
    // Plane in XY, normal = +Z, U goes along +X → tangent = (1,0,0,1)
    std::vector<Vertex> verts = {
        {{-h, -h, 0.0f}, color, {0, 0}, {0, 0, 1}, {1, 0, 0, 1}},
        {{ h, -h, 0.0f}, color, {1, 0}, {0, 0, 1}, {1, 0, 0, 1}},
        {{ h,  h, 0.0f}, color, {1, 1}, {0, 0, 1}, {1, 0, 0, 1}},
        {{-h,  h, 0.0f}, color, {0, 1}, {0, 0, 1}, {1, 0, 0, 1}},
    };
    std::vector<uint32_t> idxs = {0, 1, 2, 0, 2, 3};
    return {verts, idxs};
}
