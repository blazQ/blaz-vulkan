#include "Scene.hpp"

#include <stdexcept>
#include <filesystem>
#include <unordered_map>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>

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
        {{h, -h, h}, color, {1, 0}, {0, 0, 1}, {1, 0, 0, 1}},
        {{h, h, h}, color, {1, 1}, {0, 0, 1}, {1, 0, 0, 1}},
        {{-h, h, h}, color, {0, 1}, {0, 0, 1}, {1, 0, 0, 1}},
        // -Z face (back)   normal=(0,0,-1) U goes along -X → tangent=(-1,0,0)
        {{h, -h, -h}, color, {0, 0}, {0, 0, -1}, {-1, 0, 0, 1}},
        {{-h, -h, -h}, color, {1, 0}, {0, 0, -1}, {-1, 0, 0, 1}},
        {{-h, h, -h}, color, {1, 1}, {0, 0, -1}, {-1, 0, 0, 1}},
        {{h, h, -h}, color, {0, 1}, {0, 0, -1}, {-1, 0, 0, 1}},
        // +X face (right)  normal=(1,0,0)  U goes along -Z → tangent=(0,0,-1)
        {{h, -h, h}, color, {0, 0}, {1, 0, 0}, {0, 0, -1, 1}},
        {{h, -h, -h}, color, {1, 0}, {1, 0, 0}, {0, 0, -1, 1}},
        {{h, h, -h}, color, {1, 1}, {1, 0, 0}, {0, 0, -1, 1}},
        {{h, h, h}, color, {0, 1}, {1, 0, 0}, {0, 0, -1, 1}},
        // -X face (left)   normal=(-1,0,0) U goes along +Z → tangent=(0,0,1)
        {{-h, -h, -h}, color, {0, 0}, {-1, 0, 0}, {0, 0, 1, 1}},
        {{-h, -h, h}, color, {1, 0}, {-1, 0, 0}, {0, 0, 1, 1}},
        {{-h, h, h}, color, {1, 1}, {-1, 0, 0}, {0, 0, 1, 1}},
        {{-h, h, -h}, color, {0, 1}, {-1, 0, 0}, {0, 0, 1, 1}},
        // +Y face (top)    normal=(0,1,0)  U goes along +X → tangent=(1,0,0)
        {{-h, h, h}, color, {0, 0}, {0, 1, 0}, {1, 0, 0, 1}},
        {{h, h, h}, color, {1, 0}, {0, 1, 0}, {1, 0, 0, 1}},
        {{h, h, -h}, color, {1, 1}, {0, 1, 0}, {1, 0, 0, 1}},
        {{-h, h, -h}, color, {0, 1}, {0, 1, 0}, {1, 0, 0, 1}},
        // -Y face (bottom) normal=(0,-1,0) U goes along +X → tangent=(1,0,0)
        {{-h, -h, -h}, color, {0, 0}, {0, -1, 0}, {1, 0, 0, 1}},
        {{h, -h, -h}, color, {1, 0}, {0, -1, 0}, {1, 0, 0, 1}},
        {{h, -h, h}, color, {1, 1}, {0, -1, 0}, {1, 0, 0, 1}},
        {{-h, -h, h}, color, {0, 1}, {0, -1, 0}, {1, 0, 0, 1}},
    };
    std::vector<uint32_t> idxs;
    for (uint32_t f = 0; f < 6; ++f)
    {
        uint32_t b = f * 4;
        idxs.insert(idxs.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    return {verts, idxs};
}

// UV sphere: (stacks+1) rings × (sectors+1) vertices.
//
// Coordinate convention: Z-up. phi is the polar angle from +Z (north pole) to -Z (south pole),
// measured 0 → π. theta is the azimuthal angle around the Z axis, measured 0 → 2π.
//   pos    = radius × (sin(phi)cos(θ), sin(phi)sin(θ), cos(phi))
//   normal = pos / radius  (outward unit normal)
//   tangent= (-sin(θ), cos(θ), 0)  — direction of increasing longitude, w=+1
//   uv     = (j/sectors, i/stacks)
//
// Poles: ring 0 (north, phi=0) and ring stacks (south, phi=π) are rings of coincident
// vertices. The cap triangles that would degenerate there are skipped explicitly.
std::pair<std::vector<Vertex>, std::vector<uint32_t>>
makeSphere(glm::vec3 color, float radius, uint32_t sectors, uint32_t stacks)
{
    std::vector<Vertex> verts;
    std::vector<uint32_t> idxs;
    verts.reserve((stacks + 1) * (sectors + 1));

    for (uint32_t i = 0; i <= stacks; ++i)
    {
        float phi = glm::pi<float>() * static_cast<float>(i) / static_cast<float>(stacks);
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);

        for (uint32_t j = 0; j <= sectors; ++j)
        {
            float theta = glm::two_pi<float>() * static_cast<float>(j) / static_cast<float>(sectors);
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);

            glm::vec3 n = {sinPhi * cosTheta, sinPhi * sinTheta, cosPhi};

            Vertex v;
            v.pos = n * radius;
            v.normal = n;
            v.color = color;
            v.texCoord = {static_cast<float>(j) / static_cast<float>(sectors),
                          static_cast<float>(i) / static_cast<float>(stacks)};
            // Tangent: direction of increasing theta (east along the surface).
            // At the poles sinPhi≈0 so the tangent magnitude is tiny, but the direction
            // is still well-defined and numerically fine for lighting purposes.
            v.tangent = {-sinTheta, cosTheta, 0.0f, 1.0f};
            verts.push_back(v);
        }
    }

    // Quads: each is split into two CCW triangles viewed from outside.
    // Skip the degenerate cap triangles at the north pole (i==0, first triangle)
    // and south pole (i==stacks-1, second triangle).
    for (uint32_t i = 0; i < stacks; ++i)
    {
        for (uint32_t j = 0; j < sectors; ++j)
        {
            uint32_t v00 = i * (sectors + 1) + j;
            uint32_t v01 = v00 + 1;
            uint32_t v10 = (i + 1) * (sectors + 1) + j;
            uint32_t v11 = v10 + 1;

            if (i != 0) // top triangle (skip north-pole cap)
                idxs.insert(idxs.end(), {v00, v10, v01});
            if (i != stacks - 1) // bottom triangle (skip south-pole cap)
                idxs.insert(idxs.end(), {v01, v10, v11});
        }
    }

    return {verts, idxs};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makePlane(glm::vec3 color, float size)
{
    float h = size * 0.5f;
    // Plane in XY, normal = +Z, U goes along +X → tangent = (1,0,0,1)
    std::vector<Vertex> verts = {
        {{-h, -h, 0.0f}, color, {0, 0}, {0, 0, 1}, {1, 0, 0, 1}},
        {{h, -h, 0.0f}, color, {1, 0}, {0, 0, 1}, {1, 0, 0, 1}},
        {{h, h, 0.0f}, color, {1, 1}, {0, 0, 1}, {1, 0, 0, 1}},
        {{-h, h, 0.0f}, color, {0, 1}, {0, 0, 1}, {1, 0, 0, 1}},
    };
    std::vector<uint32_t> idxs = {0, 1, 2, 0, 2, 3};
    return {verts, idxs};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> loadOBJ(const std::string &path, bool yUpToZUp)
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

            if (yUpToZUp)
            {
                v.pos = {v.pos.x, -v.pos.z, v.pos.y};
                v.normal = {v.normal.x, -v.normal.z, v.normal.y};
                v.tangent = {v.tangent.x, -v.tangent.z, v.tangent.y, v.tangent.w};
            }

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
        glm::vec2 dUV1 = v1.texCoord - v0.texCoord;
        glm::vec2 dUV2 = v2.texCoord - v0.texCoord;

        float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
        if (std::abs(det) < 1e-6f)
            continue; // degenerate UV triangle, skip
        float inv = 1.0f / det;

        glm::vec3 tangent = inv * (dUV2.y * edge1 - dUV1.y * edge2);
        tangentAccum[indices[i]] += tangent;
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

// ── glTF loader ──────────────────────────────────────────────────────────────
//
// Returns one GltfPrimitive per glTF primitive (= one draw call).
// Geometry is in glTF space (Y-up, right-handed). The caller applies
// any coordinate-system transform via the scene.json "yUpToZUp" flag.

static glm::mat4 nodeTransform(const fastgltf::Node &node)
{
    // A node stores its local transform as either a pre-baked matrix or
    // separate TRS (Translation, Rotation as quaternion, Scale) components.
    return std::visit(fastgltf::visitor{[](const fastgltf::math::fmat4x4 &m)
                                        {
                                            // fastgltf stores matrices column-major, same as GLM.
                                            return glm::make_mat4(m.data());
                                        },
                                        [](const fastgltf::TRS &trs)
                                        {
                                            glm::vec3 t = glm::make_vec3(trs.translation.data());
                                            glm::quat r = glm::make_quat(trs.rotation.data());
                                            glm::vec3 s = glm::make_vec3(trs.scale.data());
                                            return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.0f), s);
                                        }},
                      node.transform);
}

static void visitNode(const fastgltf::Asset &asset,
                      size_t nodeIndex,
                      const glm::mat4 &parentTransform,
                      const std::string &baseDir,
                      std::vector<GltfPrimitive> &out, bool yUpToZUp)
{
    const fastgltf::Node &node = asset.nodes[nodeIndex];
    glm::mat4 worldTransform = parentTransform * nodeTransform(node);

    if (node.meshIndex.has_value())
    {
        const fastgltf::Mesh &mesh = asset.meshes[node.meshIndex.value()];

        for (const fastgltf::Primitive &prim : mesh.primitives)
        {
            GltfPrimitive result;
            result.transform = worldTransform;

            // ── Positions ────────────────────────────────────────────────
            // Every primitive must have POSITION. Fail hard if missing.
            auto posIt = prim.findAttribute("POSITION");
            if (posIt == prim.attributes.end())
                throw std::runtime_error("glTF primitive missing POSITION");

            const fastgltf::Accessor &posAcc = asset.accessors[posIt->accessorIndex];
            result.vertices.resize(posAcc.count);

            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                asset, posAcc,
                [&](fastgltf::math::fvec3 p, size_t i)
                {
                    result.vertices[i].pos = {p.x(), p.y(), p.z()};
                    result.vertices[i].color = {1.0f, 1.0f, 1.0f};
                });

            // ── Normals ──────────────────────────────────────────────────
            auto normIt = prim.findAttribute("NORMAL");
            if (normIt != prim.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, asset.accessors[normIt->accessorIndex],
                    [&](fastgltf::math::fvec3 n, size_t i)
                    {
                        result.vertices[i].normal = {n.x(), n.y(), n.z()};
                    });
            }

            // ── Texture coordinates ───────────────────────────────────────
            auto uvIt = prim.findAttribute("TEXCOORD_0");
            if (uvIt != prim.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
                    asset, asset.accessors[uvIt->accessorIndex],
                    [&](fastgltf::math::fvec2 uv, size_t i)
                    {
                        result.vertices[i].texCoord = {uv.x(), uv.y()};
                    });
            }

            // ── Tangents ─────────────────────────────────────────────────
            auto tanIt = prim.findAttribute("TANGENT");
            if (tanIt != prim.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
                    asset, asset.accessors[tanIt->accessorIndex],
                    [&](fastgltf::math::fvec4 t, size_t i)
                    {
                        // glTF tangent: xyz = direction, w = bitangent sign
                        result.vertices[i].tangent = {t.x(), t.y(), t.z(), t.w()};
                    });
            }

            // ── Indices ──────────────────────────────────────────────────
            if (prim.indicesAccessor.has_value())
            {
                const fastgltf::Accessor &idxAcc = asset.accessors[prim.indicesAccessor.value()];
                result.indices.resize(idxAcc.count);
                fastgltf::copyFromAccessor<uint32_t>(asset, idxAcc, result.indices.data());
            }

            // ── Material / texture paths ──────────────────────────────────
            if (prim.materialIndex.has_value())
            {
                const fastgltf::Material &mat = asset.materials[prim.materialIndex.value()];

                auto resolveImage = [&](size_t texIndex) -> GltfImage
                {
                    const fastgltf::Texture &tex = asset.textures[texIndex];
                    if (!tex.imageIndex.has_value())
                        return {};
                    const fastgltf::Image &img = asset.images[tex.imageIndex.value()];

                    if (const auto *uri = std::get_if<fastgltf::sources::URI>(&img.data))
                        return {baseDir + "/" + std::string(uri->uri.path()), {}};

                    if (const auto *arr = std::get_if<fastgltf::sources::Array>(&img.data))
                        return {{}, std::vector<uint8_t>(
                            reinterpret_cast<const uint8_t*>(arr->bytes.data()),
                            reinterpret_cast<const uint8_t*>(arr->bytes.data()) + arr->bytes.size())};

                    // GLB embeds images in the binary buffer chunk, referenced by a BufferView.
                    if (const auto *bv = std::get_if<fastgltf::sources::BufferView>(&img.data))
                    {
                        const fastgltf::BufferView &bufView = asset.bufferViews[bv->bufferViewIndex];
                        const fastgltf::Buffer     &buf     = asset.buffers[bufView.bufferIndex];
                        if (const auto *data = std::get_if<fastgltf::sources::Array>(&buf.data))
                        {
                            const uint8_t *start = reinterpret_cast<const uint8_t*>(data->bytes.data()) + bufView.byteOffset;
                            return {{}, std::vector<uint8_t>(start, start + bufView.byteLength)};
                        }
                    }

                    return {};
                };

                if (mat.pbrData.baseColorTexture.has_value())
                    result.baseColor = resolveImage(mat.pbrData.baseColorTexture->textureIndex);

                if (mat.normalTexture.has_value())
                    result.normalMap = resolveImage(mat.normalTexture->textureIndex);

                if (mat.pbrData.metallicRoughnessTexture.has_value())
                    result.metallicRoughness = resolveImage(mat.pbrData.metallicRoughnessTexture->textureIndex);
            }

            out.push_back(std::move(result));
        }
    }

    // Recurse into children, passing the accumulated world transform.
    for (size_t childIndex : node.children)
        visitNode(asset, childIndex, worldTransform, baseDir, out, yUpToZUp);
}

std::vector<GltfPrimitive> loadGLTF(const std::string &path, bool yUpToZUp)
{
    // The base directory is used to resolve relative texture URIs.
    std::string baseDir = std::filesystem::path(path).parent_path().string();

    auto dataResult = fastgltf::GltfDataBuffer::FromPath(path);
    if (dataResult.error() != fastgltf::Error::None)
        throw std::runtime_error("loadGLTF: could not read file '" + path + "'");

    fastgltf::Parser parser;
    auto assetResult = parser.loadGltf(dataResult.get(),
                                       std::filesystem::path(path).parent_path(),
                                       fastgltf::Options::LoadExternalImages);
    if (assetResult.error() != fastgltf::Error::None)
        throw std::runtime_error("loadGLTF: parse failed for '" + path + "': " + std::string(fastgltf::getErrorMessage(assetResult.error())));

    const fastgltf::Asset &asset = assetResult.get();

    std::vector<GltfPrimitive> primitives;

    // A glTF file can have multiple scenes; use the default one (or scene 0).
    size_t sceneIndex = asset.defaultScene.value_or(0);
    const fastgltf::Scene &scene = asset.scenes[sceneIndex];

    glm::mat4 rootTransform = glm::mat4(1.0f);
    if (yUpToZUp)
    {
        // Columns encode: new_x=old_x, new_y=-old_z, new_z=old_y
        rootTransform = glm::mat4(
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, -1, 0, 0,
            0, 0, 0, 1);
    }

    // Walk every root node; visitNode recurses into children.
    for (size_t rootNode : scene.nodeIndices)
        visitNode(asset, rootNode, rootTransform, baseDir, primitives, yUpToZUp);

    return primitives;
}
