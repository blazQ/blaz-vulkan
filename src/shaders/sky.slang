// Procedural sky shader — renders a gradient sky with a sun disk.
// Drawn as a fullscreen triangle (no vertex buffer) after the scene geometry.
// The depth state is eLessOrEqual / no-write, so sky only fills pixels where
// no geometry was drawn (depth buffer still holds the clear value of 1.0).
//
// View-ray reconstruction:
//   1. Vertex shader emits NDC position at (ndc, z=1, w=1) so the rasterizer
//      places it at the far plane (depth 1.0 after division).
//   2. Fragment shader unprojections the NDC xy using invProj (from the UBO)
//      to get a view-space direction, then rotates it to world space using
//      invViewRot (the rotation-only inverse of the view matrix).

static const float PI = 3.14159265358979323846f;

// Must match UniformBufferObject in VulkanRenderer.hpp exactly.
struct SkyUBO {
    float4x4 view;
    float4x4 proj;
    float4x4 lightSpaceMatrix;
    float4   lightDir;          // xyz = direction toward the sun (world space)
    float4   cameraPos;
    float4   materialParams;
    float4   pointLightPos[4];
    float4   pointLightColor[4];
    float4   lightCounts;
    float4   shadowParams;      // x=biasMin, y=biasMax (unused by sky, but must match layout)
    float4   pomParams;         // x=depthScale, y=minSteps, z=maxSteps (unused by sky)
    float4x4 invProj;           // inverse of the projection matrix
    float4x4 invViewRot;        // inverse of view rotation (translation zeroed)
};
ConstantBuffer<SkyUBO> ubo;

// Per-frame sky appearance, pushed from the CPU each frame.
struct SkyPush {
    float4 horizonColor; // xyz = linear RGB at the horizon
    float4 zenithColor;  // xyz = linear RGB at the zenith
    float4 groundColor;  // xyz = linear RGB below the horizon
    float4 sunParams;    // xyz = sun color (can exceed 1 for HDR), w = cos(half-angle)
};
[[vk::push_constant]] SkyPush push;

struct SkyVSOut {
    float4 pos : SV_Position;
    float2 ndc;              // raw NDC xy, passed to fragment for ray reconstruction
};

[shader("vertex")]
SkyVSOut skyVert(uint vid : SV_VertexID) {
    SkyVSOut o;
    // Three-vertex fullscreen triangle (no index buffer needed).
    // Vertex 0: (-1,-1), 1: (3,-1), 2: (-1,3) — covers the entire NDC square.
    float2 uv  = float2((vid << 1) & 2, vid & 2);
    float2 ndc = uv * 2.0f - 1.0f;
    // z = w = 1 so after perspective divide z/w = 1.0 = far-plane depth.
    o.pos = float4(ndc, 1.0f, 1.0f);
    o.ndc = ndc;
    return o;
}

[shader("fragment")]
float4 skyFrag(SkyVSOut i) : SV_TARGET {
    // --- Reconstruct world-space view direction ---
    // Unproject NDC → view space (w-divide recovers the actual position)
    float4 vsPos = mul(ubo.invProj, float4(i.ndc, 1.0f, 1.0f));
    vsPos /= vsPos.w;
    // Rotate view-space direction to world space (translation doesn't apply to directions)
    float3 dir = normalize(mul(ubo.invViewRot, float4(vsPos.xyz, 0.0f)).xyz);

    // --- Sky gradient (Z-up world) ---
    // dir.z == +1 is straight up, -1 is straight down.
    float elev = dir.z;

    float3 skyColor;
    if (elev < 0.0f) {
        // Below horizon: blend from horizon to ground.
        // Sharper falloff so the ground color doesn't bleed too high.
        float t = clamp(-elev * 4.0f, 0.0f, 1.0f);
        skyColor = lerp(push.horizonColor.xyz, push.groundColor.xyz, t);
    } else {
        // Above horizon: blend from horizon to zenith.
        float t = clamp(elev * 1.5f, 0.0f, 1.0f);
        skyColor = lerp(push.horizonColor.xyz, push.zenithColor.xyz, t);
    }

    // --- Sun disk ---
    // push.sunParams.w is cos(half-angle) of the sun disk.
    // A value of 0.9998 ≈ cos(1.1°), which gives a realistic but visible sun.
    float3 sunDir  = normalize(ubo.lightDir.xyz);
    float  cosAng  = dot(dir, sunDir);
    float  sunEdge = push.sunParams.w;
    if (cosAng > sunEdge) {
        // Smooth blend from sky to sun color inside the disk.
        float blend = clamp((cosAng - sunEdge) / (1.0f - sunEdge + 1e-6f), 0.0f, 1.0f);
        // Soft halo: square the blend for a sharper core.
        blend = blend * blend;
        skyColor = lerp(skyColor, push.sunParams.xyz, blend);
    }

    return float4(skyColor, 1.0f);
}
