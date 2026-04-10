struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
    float3 inNormal;
    float4 inTangent; // xyz = tangent direction, w = bitangent sign
};

struct UniformBuffer {
    float4x4 view;
    float4x4 proj;
    float4x4 lightSpaceMatrix;
    float4 lightDir;
    float4 cameraPos;
    float4 materialParams;      // x=ambient, y=specStrength, z=shininess, w=exposure
    float4 pointLightPos[4];    // xyz=world position, w=intensity
    float4 pointLightColor[4];  // xyz=color, w=radius
    float4 lightCounts;         // x=number of active point lights
    float4 shadowParams;        // x=biasMin, y=biasMax
    float4 pomParams;           // x=depthScale, y=minSteps, z=maxSteps
    float4 fogParams;           // x=density, y=heightFalloff, z=maxOpacity  (density=0 → no fog)
    float4 fogColor;            // xyz=linear RGB fog color
};

// ACES filmic tonemapping approximation (Krzysztof Narkowicz).
// Maps HDR linear values to [0,1] with a film-like shoulder and toe.
// Input/output are both in linear light — the sRGB gamma curve is applied
// automatically by the hardware when writing to the sRGB render target.
float3 ACESFilmic(float3 x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants {
    float4x4 model;
    uint32_t textureIndex;
    uint32_t specularMapIndex; // 0xFFFF = no specular map
    uint32_t normalMapIndex;   // 0xFFFF = use geometric normal
    uint32_t heightMapIndex;   // 0xFFFF = no POM
};
[[vk::push_constant]] PushConstants push;

// Parallax Occlusion Mapping.
// Raymarches through the height field (height map: white=raised, black=recessed)
// in tangent space to find the UV where the view ray actually hits the surface.
// Returns the displaced UV to use for all subsequent texture samples.
//
// viewTS: view direction (surface→camera) in tangent space — z component must be > 0.
// numSteps is adaptive: more steps at grazing angles to avoid aliasing.
float2 pomDisplacedUV(float2 uv, float3 viewTS, uint heightIdx, float depthScale, float minSteps, float maxSteps)
{
    // Adaptive step count: cos(angle) ≈ viewTS.z — grazing angles need more steps.
    float numSteps = lerp(maxSteps, minSteps, saturate(viewTS.z));
    float stepSize = 1.0 / numSteps;

    // UV shift per step: project the view direction into the UV plane.
    // Negated because a depressed point appears shifted opposite to the view direction.
    float2 uvStep = -(viewTS.xy / viewTS.z) * depthScale * stepSize;

    float2 currentUV    = uv;
    float  currentDepth = 0.0;  // how far into the surface we've marched (0..1)
    float  h            = textures[NonUniformResourceIndex(heightIdx)].Sample(currentUV).r;

    // Linear search: advance until the ray is below the height field.
    // Height map: white=1 = surface is at its highest (depth needed = 0).
    //             black=0 = surface maximally recessed (depth needed = 1).
    // Condition: continue while currentDepth < (1 - h), i.e., ray is still above surface.
    [loop]
    for (int i = 0; i < int(numSteps) && currentDepth < 1.0 - h; ++i)
    {
        currentUV    += uvStep;
        currentDepth += stepSize;
        h             = textures[NonUniformResourceIndex(heightIdx)].Sample(currentUV).r;
    }

    // Binary search refinement between the last two steps to sharpen the intersection.
    float2 prevUV    = currentUV - uvStep;
    float  prevDepth = currentDepth - stepSize;

    [unroll]
    for (int j = 0; j < 5; ++j)
    {
        float2 midUV    = (currentUV + prevUV) * 0.5;
        float  midDepth = (currentDepth + prevDepth) * 0.5;
        float  midH     = textures[NonUniformResourceIndex(heightIdx)].Sample(midUV).r;

        if (midDepth < 1.0 - midH)
        {
            prevUV    = midUV;   // still above surface, advance
            prevDepth = midDepth;
        }
        else
        {
            currentUV    = midUV;  // below surface, retreat
            currentDepth = midDepth;
        }
    }

    return currentUV;
}

struct VSOutput
{
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
    float4 fragPosLightSpace;
    float3 fragNormal;
    float3 fragWorldPos;
    float3 fragTangent;   // world-space tangent
    float3 fragBitangent; // world-space bitangent (cross(N,T)*sign)
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(push.model, float4(input.inPosition, 1.0));
    output.pos               = mul(ubo.proj, mul(ubo.view, worldPos));
    output.fragColor         = input.inColor;
    output.fragTexCoord      = input.inTexCoord;
    output.fragPosLightSpace = mul(ubo.lightSpaceMatrix, worldPos);
    output.fragWorldPos      = worldPos.xyz;

    // Transform normal and tangent into world space using the model's 3x3 matrix.
    // We do NOT use the inverse-transpose here because our models only have
    // uniform scale — for non-uniform scale you would need it.
    float3x3 normalMatrix = (float3x3)push.model;
    float3 N = normalize(mul(normalMatrix, input.inNormal));
    float3 T = normalize(mul(normalMatrix, input.inTangent.xyz));
    // Re-orthogonalise: remove any component of T that crept into N's direction.
    T = normalize(T - dot(T, N) * N);
    // Bitangent: perpendicular to both N and T. The sign from w handles mirrored UVs.
    float3 B = cross(N, T) * input.inTangent.w;

    output.fragNormal    = N;
    output.fragTangent   = T;
    output.fragBitangent = B;
    return output;
}

[[vk::binding(1, 0)]] Sampler2D textures[];
[[vk::binding(2, 0)]] Sampler2DShadow shadowMap;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    // --- Parallax Occlusion Mapping ---
    // Displace the UV before any texture sampling so all maps (albedo, normal, specular)
    // show the same displaced surface. Only applied when a height map is bound.
    float2 uv = vertIn.fragTexCoord;
    if (push.heightMapIndex != 0xFFFFu)
    {
        // Build the TBN matrix (rows = T, B, N in world space).
        // Multiplying a world-space vector by this matrix gives tangent-space coordinates.
        float3x3 TBN   = float3x3(normalize(vertIn.fragTangent),
                                   normalize(vertIn.fragBitangent),
                                   normalize(vertIn.fragNormal));
        float3 viewDir = normalize(ubo.cameraPos.xyz - vertIn.fragWorldPos);
        float3 viewTS  = mul(TBN, viewDir);   // view direction in tangent space

        uv = pomDisplacedUV(uv, viewTS, push.heightMapIndex,
                            ubo.pomParams.x,  // depthScale
                            ubo.pomParams.y,  // minSteps
                            ubo.pomParams.z); // maxSteps
    }

    float3 projCoords = vertIn.fragPosLightSpace.xyz / vertIn.fragPosLightSpace.w;
    float2 shadowUV = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    // Fragments outside the light frustum (XY bounds or behind far plane) are fully lit.
    // The sampler uses ClampToBorder + white, so out-of-bounds PCF taps already return
    // 1.0, but the explicit check avoids sampling entirely for clearly out-of-bounds fragments.
    // Slope-scale bias: surfaces nearly parallel to the light (low NdotL) need a
    // larger bias to avoid shadow acne. We use the geometric normal (not the
    // normal-mapped one) because the shadow map was rendered from raw geometry.
    // biasMin/biasMax are tuned at runtime via ImGui (ubo.shadowParams.xy).
    float NdotL_geo  = max(dot(normalize(vertIn.fragNormal), normalize(ubo.lightDir.xyz)), 0.0);
    float shadowBias = lerp(ubo.shadowParams.x, ubo.shadowParams.y, NdotL_geo);
    float biasedDepth = currentDepth - shadowBias;

    float shadow = 1.0;
    bool inFrustum = shadowUV.x >= 0.0 && shadowUV.x <= 1.0 &&
                     shadowUV.y >= 0.0 && shadowUV.y <= 1.0 &&
                     currentDepth >= 0.0 && currentDepth <= 1.0;
    if (inFrustum) {
        uint shadowW, shadowH;
        shadowMap.GetDimensions(shadowW, shadowH);
        float2 texelSize = 1.0 / float2(shadowW, shadowH);
        shadow = 0.0;
        for (int x = -1; x <= 1; ++x)
            for (int y = -1; y <= 1; ++y)
                shadow += shadowMap.SampleCmp(shadowUV + float2(x, y) * texelSize, biasedDepth);
        shadow /= 9.0;
    }

    float ambient         = ubo.materialParams.x;
    float specStrength    = ubo.materialParams.y;
    float shininess       = ubo.materialParams.z;

    // Normal: sample the normal map if present, otherwise use the geometric normal.
    // Normal maps store directions in tangent space encoded as RGB in [0,1].
    // We remap to [-1,1] then rotate into world space using the TBN matrix.
    float3 norm;
    if (push.normalMapIndex == 0xFFFFu)
    {
        norm = normalize(vertIn.fragNormal);
    }
    else
    {
        float3 sampledNormal = textures[NonUniformResourceIndex(push.normalMapIndex)].Sample(uv).rgb;
        sampledNormal = sampledNormal * 2.0 - 1.0; // [0,1] → [-1,1]
        // TBN columns: T, B, N. Multiplying transforms from tangent space to world space.
        float3x3 TBN = float3x3(vertIn.fragTangent, vertIn.fragBitangent, vertIn.fragNormal);
        norm = normalize(mul(transpose(TBN), sampledNormal));
    }

    float3 toLight  = normalize(ubo.lightDir.xyz);
    float3 viewDir  = normalize(ubo.cameraPos.xyz - vertIn.fragWorldPos);
    float3 reflDir  = reflect(-toLight, norm);

    float diffuse = max(dot(norm, toLight), 0.0);

    // Specular map: each texel stores how shiny that point on the surface is.
    // rgb allows tinted specular highlights (e.g. gold reflects warm light).
    // Falls back to the global specStrength slider when no map is assigned.
    float3 specMapColor = (push.specularMapIndex == 0xFFFFu)
        ? float3(specStrength, specStrength, specStrength)
        : textures[NonUniformResourceIndex(push.specularMapIndex)].Sample(uv).rgb;

    float  specFactor = (diffuse > 0.0) ? pow(max(dot(viewDir, reflDir), 0.0), shininess) : 0.0;
    float3 specular   = specFactor * specMapColor;

    float exposure   = ubo.materialParams.w;

    float3 texColor = (push.textureIndex == 0xFFFFu)
                    ? float3(1.0, 1.0, 1.0)
                    : textures[NonUniformResourceIndex(push.textureIndex)].Sample(uv).rgb;
    float3 baseColor = vertIn.fragColor * texColor;
    float3 color = baseColor * (ambient + (1.0 - ambient) * diffuse * shadow)
                 + specular * shadow;

    // Point lights: no shadow casting, quadratic falloff within radius.
    int numPointLights = (int)ubo.lightCounts.x;
    for (int i = 0; i < numPointLights; i++)
    {
        float3 plPos       = ubo.pointLightPos[i].xyz;
        float  plIntensity = ubo.pointLightPos[i].w;
        float3 plColor     = ubo.pointLightColor[i].xyz;
        float  plRadius    = ubo.pointLightColor[i].w;

        float3 toLight3D = plPos - vertIn.fragWorldPos;
        float  dist      = length(toLight3D);
        float3 plDir     = toLight3D / dist;

        // Smooth quadratic falloff: full intensity at dist=0, zero at dist=radius.
        float t   = clamp(dist / plRadius, 0.0, 1.0);
        float att = (1.0 - t * t) * (1.0 - t * t) * plIntensity;

        float plDiffuse  = max(dot(norm, plDir), 0.0);
        float3 plReflDir = reflect(-plDir, norm);
        float  plSpec    = (plDiffuse > 0.0)
                         ? pow(max(dot(viewDir, plReflDir), 0.0), shininess)
                         : 0.0;

        color += att * plColor * (baseColor * plDiffuse + specMapColor * plSpec);
    }

    // --- Fog ---
    // Applied in linear HDR space before tonemapping so the fog color goes through
    // the same ACES curve as the scene — avoids a mismatched gamma/tone response.
    //
    // Exponential height fog: fog is denser near the ground and thins with altitude.
    //   fogFactor = 1 - exp(-density * dist * heightAttenuation)
    //   heightAtt = exp(-heightFalloff * max(worldPos.z, 0))
    //
    // Setting density = 0 (via CPU when fog is disabled) makes fogFactor = 0,
    // which skips the blend with no extra shader branch.
    {
        float fogDensity      = ubo.fogParams.x;
        float fogHeightFalloff= ubo.fogParams.y;
        float fogMaxOpacity   = ubo.fogParams.z;

        float dist      = length(ubo.cameraPos.xyz - vertIn.fragWorldPos);
        float heightAtt = exp(-fogHeightFalloff * max(vertIn.fragWorldPos.z, 0.0));
        float fogFactor = 1.0 - exp(-fogDensity * dist * heightAtt);
              fogFactor = clamp(fogFactor, 0.0, fogMaxOpacity);

        color = lerp(color, ubo.fogColor.xyz, fogFactor);
    }

    // Apply exposure then tonemap into [0,1] linear range.
    // The sRGB render target applies the gamma curve automatically on write,
    // so no manual pow(color, 1/2.2) is needed here.
    // exposure == 0 is the sentinel for "tonemapping disabled" — pass through.
    if (exposure > 0.0f)
        color = ACESFilmic(color * exposure);
    else
        color = clamp(color, 0.0f, 1.0f);

    return float4(color, 1.0);
}
