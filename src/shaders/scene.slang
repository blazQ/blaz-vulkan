struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
    float3 inNormal;
};

struct UniformBuffer {
    float4x4 view;
    float4x4 proj;
    float4x4 lightSpaceMatrix;
    float4 lightDir; 
    float4 cameraPos;
    float4 materialParams;
};
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants {
    float4x4 model;
};
[[vk::push_constant]] PushConstants push;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
    float4 fragPosLightSpace;
    float3 fragNormal;
    float3 fragWorldPos;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(push.model, float4(input.inPosition, 1.0));
    output.pos = mul(ubo.proj, mul(ubo.view, worldPos));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.fragPosLightSpace = mul(ubo.lightSpaceMatrix, worldPos);
    output.fragNormal = normalize(mul((float3x3)push.model, input.inNormal));
    output.fragWorldPos = worldPos.xyz;
    return output;
}

[[vk::binding(1, 0)]] Sampler2D texture;
[[vk::binding(2, 0)]] Sampler2DShadow shadowMap;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float3 projCoords = vertIn.fragPosLightSpace.xyz / vertIn.fragPosLightSpace.w;
    float2 shadowUV = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    // Fragments outside the light frustum are always lit
    float shadow = 1.0;
    if (currentDepth <= 1.0) {
        uint shadowW, shadowH;
        shadowMap.GetDimensions(shadowW, shadowH);
        float2 texelSize = 1.0 / float2(shadowW, shadowH);
        shadow = 0.0;
        for (int x = -1; x <= 1; ++x)
            for (int y = -1; y <= 1; ++y)
                shadow += shadowMap.SampleCmp(shadowUV + float2(x, y) * texelSize, currentDepth);
        shadow /= 9.0;
    }

    float ambient         = ubo.materialParams.x;
    float specStrength    = ubo.materialParams.y;
    float shininess       = ubo.materialParams.z;

    float3 norm     = normalize(vertIn.fragNormal);
    float3 toLight  = normalize(ubo.lightDir.xyz);
    float3 viewDir  = normalize(ubo.cameraPos.xyz - vertIn.fragWorldPos);
    float3 reflDir  = reflect(-toLight, norm);

    float diffuse  = max(dot(norm, toLight), 0.0);
    float specular = (diffuse > 0.0)
                   ? pow(max(dot(viewDir, reflDir), 0.0), shininess) * specStrength
                   : 0.0;

    float3 color = vertIn.fragColor * (ambient + (1.0 - ambient) * diffuse * shadow)
                 + specular * shadow;
    return float4(color, 1.0);
}
