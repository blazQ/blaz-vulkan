// Must match the full PushConstants layout declared in the pipeline layout,
// even though the shadow pass only uses the model matrix.
struct ShadowPush {
    float4x4 model;
    uint32_t textureIndex;
    uint32_t specularMapIndex;
    uint32_t normalMapIndex;
    uint32_t heightMapIndex;   // must match full PushConstants layout; unused in shadow pass
};
[[vk::push_constant]] ShadowPush push;

struct ShadowUBO { float4x4 view; float4x4 proj; float4x4 lightSpaceMatrix; };
ConstantBuffer<ShadowUBO> ubo;

[shader("vertex")]
float4 vertMain(float3 pos : POSITION) : SV_Position {
    return mul(ubo.lightSpaceMatrix, mul(push.model, float4(pos, 1.0)));
}