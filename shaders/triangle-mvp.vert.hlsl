struct VSInput
{
	[[vk::location(0)]] float3 Position : POSITION0;
};

struct UBO
{
	float4x4 model;
	float4x4 view;
	float4x4 proj;
};

cbuffer ubo : register(b0, space0) { UBO ubo; }

struct VSOutput
{
	float4 Pos : SV_POSITION;
};

VSOutput main(VSInput input)
{
	VSOutput output = (VSOutput)0;
	output.Pos = mul(ubo.proj,mul(ubo.view, mul(ubo.model,float4(input.Position.xyz,1.0))));
	return output;
}