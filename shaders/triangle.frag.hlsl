struct FSOutput
{
	float4 color : SV_TARGET;
};

FSOutput main()
{
	FSOutput output = (FSOutput)0;
	output.color = float4(0.7, 0, 0, 1.0);
	return output;
}