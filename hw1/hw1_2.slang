import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here
    for (int i = 0; i < num_polyline; i++)
    {
        int t;
        int s;
        int qx = tid.x;
        int qy = tid.y;
        int pZeroX = polyline[i][0];
        int pZeroY = polyline[i][1];
        int pOneX = polyline[i+1][0];
        int pOneY = polyline[i+1][1];
        int count = 0;

        t = ((qy - pZeroY) / (pOneY - pZeroY));
        s = (t * (pOneY - pZeroY) + pZeroY - qy);

        if (s >= 0 && t >= 0 && t <= 1) 
        {
            if (pZeroY > pOneY)
            {
                count += 1;
            }
            else
            {
                count -= 1;
            }

            if (count != 0)
            {
                image[tid] = float4(fill_color, 1);
            }
        }
    }
    
    /* pseudo:
    Fill: 
    determine if ray to right is crossing line:
    for each line,
        check y-value on line that is the same as the ray is touching the ray
        if ray's y value is touching line to right:
            if point a's y is greater than point b's y:
                count += 1
            else 
                count -= 1
            if count > 0:
                fill
            else
                don't fill

    Stroke color:
    */
}
