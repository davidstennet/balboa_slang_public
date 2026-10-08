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
    
    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    uint x = tid.x;
    uint y = height - tid.y;

    // For loop for Fill Color
    int count = 0;
    for (int i = 0; i < num_polyline; i++)
    {
        float t;
        float s;
        float qx = float(x);
        float qy = float(y);
        float pZeroX = polyline[i].x;
        float pZeroY = polyline[i].y;
        float pOneX;
        float pOneY;
        if (i == num_polyline - 1) {
            pOneX = polyline[0].x;
            pOneY = polyline[0].y;
        }
        else {
            pOneX = polyline[i+1].x;
            pOneY = polyline[i+1].y;
        }
        
        t = ((qy - pZeroY) / (pOneY - pZeroY));
        s = (t * (pOneX - pZeroX) + pZeroX - qx);

        if (s >= 0 && t >= 0 && t <= 1) 
        {
            if (pZeroY > pOneY && pZeroY != y)
            {
                count += 1;
            }
            else if (pZeroY < pOneY && pOneY != y)
            {
                count -= 1;
            }
        }
    }

    if (count != 0 && use_fill_color)
    {
        image[tid] = float4(fill_color, 1);
    }

    // For loop for stroke color
    for (int i = 0; i < num_polyline; i++) 
    {

        // declarations for variables
        float L;
        float qminq2;
        float2 q2;
        float2 pOne;
        if (i == num_polyline - 1)
            pOne = polyline[0];
        else
            pOne = polyline[i+1];

        float2 pZero = polyline[i];
        float p1minp0 = length(pOne - pZero);
        float2 q = float2(tid);
        q.y = height - q.y;
    

        // math for required values
        L = dot(pOne - pZero, q - pZero) / p1minp0;
        q2 = pZero + L * (pOne - pZero) / p1minp0; 
        qminq2 = length(q - q2);

        if (L > 0 && L < p1minp0 && qminq2 < stroke_width/2) {
            image[tid] = float4(stroke_color, 1);
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
