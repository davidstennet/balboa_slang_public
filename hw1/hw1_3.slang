import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1.0);
    // TODO: your code here
    int num_circles = 0;
    int num_polylines = 0;

    for (int i = 0; i < num_shapes; i++)
    {
        Shape currentShape = shapes[i];

        if (currentShape.type == ShapeType::Circle)
        {
            num_circles += 1;
        }
        else if (currentShape.type == ShapeType::Polyline)
        {
            num_polylines += 1;
        }
    }

    int circle_i = num_circles - 1;
    int polyline_i = num_polylines - 1;
    
    for (int i = num_shapes - 1; i >= 0; i--)
    {
        Shape currentShape = shapes[i];
        
        if (currentShape.type == ShapeType::Circle)
        {
            float4 stroke_rgba = currentShape.stroke_rgba;
            float4 fill_rgba = currentShape.fill_rgba;

            Circle currentCircle = circles[circle_i];

            float distance = sqrt(pow(currentCircle.center.x - tid.x, 2) + pow(currentCircle.center.y - tid.y, 2));

            // Stroke
            if (currentShape.use_stroke)
            {
                if (distance < currentCircle.radius + currentShape.stroke_width && distance > currentCircle.radius - currentShape.stroke_width)
                {
                    image[tid] = stroke_rgba;
                }
            }
            
            // Fill
            else if (currentShape.use_fill)
            {
                
                if (distance < currentCircle.radius)
                {
                    image[tid] = fill_rgba;
                }
            }

            circle_i--;
            
        }
        else if (currentShape.type == ShapeType::Polyline)
        {
            image[tid] = float4(0.5, 0.5, 0.5, 1);
        } 
    }
}
