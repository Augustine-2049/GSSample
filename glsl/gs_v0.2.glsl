#version 330 core

// 1. 定义输入和输出类型
layout (points) in; // 输入是一个点
layout (triangle_strip, max_vertices = 4) out; // 输出是一个最多4个顶点的三角带

// 接收来自顶点着色器的数据
in VS_OUT {
    vec3 color;
    vec3 scale;
    vec4 rotate;
} gs_in[]; // 注意是数组，因为一个图元可能有多个顶点

// 输出到片元着色器
out vec3 f_color;
out vec3 f_scale;
out vec4 f_rotate;

// uniform变量，用于从Python控制方形的大小
uniform float point_size; // 这里的size是在裁剪空间中的大小



void main() {
    // --- 1. 获取中心点和所有属性 ---
    vec4 center_pos = gl_in[0].gl_Position;
    vec3 color_val = gs_in[0].color;
    vec3 u_scale = gs_in[0].scale;
    vec4 u_rotation = gs_in[0].rotate;

    // 在裁剪空间中计算四边形的四个角点
    // 注意：gl_Position的w分量用于透视除法，这里需要保持一致
    // 我们在xy平面上扩展，大小为 point_size
    // 左下
    f_scale = u_scale;
    f_rotate = u_rotation;

    f_color = color_val;
    gl_Position = center_pos + vec4(-point_size, -point_size, 0.0, 0.0);
    EmitVertex(); // 发射第一个顶点

    // 左上
    gl_Position = center_pos + vec4(-point_size, point_size, 0.0, 0.0);
    EmitVertex(); // 发射第二个顶点

    // 右下
    gl_Position = center_pos + vec4(point_size, -point_size, 0.0, 0.0);
    EmitVertex(); // 发射第三个顶点

    // 右上
    gl_Position = center_pos + vec4(point_size, point_size, 0.0, 0.0);
    EmitVertex(); // 发射第四个顶点


    // 结束当前图元的生成
    EndPrimitive();
}