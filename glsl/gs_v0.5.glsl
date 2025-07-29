#version 430 core

// 1. 定义输入和输出类型
layout (points) in; // 输入是一个点
layout (points, max_vertices = 3) out; // 输出是一个点

// 接收来自顶点着色器的数据
in VS_OUT {
    vec3 color;
    vec3 scale;
    vec4 rotate;
} gs_in[]; // 注意是数组，因为一个图元可能有多个顶点

// 输出到片元着色器
out vec3 f_color;

// uniform变量，用于从Python控制方形的大小
// uniform float elem_size; // 这里的size是在裁剪空间中的大小



void main() {
    // --- 1. 获取中心点和所有属性 ---
    gl_Position = gl_in[0].gl_Position;
    f_color = gs_in[0].color;

    EmitVertex(); // 发射第一个顶点


    // 结束当前图元的生成
    EndPrimitive();
}