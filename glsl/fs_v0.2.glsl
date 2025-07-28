#version 330 core

// Input: 从顶点着色器插值而来的数据
in vec3 f_color;  // v_color
in vec3 f_scale;
in vec4 f_rotate;

// Output: 该片元的最终颜色
out vec4 fragColor;

// 接收来自顶点着色器的数据
//in VS_OUT {
//    vec3 color;
//    vec3 scale;
//    vec4 rotate;
//} gs_in[]; // 注意是数组，因为一个图元可能有多个顶点


// uniform float u_time;  // seed




void main() {
    // 核心：设置该像素的颜色
    fragColor = vec4(f_color, 1.0) + vec4(f_scale, 1.0f) * 0.00000000001f + f_rotate * 0.0000000000001f ;
}