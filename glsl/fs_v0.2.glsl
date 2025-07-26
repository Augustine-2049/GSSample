#version 330 core

// Input: 从顶点着色器插值而来的数据
in vec3 f_color;  // v_color
in vec3 f_scale;
in vec4 f_rotate;

// Output: 该片元的最终颜色
out vec4 fragColor;
uniform float u_time;  // seed

void main() {
    // 核心：设置该像素的颜色
    fragColor = vec4(f_color, 1.0) + vec4(f_scale, u_time) * 0.00000000001f + f_rotate * 0.0000000000001f ;
}