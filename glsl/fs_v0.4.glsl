#version 330 core

// Input: 从顶点着色器插值而来的数据
in vec3 f_color;  // v_color

// Output: 该片元的最终颜色
out vec4 fragColor;
uniform float u_time;  // seed

void main() {
    // 核心：设置该像素的颜色
    fragColor = vec4(f_color, 1.0);
}