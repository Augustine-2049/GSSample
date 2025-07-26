#version 330

// Input: 从顶点着色器插值而来的数据
in vec3 v_color;

// Output: 该片元的最终颜色
out vec4 fragColor;

void main() {
    // 核心：设置该像素的颜色
    fragColor = vec4(v_color, 1.0);
}