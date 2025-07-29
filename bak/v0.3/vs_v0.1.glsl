#version 330

// Uniforms: 全局变量，由Python设置
uniform mat4 projection;
uniform mat4 view;

// Input: 从VBO读取的单个顶点属性
in vec3 in_vert;
in vec3 in_color;

// Output: 传递给片元着色器
out vec3 v_color;

void main() {
    // 核心计算：将顶点从模型空间变换到裁剪空间
    gl_Position = projection * view * vec4(in_vert, 1.0);
    
    // 将颜色直接传递下去
    v_color = in_color;
}