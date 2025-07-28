#version 330 core

// Uniforms: 全局变量，由Python设置
uniform mat4 projection;
uniform mat4 view;

// Input: 从VBO读取的单个顶点属性
in vec3 in_vert;
in vec3 in_color;
in vec3 in_scale;
in vec4 in_rotate;
// in vec2 in_rand_seed;
// 假设你的点云总数小于某个很大的值，比如 10,000,000
const int MAX_INSTANCES = 3; // 假设你每点最多实例化1024次


// Output: 传递给几何着色器
out VS_OUT{
    vec3 color;
    vec3 scale;
    vec4 rotate;
} vs_out;

// 1. 简单的伪随机数生成器 (0 to 1)
// 使用 gl_PrimitiveIDIn 作为种子，确保每个点都有不同的随机序列
float rand(vec2 seed){
    return fract(sin(dot(seed.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// 2. Box-Muller 变换: 用两个均匀随机数生成两个高斯(正态分布)随机数
vec2 boxMuller(vec2 uniform_rand) {
    float r = sqrt(-2.0 * log(uniform_rand.x));
    float angle = 2.0 * 3.1415926535 * uniform_rand.y;
    return r * vec2(cos(angle), sin(angle));
}

// 3. 四元数旋转向量的函数
vec3 quat_rotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    // 核心计算：将顶点从模型空间变换到裁剪空间
    gl_Position = projection * view * vec4(in_vert, 1.0);


    // 将颜色直接传递下去
    vs_out.color = in_color;
    vs_out.scale = in_scale;
    vs_out.rotate = in_rotate;
    // f_color = in_color;
    // f_scale = in_scale;
    // f_rotate = in_rotate;
}