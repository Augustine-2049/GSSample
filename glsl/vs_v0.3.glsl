#version 330 core

// Uniforms: 全局变量，由Python设置
uniform mat4 projection;
uniform mat4 view;
const int MAX_INSTANCES = 100; // 假设你每点最多实例化1024次

// Input: 从VBO读取的单个顶点属性
in vec3 in_vert;
in vec3 in_color;
in vec3 in_scale;
in vec4 in_rotate;
in vec2 in_rand_seed;


// Output: 传递给几何着色器
out VS_OUT{
    vec3 color;
    vec3 scale;
    vec4 rotate;
} vs_out;
// 关键：声明一个自定义的 out 变量来捕获 gl_Position
// out vec4 captured_gl_position;

// uniform float u_time;  // seed



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
    // 1. 将 gl_VertexID 和 gl_InstanceID 组合成一个全局唯一的 ID
    float unique_id = float(gl_VertexID * MAX_INSTANCES + gl_InstanceID);
    gl_Position = projection * view * vec4(in_vert, 1.0);
    vec3 u_scale = exp(in_scale); //  * 100;
    vec4 u_rotation = normalize(in_rotate);
    // --- 2. 在椭球内生成随机偏移向量 ---
    // a. 创建一个独一无二的随机种子
    vec2 seed = vec2(unique_id, in_rand_seed[0]);

    // b. 在单位球体表面生成一个均匀随机点 (高斯法)
    vec2 u_gauss1 = vec2(rand(seed), rand(seed.yx));
    vec2 u_gauss2 = vec2(rand(seed * 0.5), rand(seed.yx * 0.7));
    vec2 gauss1 = boxMuller(u_gauss1);
    vec2 gauss2 = boxMuller(u_gauss2);
    vec3 point_on_sphere_surface = normalize(vec3(gauss1.x, gauss1.y, gauss2.x));

    // c. 生成一个 [0, 1] 范围内的随机半径，并应用立方根使其在球体内部分布均匀
    float u_radius = rand(seed * 2.0);
    float radius = pow(u_radius, 1.0/3.0);

    // d. 得到单位实心球体内的随机点
    vec3 random_point_in_sphere = radius * point_on_sphere_surface;

    // e. 将该点变换到目标椭球空间，得到最终的偏移向量
    vec3 scaled_offset = random_point_in_sphere * u_scale;
    vec3 final_offset_vector = quat_rotate(u_rotation, scaled_offset);


    float q_w = u_rotation.x;
    vec3 q_vec = u_rotation.yzw;
    vec3 cross1 = cross(q_vec, scaled_offset);
    vec3 term_inside_cross2 = cross1 + q_w * scaled_offset;
    vec3 cross2 = cross(q_vec, term_inside_cross2);
    vec3 rotated_vector = scaled_offset + 2 * cross2;


    // --- 3. 计算出最终被偏移的中心点 ---
    // 我们在视图空间(View Space)或裁剪空间(Clip Space)进行偏移
    // 注意：center_pos是齐次坐标(x,y,z,w)，偏移向量是三维的
    vec3 final_pos = in_vert + rotated_vector;
    gl_Position = projection * view * vec4(final_pos, 1.0);

    // 将颜色直接传递下去
    vs_out.color = in_color;
    vs_out.scale = in_scale;
    vs_out.rotate = in_rotate;
}