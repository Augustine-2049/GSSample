#version 330 core

// Uniforms: 全局变量，由Python设置
uniform mat4 projection;
uniform mat4 view;
// const int MAX_INSTANCES = 8192; // 假设你每点最多实例化1024次

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
uniform int MAX_INSTANCES;
uniform float scale_max;  // sample


// 1. 简单的伪随机数生成器 (0 to 1)
// 新的、高质量的伪随机数生成器 (PCG Hash)
// 输入一个种子，返回一个在 [0, 1) 范围内的浮点数
float rand(inout uint seed) {
    seed = seed * 747796405u + 2891336453u;
    uint result = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    result = (result >> 22u) ^ result;
    return float(result) / 4294967295.0; // 2^32 - 1
}
// 新的、更鲁棒的球内均匀采样函数，均匀球内采样
// 输入一个会变化的种子，返回一个在单位实心球体内的随机点
vec3 sample_point_in_unit_sphere(inout uint seed) {
    vec3 point;
    // GLSL虽然可以使用循环，但rand容易导致盲等，这里E=1.9次，不若展开：
    // 这是一个常见的着色器技巧
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;
    point = vec3(rand(seed)*2-1, rand(seed)*2-1, rand(seed)*2-1);
    if (dot(point, point) <= 1.0) return point;    
    // 几次采样无果后，返回球心：
    // 4次-0.075的在球心
    // 8次-0.005的在球心
    // 优化方向：外部随机三个值，
    return vec3(0.0);
}


// 3. 四元数旋转向量的函数
vec3 quat_rotate(vec4 q, vec3 v) {

    return v + 2.0 * cross(q.yzw, cross(q.yzw, v) + q.x * v);
}

void main() {
    // 核心计算：将顶点从模型空间变换到裁剪空间
    // 1. 将 gl_VertexID 和 gl_InstanceID 组合成一个全局唯一的 ID
    float unique_id = float(gl_VertexID * MAX_INSTANCES + gl_InstanceID);
    vec3 u_scale = in_scale * 3; //  * 100;
    float scale = u_scale.x *  u_scale.y * u_scale.z;
    float ratio = scale / scale_max;
    float scaled_instances = ratio * float(MAX_INSTANCES);
    int point_num = int(max(scaled_instances, 1.0));
    if(gl_InstanceID - 10 > point_num){
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec4 u_rotation = in_rotate;
    // --- 2. 在椭球内生成随机偏移向量 ---
    // a. 创建一个独一无二的随机种子
    uint seed = uint(gl_VertexID * MAX_INSTANCES + gl_InstanceID) + uint(in_rand_seed.x * 1000.0);
    // b. 在单位实心球体内生成一个均匀随机点
    vec3 random_point_in_sphere = sample_point_in_unit_sphere(seed);


    // c. 生成一个 [0, 1] 范围内的随机半径，并应用立方根使其在球体内部分布均匀
    //float u_radius = rand(seed * 2.0);
    //float radius = pow(u_radius, 1.0/3.0);

    // d. 得到单位实心球体内的随机点
    //vec3 random_point_in_sphere = radius * point_on_sphere_surface;

    // e. 将该点变换到目标椭球空间，得到最终的偏移向量
    vec3 scaled_offset = random_point_in_sphere * u_scale;
    vec3 rotated_vector = quat_rotate(u_rotation, scaled_offset);


    // float q_w = u_rotation.x;
    // vec3 q_vec = u_rotation.yzw;
    // vec3 cross1 = cross(q_vec, scaled_offset);
    // vec3 term_inside_cross2 = cross1 + q_w * scaled_offset;
    // vec3 cross2 = cross(q_vec, term_inside_cross2);
    // vec3 rotated_vector = scaled_offset + 2 * cross2;


    // --- 3. 计算出最终被偏移的中心点 ---
    // 我们在视图空间(View Space)或裁剪空间(Clip Space)进行偏移
    // 注意：center_pos是齐次坐标(x,y,z,w)，偏移向量是三维的
    if (isnan(rotated_vector.x) || isnan(rotated_vector.y) || isnan(rotated_vector.z)) {
        rotated_vector = vec3(0.0);
    }
    vec3 final_pos = in_vert + rotated_vector;
    gl_Position = projection * view * vec4(final_pos, 1.0);

    // 将颜色直接传递下去
    vs_out.color = in_color;
    vs_out.scale = in_scale;
    vs_out.rotate = in_rotate;
}