#version 330 core

// Uniforms: 全局变量，由Python设置
uniform mat4 projection;
uniform mat4 view;
// const int MAX_INSTANCES = 16384; // 下面作为uniform传入

// Input: 从VBO读取的单个顶点属性
in vec3 in_vert;
in vec3 in_color;
in vec3 in_scale;
in vec4 in_rotate;
// in float in_opacity;
in int in_cnt;
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
uniform float opacity_max; // sample

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
    float z = rand(seed) * 2.0 - 1.0;
    float a = rand(seed) * 2.0 * 3.14159265;
    float r = sqrt(1.0 - z*z);
    float x = r * cos(a);
    float y = r * sin(a);
    
    // 应用立方根半径
    float radius = pow(rand(seed), 1.0/3.0);
    return vec3(x, y, z) * radius;
}


// 3. 四元数旋转向量的函数
vec3 quat_rotate(vec4 q, vec3 v) {

    return v + 2.0 * cross(q.yzw, cross(q.yzw, v) + q.x * v);
}

void main() {
    // 核心计算：将顶点从模型空间变换到裁剪空间
    // 1. 将 gl_VertexID 和 gl_InstanceID 组合成一个全局唯一的 ID
    float unique_id = float(gl_VertexID * MAX_INSTANCES + gl_InstanceID);
    if(gl_InstanceID + 1 > in_cnt){
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    //float scale = u_scale.x *  u_scale.y * u_scale.z;
    //float ratio = scale > scale_max ? 1 : scale / scale_max;
    //int scaled_instances = int(max(ratio * float(MAX_INSTANCES), 1.0));
    //if(gl_InstanceID + 1 > scaled_instances){
    //    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    //    return;
    //}
    //ratio = in_opacity > opacity_max ? 1 : in_opacity / opacity_max;
    //int opacity_instances = int(max(ratio * float(MAX_INSTANCES), 1.0));
    //if(gl_InstanceID + 1 > opacity_instances){
    //    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    //    return;
    //}

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
    vec3 scaled_offset = random_point_in_sphere * in_scale;
    vec4 u_rotation = in_rotate;
    vec3 rotated_vector = quat_rotate(u_rotation, scaled_offset);



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