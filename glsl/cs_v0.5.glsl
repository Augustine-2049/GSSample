#version 430 core

// /////////// 尝试裁剪，还是非常慢40s/it //////////////


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

// --- 缓冲区绑定 ---
layout(std430, binding = 0) readonly buffer AllPositionsBuffer { vec3 positions[]; };
layout(std430, binding = 1) readonly buffer CntBuffer { int cnts[]; };

struct DrawArraysIndirectCommand {
    uint count;         // 总是 1
    uint instanceCount; // 从 cnts[] 读取
    uint first;         // 顶点的原始索引
    uint baseInstance;  // 总是 0
};

layout(std430, binding = 2) writeonly buffer CommandBuffer {
    DrawArraysIndirectCommand commands[];
};

layout(std430, binding = 3) buffer AtomicCounterBuffer {
    uint visible_command_count;
};

// --- Uniforms ---
uniform mat4 u_proj_view;
uniform uint u_total_points;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= u_total_points) return;

    vec4 clip_pos = u_proj_view * vec4(positions[index], 1.0);

    // 视锥体裁剪测试
    if (clip_pos.x >= -clip_pos.w && clip_pos.x <= clip_pos.w &&
        clip_pos.y >= -clip_pos.w && clip_pos.y <= clip_pos.w &&
        clip_pos.z >= 0.0 && clip_pos.z <= clip_pos.w)
    {
        // 如果可见...
        // 1. 获取要写入的命令槽位
        uint store_index = atomicAdd(visible_command_count, 1);
        
        // 2. 填充绘制命令
        commands[store_index].count = 1;
        commands[store_index].instanceCount = uint(cnts[index]);
        commands[store_index].first = index; // ** 这里的 first 就是原始顶点索引 **
        commands[store_index].baseInstance = 0;
    }
}