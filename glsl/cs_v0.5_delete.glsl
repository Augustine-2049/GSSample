#version 430 core


// /////////// 不成功的尝试，预计算SPP，但更慢（14s/it） ///////////


// 定义工作组大小
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

// 输入：每个点的实例数量
layout(std430, binding = 0) readonly buffer CntBuffer {
    int cnts[];
};

// 输出：绘制命令
// GL_DrawArraysIndirectCommand in C++
struct DrawArraysIndirectCommand {
    uint count;           // 总是 1，因为我们画的是点
    uint instanceCount;   // 从 cnts[] 读取
    uint first;           // 顶点的索引
    uint baseInstance;    // 实例的起始偏移，这里是 0
};

layout(std430, binding = 1) writeonly buffer CommandBuffer {
    DrawArraysIndirectCommand commands[];
};

uniform int u_point_count;

void main() {
    // 获取当前线程处理的顶点索引
    uint index = gl_GlobalInvocationID.x;

    if (index >= u_point_count) {
        return;
    }

    // 从 SSBO 读取这个顶点需要的实例数
    uint instance_count = uint(cnts[index]);

    // 填充绘制命令
    commands[index].count = 1;
    commands[index].instanceCount = instance_count;
    commands[index].first = index;
    commands[index].baseInstance = 0;
}