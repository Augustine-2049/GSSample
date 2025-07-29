// postproc_frag_shader (fs_hole_filling_advanced.glsl)
#version 430 core

// 输入：从第一步渲染得到的纹理
uniform sampler2D u_render_texture;

// 输入：从全屏四边形的顶点着色器传来的纹理坐标
in vec2 v_texCoord;

// 输出：最终的像素颜色
out vec4 fragColor;

// 一个简单的伪随机数生成器
float rand(vec2 seed){
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // 1. 检查当前像素是否有内容
    if (texture(u_render_texture, v_texCoord).a > 0.0) {
        fragColor = texture(u_render_texture, v_texCoord);
        return;
    }

    // --- 如果是空洞，则开始扩大范围搜索 ---

    const int MAX_NEIGHBORS_TO_FIND = 3;
    const int MAX_SEARCH_RADIUS = 50; // 设置一个最大搜索半径，防止性能问题

    vec4 valid_neighbors[MAX_NEIGHBORS_TO_FIND];
    int valid_count = 0;

    vec2 texel_size = 1.0 / vec2(textureSize(u_render_texture, 0));

    // 2. 循环扩大搜索半径
    for (int radius = 1; radius <= MAX_SEARCH_RADIUS; radius++) {
        
        // 遍历当前半径 'radius' 形成的正方形边框
        // 我们需要检查4条边：上、下、左、右
        for (int i = -radius; i <= radius; i++) {
            // 上边和下边
            if (abs(i) == radius) { // 检查水平边
                for (int j = -radius; j <= radius; j++) {
                     vec4 neighbor = texture(u_render_texture, v_texCoord + vec2(j, i) * texel_size);
                     if(neighbor.a > 0.0) {
                         valid_neighbors[valid_count++] = neighbor;
                         if (valid_count >= MAX_NEIGHBORS_TO_FIND) break;
                     }
                }
            } else { // 检查垂直边 (避免重复检查角点)
                vec4 neighbor_left = texture(u_render_texture, v_texCoord + vec2(-radius, i) * texel_size);
                if(neighbor_left.a > 0.0) {
                    valid_neighbors[valid_count++] = neighbor_left;
                    if (valid_count >= MAX_NEIGHBORS_TO_FIND) break;
                }
                vec4 neighbor_right = texture(u_render_texture, v_texCoord + vec2(radius, i) * texel_size);
                 if(neighbor_right.a > 0.0) {
                    valid_neighbors[valid_count++] = neighbor_right;
                    if (valid_count >= MAX_NEIGHBORS_TO_FIND) break;
                }
            }
            if (valid_count >= MAX_NEIGHBORS_TO_FIND) break;
        }

        // 如果已经找到了足够的邻居，就跳出最外层循环
        if (valid_count >= MAX_NEIGHBORS_TO_FIND) {
            break;
        }
    }

    // 3. 根据搜索结果决定最终颜色
    if (valid_count == 0) {
        // 如果在最大半径内一个邻居都没找到，放弃这个像素
        discard;
    } else {
        // 从找到的有效邻居中随机挑选一个
        // 用当前片元的屏幕坐标作为随机种子
        float random_val = rand(gl_FragCoord.xy);
        // random_index 的范围是 [0, valid_count - 1]
        int random_index = min(int(random_val * float(valid_count)), valid_count - 1);

        fragColor = valid_neighbors[random_index];
    }
}
