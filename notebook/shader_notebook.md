# version abstract
- v0.1 : 
    - 点基渲染，ctx.point_size单纯点放大成正方形3 * 3
- v0.2 : 
    - VS->GS->FS
    - VS在椭球上随机新点的位置，实例复用的方式传给GS
    - GS面渲染，layout (triangle_strip, max_vertices = 4) out;
    - vs_test : 使用变换反馈(Transform Feedback)
- v0.3 :
    - GS点渲染




## v0.2
- 注意下面点转面的方法，由于顶点在gl_Position在[-1, 1]，所以下面方法得到的实际是长方形
- v0.2test可以留下来专门检查随机椭球采样对不对
```glsl

layout (points) in; // 输入是一个点
layout (triangle_strip, max_vertices = 4) out; // 输出是一个最多4个顶点的三角带
void main(){
    f_color = color_val;
    gl_Position = center_pos + vec4(-point_size, -point_size, 0.0, 0.0);
    EmitVertex(); // 发射第一个顶点

    // 左上
    gl_Position = center_pos + vec4(-point_size, point_size, 0.0, 0.0);
    EmitVertex(); // 发射第二个顶点

    // 右下
    gl_Position = center_pos + vec4(point_size, -point_size, 0.0, 0.0);
    EmitVertex(); // 发射第三个顶点

    // 右上
    gl_Position = center_pos + vec4(point_size, point_size, 0.0, 0.0);
    EmitVertex(); // 发射第四个顶点


    // 结束当前图元的生成
    EndPrimitive();
}
```
