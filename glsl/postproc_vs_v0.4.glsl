#version 330 core
in vec2 in_vert;
out vec2 v_texCoord;
void main() {
    v_texCoord = in_vert * 0.5 + 0.5;  // [0, 1]
    gl_Position = vec4(in_vert, 0.0, 1.0);  // [[0-1], [0-1], 0.0, 1.0]
}