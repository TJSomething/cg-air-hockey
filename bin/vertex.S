attribute vec3 v_position;
attribute vec3 v_color;
attribute vec2 v_tex_coord;
attribute float v_tex_opacity;
attribute vec3 v_normal;
attribute vec3 v_ambient;
attribute vec3 v_diffuse;
attribute vec3 v_specular;
attribute float v_shininess; 

varying vec3 color;
varying vec2 texCoords;
varying float texOpacity;
varying vec3 normal;
varying vec3 ambient;
varying vec3 diffuse;
varying vec3 specular;
varying float shininess;
varying vec4 eyeCoords;

uniform mat4 mvpMatrix;
uniform mat4 toWorldMatrix;

void main(void){
   gl_Position = mvpMatrix * vec4(v_position, 1.0);
   color = v_color;
   texCoords = v_tex_coord;
   texOpacity = v_tex_opacity;
   normal = (toWorldMatrix*mvpMatrix * vec4(v_normal, 0.0)).xyz;
   ambient = v_ambient;
   diffuse = v_diffuse;
   specular = v_specular;
   shininess = v_shininess;
   eyeCoords = toWorldMatrix * gl_Position;
}
