#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <memory>

#include "KDTree.hpp" // Implementación de KDTree descargada de https://github.com/crvs/KDTree

// -------------- VARIABLES GLOBALES ------------------

// TESELACIÓN
GLuint maxDetalleExterno = 14;
GLuint minDetalleExterno = 2;
GLuint maxDetalleInterno = 14;
GLuint minDetalleInterno = 2;
GLfloat distanciaDetalleCerca = 8.0f;
GLfloat distanciaDetalleLejos = 90.0f;
GLfloat intensidadDesplazamiento = 0.6f;

// CONSTANTES ASTEROIDES
const int NUM_ASTEROIDES = 3000;
const float RADIO_CAMPO_ASTEROIDES = 70.0f;
const float RADIO_INICIAL_ASTERIODE = 0.5f;

// CÁMARA
glm::vec3 pos_obs = glm::vec3(0.0f, 0.0f, RADIO_CAMPO_ASTEROIDES * 1.2f);
glm::vec3 target = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 obs_up = glm::vec3(0.0f, 1.0f, 0.0f);
float az = -90.0f;
float elev = 0.0f;

const float VELOCIDAD_MOVIMIENTO_CAMARA = 0.3f;
const float VELOCIDAD_ROTACION_CAMARA = 1.0f;


// TIEMPO
double lastFPSTime = 0.0;

// CAMPO DE VISIÓN (sacado de documentación https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling)
struct Plane {
    glm::vec3 normal = glm::vec3(0.0f);
    float distance = 0.0f;

    Plane() = default;
    Plane(const glm::vec3& n, float d) : normal(n), distance(d) {}

    void normalize() {
        float mag = glm::length(normal);
        if (mag > 0.00001f) {
            normal /= mag; distance /= mag;
        }
    }

    float getSignedDistanceToPoint(const glm::vec3& point) const {
        return glm::dot(normal, point) + distance;
    }
};

struct Frustum {
    Plane planes[6];
    void update(const glm::mat4& vpMatrix) {
        const glm::mat4& m = vpMatrix;
        planes[0] = Plane({ m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0] }, m[3][3] + m[3][0]);
        planes[1] = Plane({ m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0] }, m[3][3] - m[3][0]);
        planes[2] = Plane({ m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1] }, m[3][3] + m[3][1]);
        planes[3] = Plane({ m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1] }, m[3][3] - m[3][1]);
        planes[4] = Plane({ m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2] }, m[3][3] + m[3][2]);
        planes[5] = Plane({ m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2] }, m[3][3] - m[3][2]);
        for (int i = 0; i < 6; ++i) planes[i].normalize();
    }

    bool isSphereInside(const glm::vec3& center, float radius) const {
        for (int i = 0; i < 6; ++i) { if (planes[i].getSignedDistanceToPoint(center) < -radius) return false; } return true;
    }
};

// ASTEROIDES
std::unique_ptr<KDTree>     kdtree_asteroide;
std::vector<glm::vec3>      pos_asteroides;
std::vector<glm::mat4>      R;
std::vector<glm::vec3>      velocidad_direccion_asteroides;
std::vector<glm::vec3>      eje_rotacion_asteroides;
std::vector<float>          magnitud_rotacion_asteroides;
std::vector<float>          escalas_asteroides;
std::vector<glm::mat4>      M;
Frustum                     campoVision;

// Declaración CALLBACKS
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

GLuint LoadShaders(const char* vertex_file_path, const char* tess_control_file_path, const char* tess_eval_file_path, const char* fragment_file_path);
void generarVerticesIcosfera(std::vector<glm::vec3>& vertices, float radius);

// ---------------- SHADERS -----------------
const char* vertexShaderSource = R"(
#version 410 core
layout (location = 0) in vec3 aPos_model; // Posicion de cada vertice de la icosfera base
layout (location = 1) in mat4 instanceModelMatrix; // Matriz de transformacion del asteroide

out VS_TCS_INTERFACE { // Lo que va a ir al TCS
    vec3 modelPos_tcs;
    mat4 instanceMatrix_tcs;
} vs_out;

// Lo pasamos al TCS
void main() {
    vs_out.modelPos_tcs = aPos_model;
    vs_out.instanceMatrix_tcs = instanceModelMatrix;
    gl_Position = vec4(aPos_model, 1.0);
}
)";

const char* tessControlShaderSource = R"(
#version 410 core
layout (vertices = 3) out; // Salida del patch de 3 vertices

// Uniforms que recibe
uniform vec3 cameraPosition_world;
uniform int maxOuterLodLevel;
uniform int minOuterLodLevel;
uniform int maxInnerLodLevel;
uniform int minInnerLodLevel;
uniform float lodNearDistance;
uniform float lodFarDistance;

in VS_TCS_INTERFACE { // Tres vertices del triangulo (el patch)
    vec3 modelPos_tcs;
    mat4 instanceMatrix_tcs;
} tcs_in[];

// Salida por patch
patch out mat4 instanceMatrix_tes_patch;
patch out vec3 modelPos_tes_patch[3];
patch out float displacementFactor_tes_patch;
// Tambien salen el gl_TessLevelInner y Outer

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    modelPos_tes_patch[gl_InvocationID] = tcs_in[gl_InvocationID].modelPos_tcs; // Pasamos la posicion de los vertices originales al pathc

    // Calculamos nivel de detalle
    if (gl_InvocationID == 0) {
        vec3 asteroidCenter_world = vec3(tcs_in[0].instanceMatrix_tcs[3]);
        float distToCamera = distance(asteroidCenter_world, cameraPosition_world); // Distancia del centro del asteroide a la camara

        // Calculamos el detalle
        // Con smoothstep si el asteroide está a más de lodFarDsitance, hace que el detalle sea 0
        // Si está más cerca que Near, es 1
        // Entre medias, interpola
        float lodFactor = smoothstep(lodFarDistance, lodNearDistance, distToCamera);

        // Interpolamos linealmente entre minimo y maximo detalle para subdivisiones del exterior e interior del triangulo
        int currentOuterLevel = int(mix(float(minOuterLodLevel), float(maxOuterLodLevel), lodFactor));
        int currentInnerLevel = int(mix(float(minInnerLodLevel), float(maxInnerLodLevel), lodFactor));

        // Guardamos el nivel de teselacion calculado
        gl_TessLevelInner[0] = currentInnerLevel;
        gl_TessLevelOuter[0] = currentOuterLevel;
        gl_TessLevelOuter[1] = currentOuterLevel;
        gl_TessLevelOuter[2] = currentOuterLevel;

        // Pasar la matriz de instancia del asteroide
        instanceMatrix_tes_patch = tcs_in[0].instanceMatrix_tcs;

        // Intensidad del desplazamiento
        float farDistanceDisplacementScale = 0.0;
        float nearDistanceDisplacementScale = 1.0;
        displacementFactor_tes_patch = mix(farDistanceDisplacementScale, nearDistanceDisplacementScale, lodFactor);
    }
}
)";

const char* tessEvalShaderSource = R"(
#version 410 core
layout(triangles, fractional_even_spacing, ccw) in;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform float displacementStrength_tes;

// Recibir los datos por parche
patch in mat4 instanceMatrix_tes_patch;  // Matriz de instancia del asteroide
patch in vec3 modelPos_tes_patch[3]; // Posiciones vertices originales del patch
patch in float displacementFactor_tes_patch; // Factor del desplazamiento

// Salida del TCS
out TES_FS_INTERFACE {
    vec3 FragPos_world;
    vec3 Normal_world;
} tes_out;

// Funcion para calcular el desplazamiento
float getDisplacementValue(vec3 coord_for_displacement) {
    float displacement = 0.0;

    // Se va a calcular el desplazamiento como por "ondas"
    float frequency1 = 12.0; // frecuencia de la onda
    float amplitude1 = 0.05; // amplitud de la onda

    float frequency2 = 5.0;
    float amplitude2 = 0.08;

    float frequency3 = 20.0;
    float amplitude3 = 0.03;

    displacement += sin(coord_for_displacement.x * frequency1 + coord_for_displacement.y * frequency1 * 0.5) * amplitude1;
    displacement += cos(coord_for_displacement.y * frequency2 - coord_for_displacement.z * frequency2 * 0.7) * amplitude2;
    displacement += sin(coord_for_displacement.z * frequency3) * cos(coord_for_displacement.x * frequency3 * 0.3) * amplitude3;
    
    return displacement;
}

// Funcion de la luz para caluclar la nueva normal despues de haber movido los vertices de la superficie
// Recibe
// p_model_interpolated: la posición del vértice oriignial (antes del displacement)
// initial_normal_model: la normal originial en ese putno, nos va a servir como un eje
// effective_disp_strength: la fuerza del desplazamiento que hemos aplicado
vec3 calculateDisplacedNormal_model(vec3 p_model_interpolated, vec3 initial_normal_model, float effective_disp_strength) {
    
    // Para fuerza nula, no calculamos
    if (effective_disp_strength < 0.0001) {
        return initial_normal_model;
    }

    float epsilon = 0.01; // La inclinacion teniendo en cuenta lo que le rodea
    vec3 tangentU, tangentV; // Son como los nuevos ejes junto con la normal original

    if (abs(initial_normal_model.y) < 0.95) { // Si la normal no es practicamente vertical
        tangentU = normalize(cross(initial_normal_model, vec3(0.0, 1.0, 0.0))); // Si no lo es, hacemos producto con eje Y
    } else { // Si lo es, hacemos producto con eje X
        tangentU = normalize(cross(initial_normal_model, vec3(1.0, 0.0, 0.0)));
    }
    tangentV = normalize(cross(initial_normal_model, tangentU)); // El ultimo es el normal de ambos

    // Calculamos posiciones cercanas (moviendonos epsilon por los ejes)
    vec3 p_plus_u  = p_model_interpolated + tangentU * epsilon;
    vec3 p_minus_u = p_model_interpolated - tangentU * epsilon;
    vec3 p_plus_v  = p_model_interpolated + tangentV * epsilon;
    vec3 p_minus_v = p_model_interpolated - tangentV * epsilon;

    // Normalizamos, lo que nos da la normal en este punto de la esfera
    vec3 n_p_plus_u  = normalize(p_plus_u);
    vec3 n_p_minus_u = normalize(p_minus_u);
    vec3 n_p_plus_v  = normalize(p_plus_v);
    vec3 n_p_minus_v = normalize(p_minus_v);
    
    // Calculamos el desplazamiento de esos puntos cercanos
    float disp_val_plus_u  = getDisplacementValue(n_p_plus_u);
    float disp_val_minus_u = getDisplacementValue(n_p_minus_u);
    float disp_val_plus_v  = getDisplacementValue(n_p_plus_v);
    float disp_val_minus_v = getDisplacementValue(n_p_minus_v);

    // Calculamos los nuevos puntos post desplazamiento
    vec3 displaced_p_plus_u  = p_plus_u  + n_p_plus_u  * disp_val_plus_u  * effective_disp_strength;
    vec3 displaced_p_minus_u = p_minus_u + n_p_minus_u * disp_val_minus_u * effective_disp_strength;
    vec3 displaced_p_plus_v  = p_plus_v  + n_p_plus_v  * disp_val_plus_v  * effective_disp_strength;
    vec3 displaced_p_minus_v = p_minus_v + n_p_minus_v * disp_val_minus_v * effective_disp_strength;

    // Esta seria la derivada o pendiente en las direcciones de los nuevos ejes
    vec3 dPdu = (displaced_p_plus_u - displaced_p_minus_u) / (2.0 * epsilon);
    vec3 dPdv = (displaced_p_plus_v - displaced_p_minus_v) / (2.0 * epsilon);

    // Nueva normal del punto desplazado (perpendicular a ambas pendientes)
    vec3 displaced_normal = normalize(cross(dPdu, dPdv));
    
    // Si apunta hacia adentro la nueva normal, hacer que apunte hacia afuera
    if (dot(displaced_normal, initial_normal_model) < 0.0) {
        displaced_normal = -displaced_normal;
    }
    return displaced_normal;
}


void main() {
    vec3 p0_model_patch = modelPos_tes_patch[0];
    vec3 p1_model_patch = modelPos_tes_patch[1];
    vec3 p2_model_patch = modelPos_tes_patch[2];

    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    float w = gl_TessCoord.z;

    vec3 interpolatedPos_model = u * p0_model_patch + v * p1_model_patch + w * p2_model_patch; // Calculamos la posicion del nuevo vertice interpolando
    
    vec3 initial_normal_model = normalize(interpolatedPos_model);
    if (length(interpolatedPos_model) < 0.0001)
        initial_normal_model = vec3(0,1,0);
    
    float actual_displacement_strength = displacementStrength_tes * displacementFactor_tes_patch; // Tenemos en cuenta la distancia de la camara
    
    float displacement_value = getDisplacementValue(initial_normal_model); // Aqui llamamos a la funcion de ondas para el vertice
    
    vec3 displacedPos_model = interpolatedPos_model + initial_normal_model * displacement_value * actual_displacement_strength; // Inicial mas el displacement calculado
    
    vec3 displaced_normal_model = calculateDisplacedNormal_model(interpolatedPos_model, initial_normal_model, actual_displacement_strength); // Calculo de la normal nueva (la originial no vale ya)
    
    tes_out.FragPos_world = vec3(instanceMatrix_tes_patch * vec4(displacedPos_model, 1.0)); // Transformar posicion del vertice desplazado segun la matriz de la instancia

    // Normal en el mundo
    mat3 M_model_inv_T = transpose(inverse(mat3(instanceMatrix_tes_patch)));
    tes_out.Normal_world = normalize(M_model_inv_T * displaced_normal_model); // pAsamos la normal para la luz

    // Posicion final despues de aplicar vista y proyeccion
    gl_Position = projectionMatrix * viewMatrix * vec4(tes_out.FragPos_world, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 410 core
out vec4 FragColor;

in TES_FS_INTERFACE {
    vec3 FragPos_world;
    vec3 Normal_world;
} fs_in;

// Variables de la luz y color
uniform vec3 lightPos_world;
uniform vec3 viewPos_world;
uniform vec3 lightColor;
uniform vec3 asteroidBaseColor;

void main() {
    float ambientStrength = 0.20;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(fs_in.Normal_world);
    vec3 lightDir = normalize(lightPos_world - fs_in.FragPos_world);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.25;
    float shininess = 8.0;
    vec3 viewDir = normalize(viewPos_world - fs_in.FragPos_world);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 resultingColor = (ambient + diffuse + specular) * asteroidBaseColor; // Combinamos componentes de la luz
    
    FragColor = vec4(resultingColor, 1.0); // Asignamos color final
}
)";

int main() {
    if (!glfwInit()) { 
        std::cerr << "Fallo al iniciar GLFW" << std::endl; 
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Asteroides", NULL, NULL);
    if (!window) { 
        std::cerr << "Fallo al crear ventana GLFW" << std::endl; glfwTerminate(); 
        return -1; 
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    if (glewInit() != GLEW_OK) { 
        std::cerr << "Fallo al inicializar GLEW" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    GLuint shaderProgram = LoadShaders(vertexShaderSource, tessControlShaderSource, tessEvalShaderSource, fragmentShaderSource);
    glUseProgram(shaderProgram);

    std::vector<glm::vec3> base_vertices;
    generarVerticesIcosfera(base_vertices, RADIO_INICIAL_ASTERIODE);
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, base_vertices.size() * sizeof(glm::vec3), base_vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0); glBindVertexArray(0);

    M.resize(NUM_ASTEROIDES);
    pos_asteroides.resize(NUM_ASTEROIDES);
    R.resize(NUM_ASTEROIDES);
    velocidad_direccion_asteroides.resize(NUM_ASTEROIDES);
    eje_rotacion_asteroides.resize(NUM_ASTEROIDES);
    magnitud_rotacion_asteroides.resize(NUM_ASTEROIDES);
    escalas_asteroides.resize(NUM_ASTEROIDES);

    pointVec centros_asteroides_kdtree;
    centros_asteroides_kdtree.reserve(NUM_ASTEROIDES);
    srand(static_cast<unsigned int>(glfwGetTime()));

    for (int i = 0; i < NUM_ASTEROIDES; ++i) {

		// Generar posición, orientación y escala aleatorias para cada asteroide
        pos_asteroides[i] = glm::sphericalRand(RADIO_CAMPO_ASTEROIDES * glm::linearRand(0.5f, 1.0f)); // Posición aleatoria en el campo (esfera) de asteroides
        float angulo = glm::linearRand(0.0f, 2.0f * glm::pi<float>()); // Rotación
        glm::vec3 eje_inicial = glm::normalize(glm::vec3(glm::linearRand(-1.0f, 1.0f), glm::linearRand(-1.0f, 1.0f), glm::linearRand(-1.0f, 1.0f))); // Eje
        R[i] = glm::rotate(glm::mat4(1.0f), angulo, eje_inicial);

        float randVal = glm::linearRand(0.0f, 1.0f);
        if (randVal < 0.8f) { // 80% Normales y pequeños
            escalas_asteroides[i] = glm::linearRand(0.2f, 1.5f);
        }
		else if (randVal < 0.95f) { // 15% Grandes
            escalas_asteroides[i] = glm::linearRand(1.5f, 4.0f);
        }
		else escalas_asteroides[i] = glm::linearRand(4.0f, 7.0f); // 5% Gigantes

        M[i] = glm::translate(glm::mat4(1.0f), pos_asteroides[i]) * R[i] * glm::scale(glm::mat4(1.0f), glm::vec3(escalas_asteroides[i]));

        // Guardamos las coordenadas del asteroide en el KDTree
        centros_asteroides_kdtree.push_back({ (double)pos_asteroides[i].x, (double)pos_asteroides[i].y, (double)pos_asteroides[i].z });

        velocidad_direccion_asteroides[i] = glm::sphericalRand(1.0f);
        eje_rotacion_asteroides[i] = glm::normalize(glm::vec3(glm::linearRand(-1.0f, 1.0f), glm::linearRand(-1.0f, 1.0f), glm::linearRand(-1.0f, 1.0f)));
        if (glm::length(eje_rotacion_asteroides[i]) < 0.001f) eje_rotacion_asteroides[i] = glm::vec3(0.0f, 1.0f, 0.0f);
        magnitud_rotacion_asteroides[i] = glm::linearRand(0.5f, 1.0f);
    }

    if (!centros_asteroides_kdtree.empty()) {
        kdtree_asteroide = std::make_unique<KDTree>(centros_asteroides_kdtree);
    }

    GLuint instanceMatrixVBO;
    glGenBuffers(1, &instanceMatrixVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceMatrixVBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_ASTEROIDES * sizeof(glm::mat4), M.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(VAO);
    GLsizei vec4Size = sizeof(glm::vec4);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)0);
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(1 * vec4Size));
    glEnableVertexAttribArray(3); glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(2 * vec4Size));
    glEnableVertexAttribArray(4); glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(3 * vec4Size));
    glVertexAttribDivisor(1, 1); glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1); glVertexAttribDivisor(4, 1);
    glBindVertexArray(0);

    glPatchParameteri(GL_PATCH_VERTICES, 3); // TriángUlos

    // Variables a los shaders
    GLint maxOuterLoc = glGetUniformLocation(shaderProgram, "maxOuterLodLevel");
    GLint minOuterLoc = glGetUniformLocation(shaderProgram, "minOuterLodLevel");
    GLint maxInnerLoc = glGetUniformLocation(shaderProgram, "maxInnerLodLevel");
    GLint minInnerLoc = glGetUniformLocation(shaderProgram, "minInnerLodLevel");
    GLint lodNearLoc = glGetUniformLocation(shaderProgram, "lodNearDistance");
    GLint lodFarLoc = glGetUniformLocation(shaderProgram, "lodFarDistance");
    GLint cameraPosLoc = glGetUniformLocation(shaderProgram, "cameraPosition_world");
    GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos_world");
    GLint viewPosWorldLocFS = glGetUniformLocation(shaderProgram, "viewPos_world");
    GLint lightColorLoc = glGetUniformLocation(shaderProgram, "lightColor");
    GLint asteroidBaseColorLoc = glGetUniformLocation(shaderProgram, "asteroidBaseColor");
    GLint dispStrLoc = glGetUniformLocation(shaderProgram, "displacementStrength_tes");
    GLint viewMatrixLoc = glGetUniformLocation(shaderProgram, "viewMatrix");
    GLint projectionMatrixLoc = glGetUniformLocation(shaderProgram, "projectionMatrix");

    glm::vec3 lightPos_world_val(RADIO_CAMPO_ASTEROIDES * 0.7f, RADIO_CAMPO_ASTEROIDES * 0.6f, RADIO_CAMPO_ASTEROIDES * 2.5f);
    glm::vec3 lightColor_val(1.0f, 1.0f, 0.95f);
    glm::vec3 colorAsteroides_val(110.0f / 255.0f, 85.0f / 255.0f, 65.0f / 255.0f); // Marrón

    int nbFrames = 0;
    int asteroidesVisibles = 0;
    lastFPSTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double currentFrameTime = glfwGetTime();
        nbFrames++;
        if (currentFrameTime - lastFPSTime >= 1.0) {
            char title[256];
            sprintf(title, "Asteroides - FPS: %d - Visible: %d/%d", nbFrames, asteroidesVisibles, NUM_ASTEROIDES);
            glfwSetWindowTitle(window, title);
            nbFrames = 0; lastFPSTime = currentFrameTime;
        }

        // Posición y rotación de los asteroides
        for (int i = 0; i < NUM_ASTEROIDES; ++i) {
            pos_asteroides[i] += velocidad_direccion_asteroides[i] * 0.002f;
            float rotacion = magnitud_rotacion_asteroides[i] * 0.002f;
            R[i] = glm::rotate(R[i], rotacion, eje_rotacion_asteroides[i]);

            glm::mat4 T = glm::translate(glm::mat4(1.0f), pos_asteroides[i]);
            glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(escalas_asteroides[i]));
            M[i] = T * R[i] * S;
        }

        glClearColor(0.005f, 0.005f, 0.01f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);

        glm::mat4 P = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.1f, RADIO_CAMPO_ASTEROIDES * 8.0f);
        glm::mat4 V = glm::lookAt(pos_obs, pos_obs + target, obs_up);
        campoVision.update(P * V);

		// Actualizar KDTree
        point_t camera_pos_kdtree_pt = { (double)pos_obs.x, (double)pos_obs.y, (double)pos_obs.z };
        double search_radius = RADIO_CAMPO_ASTEROIDES * 8.0f;
        indexArr indices_candidatos;
        if (kdtree_asteroide) {
            indices_candidatos = kdtree_asteroide->neighborhood_indices(camera_pos_kdtree_pt, search_radius);
        }
        else {
            indices_candidatos.resize(NUM_ASTEROIDES);
            if (!indices_candidatos.empty()) std::iota(indices_candidatos.begin(), indices_candidatos.end(), 0);
        }

		// Filtrar asteroides visibles
        std::vector<glm::mat4> matrices_instancias_visibles;
        if (!indices_candidatos.empty()) {
            matrices_instancias_visibles.reserve(indices_candidatos.size());
            for (size_t i : indices_candidatos) {
                if (i < pos_asteroides.size() && i < escalas_asteroides.size()) {
                    const glm::vec3& current_asteroid_pos = pos_asteroides[i];
                    float escala_actual = escalas_asteroides[i];
                    float radio_asteroide_real = RADIO_INICIAL_ASTERIODE * escala_actual;
                    if (campoVision.isSphereInside(current_asteroid_pos, radio_asteroide_real)) {
                        if (i < M.size()) {
                            matrices_instancias_visibles.push_back(M[i]);
                        }
                    }
                }
            }
        }
        asteroidesVisibles = static_cast<int>(matrices_instancias_visibles.size());

        glUniform1i(maxOuterLoc, maxDetalleExterno); glUniform1i(minOuterLoc, minDetalleExterno);
        glUniform1i(maxInnerLoc, maxDetalleInterno); glUniform1i(minInnerLoc, minDetalleInterno);
        glUniform1f(lodNearLoc, distanciaDetalleCerca); glUniform1f(lodFarLoc, distanciaDetalleLejos);
        glUniform3fv(cameraPosLoc, 1, glm::value_ptr(pos_obs));
        glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos_world_val));
        glUniform3fv(viewPosWorldLocFS, 1, glm::value_ptr(pos_obs));
        glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor_val));
        glUniform3fv(asteroidBaseColorLoc, 1, glm::value_ptr(colorAsteroides_val));
        glUniform1f(dispStrLoc, intensidadDesplazamiento);
        glUniformMatrix4fv(viewMatrixLoc, 1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(projectionMatrixLoc, 1, GL_FALSE, glm::value_ptr(P));

        if (!matrices_instancias_visibles.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, instanceMatrixVBO);
            glBufferData(GL_ARRAY_BUFFER, matrices_instancias_visibles.size() * sizeof(glm::mat4), matrices_instancias_visibles.data(), GL_DYNAMIC_DRAW);

            glBindVertexArray(VAO);
            glDrawArraysInstanced(GL_PATCHES, 0, static_cast<GLsizei>(base_vertices.size()), static_cast<GLsizei>(matrices_instancias_visibles.size()));
            glBindVertexArray(0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &instanceMatrixVBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
}

// Teclado
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {

		// Intensidad de desplazamiento
        if (key == GLFW_KEY_Z) intensidadDesplazamiento = std::max(0.0f, intensidadDesplazamiento - 0.05f);
        else if (key == GLFW_KEY_X) intensidadDesplazamiento = intensidadDesplazamiento + 0.05f;

		// Movimiento
        else if (key == GLFW_KEY_W) pos_obs += VELOCIDAD_MOVIMIENTO_CAMARA * target;
        else if (key == GLFW_KEY_S) pos_obs -= VELOCIDAD_MOVIMIENTO_CAMARA * target;
        else if (key == GLFW_KEY_A) pos_obs -= glm::normalize(glm::cross(target, obs_up)) * VELOCIDAD_MOVIMIENTO_CAMARA;
        else if (key == GLFW_KEY_D) pos_obs += glm::normalize(glm::cross(target, obs_up)) * VELOCIDAD_MOVIMIENTO_CAMARA;

        // Rotación
        else if (key == GLFW_KEY_LEFT) az -= VELOCIDAD_ROTACION_CAMARA;
        else if (key == GLFW_KEY_RIGHT) az += VELOCIDAD_ROTACION_CAMARA;
        else if (key == GLFW_KEY_UP) elev += VELOCIDAD_ROTACION_CAMARA;
        else if (key == GLFW_KEY_DOWN) elev -= VELOCIDAD_ROTACION_CAMARA;

		glm::vec3 front_calc;
		front_calc.x = cos(glm::radians(az)) * cos(glm::radians(elev));
		front_calc.y = sin(glm::radians(elev));
		front_calc.z = sin(glm::radians(az)) * cos(glm::radians(elev));
		target = glm::normalize(front_calc);
		glm::vec3 right_vec = glm::normalize(glm::cross(target, up));
		obs_up = glm::normalize(glm::cross(right_vec, target));

        static bool wireframe_mode = false;
        if (key == GLFW_KEY_V && action == GLFW_PRESS) { // Cambiar modo alambre o relleno
            wireframe_mode = !wireframe_mode;
            if (wireframe_mode) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
    }
}

// Función para generar los vértices de la icosfera (basado en https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/)
void generarVerticesIcosfera(std::vector<glm::vec3>& vertices_out, float radius) {
    const float X = .525731112119133606f;
    const float Z = .850650808352039932f;
    const float N = 0.f;

    static const glm::vec3 vertices[12] = {
      {-X,N,Z}, {X,N,Z}, {-X,N,-Z}, {X,N,-Z},
      {N,Z,X}, {N,Z,-X}, {N,-Z,X}, {N,-Z,-X},
      {Z,X,N}, {-Z,X, N}, {Z,-X,N}, {-Z,-X, N}
    };

    static const GLuint idx[60] = {
        0,4,1,  0,9,4,  9,5,4,  4,5,8,  4,8,1,
        8,10,1, 8,3,10, 5,3,8,  5,2,3,  2,7,3,
        7,10,3, 7,6,10, 7,11,6, 11,0,6, 0,1,6,
        6,1,10, 9,0,11, 9,11,2, 9,2,5,  7,2,11
    };
    vertices_out.clear();
    vertices_out.reserve(60);
    for (int i = 0; i < 60; ++i) {
        vertices_out.push_back(vertices[idx[i]] * radius);
    }
}

// Función para manejar redimensionamiento de ventana
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Función para compilar y enlazar shaders
GLuint LoadShaders(const char* vertexShaderSource, const char* tessControlShaderSource,
    const char* tessEvalShaderSource, const char* fragmentShaderSource) {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint tessControlShader = glCreateShader(GL_TESS_CONTROL_SHADER);
    glShaderSource(tessControlShader, 1, &tessControlShaderSource, NULL);
    glCompileShader(tessControlShader);

    GLuint tessEvalShader = glCreateShader(GL_TESS_EVALUATION_SHADER);
    glShaderSource(tessEvalShader, 1, &tessEvalShaderSource, NULL);
    glCompileShader(tessEvalShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, tessControlShader);
    glAttachShader(shaderProgram, tessEvalShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(tessControlShader);
    glDeleteShader(tessEvalShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}