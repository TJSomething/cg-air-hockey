#include <GL/glew.h> // glew must be included before the main gl libs
#include <GL/glut.h> // doing otherwise causes compiler shouting
#include <iostream>
#include <chrono>
#include <deque>
#include <array>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <set>
#include <string>
#include <map>

#include <Box2D/Box2D.h>

#include <boost/filesystem/convenience.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/ring.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/multi/geometries/multi_point.hpp>
#include <boost/geometry/views/box_view.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> //Makes passing matrices to shaders easier

#include <assimp/Importer.hpp>

#include <stb_image.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "read_file.h"
#include "make_unique.hpp"

BOOST_GEOMETRY_REGISTER_POINT_2D(b2Vec2, float32, cs::cartesian, x, y)


namespace fs = boost::filesystem;
namespace gil = boost::gil;
namespace geo = boost::geometry;

typedef geo::model::polygon<b2Vec2> polygon;

//--Data types
//This object will define the attributes of a vertex(position, color, etc...)
struct Vertex
{
    GLfloat position[3];
    GLfloat color[3];
    GLfloat tex_coord[2];
    GLfloat tex_opacity;
    GLfloat normal[3];
    GLfloat ambient[3];
    GLfloat diffuse[3];
    GLfloat specular[3];
    GLfloat emission[3];
    GLfloat shininess;
};

enum Material {
    BLUE_PLASTIC,
    TEXTURED_PLASTIC,
};

struct Model {
    Model();
    void setMaterial(Material);
    GLuint geometryVBO;
    GLuint textureVBO;
    glm::mat4 modelMatrix;
    std::list<Model> children;
    gil::rgba8_image_t texture;
    std::vector<Vertex> geometry;
    GLenum drawMode;
};

std::list<Model> geometryRoot;

namespace Models {
    Model *puck, *table, *paddle1, *paddle2;
}

// Things that we need to know
GLfloat puckY, paddleY, puckYStart, puckYVel = 0;
const int LIGHT_COUNT = 1;
GLfloat puckRadius, paddleRadius;
const GLfloat goalWidthFraction = 0.3;
GLfloat tableLeft, tableRight;
b2Vec2 tableCenter;
int score[2] = {0,0};
bool matchOver = false;
GLfloat theta = -M_PI/2, phi = M_PI/4, r = 10;

//--Evil Global variables
int w = 640, h = 480;// Window size
GLuint program;// The GLSL program handle

//uniform locations
GLint loc_mvpmat;// Location of the modelviewprojection matrix in the shader
GLint loc_lights[LIGHT_COUNT];
GLint loc_to_world_mat;

//attribute locations
GLint loc_position;
GLint loc_color;
GLint loc_tex_coord;
GLint loc_tex_opacity;
GLint loc_texmap;
GLint loc_normal;
GLint loc_ambient;
GLint loc_diffuse;
GLint loc_specular;
GLint loc_shininess;
GLint loc_light_colors;
GLint loc_light_positions;
GLint loc_cam_position;

// Lighting info
glm::vec3 lightColors[LIGHT_COUNT];
glm::vec3 lightPosition[LIGHT_COUNT];
glm::vec3 cameraPosition;

//transform matrices
namespace mats {
    glm::mat4 view;//world->eye
    glm::mat4 projection;//eye->clip
    glm::mat4 toWorld;
}

#define BALL_COLOR {0.00, 0.00, 0.8}

//--GLUT Callbacks
void render();
void update();
void reshape(int n_w, int n_h);
void keyboard(unsigned char key, int x_pos, int y_pos);
void keyboardUp(unsigned char key, int x_pos, int y_pos);
void passiveMotion(int, int);
void specialKey(int, int, int);
void specialKeyUp(int key, int x, int y);
void mouse(int button, int state, int x, int y);

//--Resource management
bool initialize();
void cleanUp();

//--Random time things
float getDT();
std::chrono::time_point<std::chrono::high_resolution_clock> t1,t2;

// Physics
namespace phys {
    b2World world( b2Vec2_zero );
    b2Body* puck;
    b2Body* paddle1;
    b2Body* paddle2;
    b2Body* table;
}

bool captureMouse = false;

// Controls
bool specialKeys[256];
bool keys[256];
int mouseX = 0, mouseY = 0;
float keyboardSensitivity = 5.0f;
float mouseSensitivity = 0.005f;
bool leftMouseButton = false;
bool rightMouseButton = false;

std::mt19937 gen(std::random_device().operator()());

// Utility functions
Vertex makeVertex(Material m, glm::vec3 pos, glm::vec3 normal);
void addTriangle(std::vector<Vertex> &geometry,
        Material m1, glm::vec3 pos1,
        Material m2, glm::vec3 pos2,
        Material m3, glm::vec3 pos3);
GLint getGLLocation(const std::string &name, bool isUniform) ;
Model convertAssimpScene (const aiScene &sc);
void forAllModels(std::function<void(Model&)>);
void forChildModels(std::list<Model*>, std::function<void (Model &)>);

//--Main
int main(int argc, char **argv)
{
    // Initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(w, h);
    // Name and create the Window
    glutCreateWindow("Air Hockey");

    // Now that the window is created the GL context is fully set up
    // Because of that we can now initialize GLEW to prepare work with shaders
    GLenum status = glewInit();
    if( status != GLEW_OK)
    {
        std::cerr << "[F] GLEW NOT INITIALIZED: ";
        std::cerr << glewGetErrorString(status) << std::endl;
        return -1;
    }

    // Set all of the callbacks to GLUT that we need
    glutDisplayFunc(render);// Called when its time to display
    glutReshapeFunc(reshape);// Called if the window is resized
    glutIdleFunc(update);// Called if there is nothing else to do
    glutKeyboardFunc(keyboard);// Called if there is keyboard input
    glutKeyboardUpFunc(keyboardUp);
    glutPassiveMotionFunc(passiveMotion);
    glutSpecialFunc(specialKey);
    glutSpecialUpFunc(specialKeyUp);
    glutMouseFunc(mouse);

    // To get our textures and shaders, we should be in the executable directory
    fs::current_path(fs::path(argv[0]).parent_path());

    // Initialize all of our resources(shaders, geometry)
    bool init = initialize();
    if(init)
    {
        t1 = std::chrono::high_resolution_clock::now();
        glutMainLoop();
        // Clean up after ourselves
        cleanUp();
    } else {
        std::cout << "Initialization failure" << std::endl;
    }

    return 0;
}

void initKeys() {
    for ( auto &key : specialKeys ) {
        key = false;
    }
    for ( auto &key : keys ) {
        key = false;
    }
}

void attachModelToBuffer(GLuint &vbo, const std::vector<Vertex> &model) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //std::cout << model.size() << std::endl;
    /*for (auto v : model) {
        std::cout << v.position[0] << std::endl;
    }*/
    glBufferData(GL_ARRAY_BUFFER,
            model.size()*sizeof(Vertex),
            model.data(),
            GL_STATIC_DRAW);
}

bool initShaders() {
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

    //Shader Sources
    std::string vs = read_file("vertex.S");
    auto vsPtr = vs.c_str();

    std::string fs = read_file("fragment.S");
    auto fsPtr = fs.c_str();

    //compile the shaders
    GLint shader_status;

    // Vertex shader first
    glShaderSource(vertex_shader, 1, &vsPtr, NULL);
    glCompileShader(vertex_shader);
    //check the compile status
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &shader_status);
    if(!shader_status)
    {
        std::cerr << "[F] FAILED TO COMPILE VERTEX SHADER!" << std::endl;

        int errorLength;
        std::unique_ptr<char[]> error;
        glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &errorLength);

        error = std::unique_ptr<char[]>(new char[errorLength]);
        glGetShaderInfoLog(vertex_shader, errorLength, nullptr, error.get());

        std::cout << error.get() << std::endl;

        return false;
    }

    // Now the Fragment shader
    glShaderSource(fragment_shader, 1, &fsPtr, NULL);
    glCompileShader(fragment_shader);
    //check the compile status
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &shader_status);
    if(!shader_status)
    {
        std::cerr << "[F] FAILED TO COMPILE FRAGMENT SHADER!" << std::endl;

        int errorLength;
        std::unique_ptr<char[]> error;
        glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &errorLength);

        error = std::unique_ptr<char[]>(new char[errorLength]);
        glGetShaderInfoLog(fragment_shader, errorLength, nullptr, error.get());

        std::cout << error.get() << std::endl;

        return false;
    }

    //Now we link the 2 shader objects into a program
    //This program is what is run on the GPU
    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    //check if everything linked ok
    glGetProgramiv(program, GL_LINK_STATUS, &shader_status);
    if(!shader_status)
    {
        std::cerr << "[F] THE SHADER PROGRAM FAILED TO LINK" << std::endl;
        return false;
    }

    return true;
}

void initShaderInputLocations() {
    //Now we set the locations of the attributes and uniforms
    //this allows us to access them easily while rendering
    loc_position = getGLLocation("v_position", false);

    loc_color = getGLLocation("v_color", false);

    loc_tex_coord = getGLLocation("v_tex_coord", false);

    loc_tex_opacity = getGLLocation("v_tex_opacity", false);

    loc_mvpmat = getGLLocation("mvpMatrix", true);

    loc_texmap = getGLLocation("texMap", true);
    glUniform1i(loc_texmap, 0);

    loc_normal = getGLLocation("v_normal", false);
    loc_ambient = getGLLocation("v_ambient", false);
    loc_diffuse = getGLLocation("v_diffuse", false);
    loc_specular = getGLLocation("v_specular", false);
    loc_shininess = getGLLocation("v_shininess", false);
    loc_light_colors = getGLLocation("lightColor", true);
    loc_light_positions = getGLLocation("lightPositions", true);
    loc_cam_position = getGLLocation("cameraPosition", true);
    loc_to_world_mat = getGLLocation("toWorldMatrix", true);
}

void initTextures() {
    // Bind textures
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    forAllModels([&](Model& current) {
        glGenTextures(1, &current.textureVBO);
        glBindTexture(GL_TEXTURE_2D, current.textureVBO);

        // At repeat seam, textures are reflected. I just assume this is true
        // for all textures. It usually looks pretty nice.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR_MIPMAP_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, current.texture.width(),
                current.texture.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                gil::interleaved_view_get_raw_data(gil::view(current.texture)));
        glGenerateMipmap(GL_TEXTURE_2D);
    });
}

void updateVPMatrices() {
	auto center = glm::vec3(tableCenter.x, puckYStart, tableCenter.y);
	cameraPosition = center + glm::vec3(r*cos(theta)*cos(phi), r*sin(phi), r*sin(theta)*cos(phi));
    //--Init the view and projection matrices
    //  if you will be having a moving camera the view matrix will need to more dynamic
    //  ...Like you should update it before you render more dynamic
    //  for this project having them static will be fine
    mats::view = glm::lookAt( cameraPosition, //Eye Position
                        glm::vec3(tableCenter.x, puckYStart, tableCenter.y), //Focus point
                        glm::vec3(0.0, 1.0, 0.0)); //Positive Y is up

    mats::projection = glm::perspective( 45.0f, //the FoV typically 90 degrees is good which is what this is set to
                                   float(w)/float(h), //Aspect Ratio, so Circles stay Circular
                                   0.01f, //Distance to the near plane, normally a small value like this
                                   100.0f); //Distance to the far plane,

    mats::toWorld = glm::inverse(mats::projection * mats::view);
}

void resetPuck() {
    puckY = puckYStart;
    puckYVel = 0;
    phys::puck->SetTransform(tableCenter, 0);
    phys::puck->ApplyLinearImpulse(
            -1.5*phys::puck->GetMass()*phys::puck->GetLinearVelocity(), phys::puck->GetWorldCenter());
    if (!matchOver)
        phys::puck->ApplyLinearImpulse(
                phys::puck->GetMass()*b2Vec2{2,1}, phys::puck->GetWorldCenter());
    matchOver = false;
}

void initPhysics() {
    b2FixtureDef fixtureDef;

	// Find the table's height
	polygon tablePoints;
	geo::model::ring<b2Vec2> tableEdge;
	polygon lipPoints;
	std::vector<b2Vec2> innerLipPoints;
	std::set<GLfloat> tableHeights;
	forChildModels(std::list<Model*>{Models::table},
		[&](Model &m) {
			for (auto pt : m.geometry) {
				if (fabs(pt.normal[0]) < 0.01f && pt.normal[1] > 0.99f && fabs(pt.normal[2]) < 0.01f)
					tableHeights.insert(pt.position[1]);
			}
		});


	// And the height of the table
	auto heightFinder = tableHeights.rbegin();
	GLfloat maxHeight = *heightFinder;
	while (*heightFinder > maxHeight - 0.1)
		heightFinder++;
	GLfloat tableHeight = *heightFinder;

	// Then the edge of the table
	forChildModels(std::list<Model*>{Models::table},
		[&](Model &m) {
			for (auto pt : m.geometry) {
				if (abs(pt.position[1]-tableHeight) < 0.01)
					tablePoints.outer().push_back(
							b2Vec2{pt.position[0], pt.position[2]});

				if (pt.position[1] > tableHeight + 0.1)
					lipPoints.outer().push_back(
							b2Vec2{pt.position[0], pt.position[2]});
			}
		});

	// Get the outside points of the surface
	polygon tablePoints2 = tablePoints;
	tablePoints.clear();
	geo::convex_hull(tablePoints2, tablePoints);

	// Get center of the center of the table
	geo::centroid(tablePoints, tableCenter);

	// Find the points of the lip that are above the table
	for(auto pt : lipPoints.outer()) {
		if (geo::within(pt, tablePoints))
			innerLipPoints.push_back(pt);
	}

	// To make other functions better behaved, sort the points by their
	//  angle from the center of the table
	std::sort(innerLipPoints.begin(), innerLipPoints.end(),
			[=](b2Vec2 pt1, b2Vec2 pt2){
				b2Vec2 diff1 = pt1 - tableCenter;
				b2Vec2 diff2 = pt2 - tableCenter;
				return atan2(diff1.y, diff1.x) < atan2(diff2.y, diff2.x);
			});
	geo::model::ring<b2Vec2> innerLip(innerLipPoints.begin(),
			innerLipPoints.end());
	geo::correct(innerLip);
	geo::convex_hull(innerLip, tableEdge);

	// We need to fix some table normals
	forChildModels(std::list<Model*>{Models::table},
		[&](Model &m) {
			for (auto &pt : m.geometry) {
				if (fabs(pt.position[1]-tableHeight) < 0.01) {
					pt.normal[0] = 0.0;
					pt.normal[1] = 1.0;
					pt.normal[2] = 0.0;
				}
			}
			m.drawMode = GL_TRIANGLES;
		});

    forAllModels([&](Model& current) {
        attachModelToBuffer(current.geometryVBO, current.geometry);
    });

	// Cut out the goals from the edge
	geo::model::box<b2Vec2> tableBounds;
	geo::envelope(tableEdge, tableBounds);
	polygon goalMask[2];

	goalMask[0].outer().push_back(
		b2Vec2{2.0f*(tableBounds.min_corner().x - tableCenter.x),
		       2.0f*(tableBounds.max_corner().y - tableCenter.y)} +
		tableCenter);
	goalMask[0].outer().push_back(
		b2Vec2{2.0f*(tableBounds.min_corner().x - tableCenter.x),
		       -(goalWidthFraction)*(tableBounds.max_corner().y - tableCenter.y)} +
		tableCenter);
	goalMask[0].outer().push_back(
		b2Vec2{2.0f*(tableBounds.max_corner().x - tableCenter.x),
		       -(goalWidthFraction)*(tableBounds.max_corner().y - tableCenter.y)} +
		tableCenter);
	goalMask[0].outer().push_back(
		b2Vec2{2.0f*(tableBounds.max_corner().x - tableCenter.x),
		       2.0f*(tableBounds.max_corner().y - tableCenter.y)} +
		tableCenter);

	goalMask[1].outer().push_back(
		b2Vec2{2.0f*(tableBounds.max_corner().x - tableCenter.x),
		       2.0f*(tableBounds.min_corner().y - tableCenter.y)} +
		tableCenter);
	goalMask[1].outer().push_back(
		b2Vec2{2.0f*(tableBounds.max_corner().x - tableCenter.x),
		       -(goalWidthFraction)*(tableBounds.min_corner().y - tableCenter.y)} +
		tableCenter);
	goalMask[1].outer().push_back(
		b2Vec2{2.0f*(tableBounds.min_corner().x - tableCenter.x),
		       -(goalWidthFraction)*(tableBounds.min_corner().y - tableCenter.y)} +
		tableCenter);
	goalMask[1].outer().push_back(
		b2Vec2{2.0f*(tableBounds.min_corner().x - tableCenter.x),
		       2.0f*(tableBounds.min_corner().y - tableCenter.y)} +
		tableCenter);

	std::vector<polygon> tableSides[2];
	geo::intersection(tableEdge, goalMask[0], tableSides[0]);
	geo::intersection(tableEdge, goalMask[1], tableSides[1]);

    // Rotate the arrays such that the insides don't block the puck
    std::stable_sort(tableSides[0][0].outer().begin(),
    		tableSides[0][0].outer().end(),
			[=](b2Vec2 pt1, b2Vec2 pt2){return pt1.x<pt2.x;});
    std::stable_sort(tableSides[1][0].outer().begin(),
    		tableSides[1][0].outer().end(),
			[=](b2Vec2 pt1, b2Vec2 pt2){return pt1.x<pt2.x;});

	// We also need to bind the paddles to their sides
	geo::model::box<b2Vec2> sideMask[2];
	std::vector<polygon> paddleSides[2];

	sideMask[0].max_corner() = tableBounds.max_corner();
	sideMask[0].min_corner() =
			b2Vec2{tableCenter.x, tableBounds.min_corner().y};
	geo::correct(sideMask[0]);

	sideMask[1].max_corner() =
			b2Vec2{tableCenter.x, tableBounds.max_corner().y};
	sideMask[1].min_corner() = tableBounds.max_corner();
	geo::correct(sideMask[1]);

	for (int i = 0; i < 2; i++) {
	    auto boxView = geo::box_view<geo::model::box<b2Vec2>>(sideMask[i]);
		geo::intersection(tableEdge, sideMask[i], paddleSides[i]);
	}

	// Remember the sides of the table
	tableLeft = tableBounds.min_corner().x;
	tableRight = tableBounds.max_corner().x;

    // Puck
	// Find the puck's radius by finding the point on the edge furthest from
	//  the center
	polygon puckPoints, puckEdge;
	b2Vec2 puckCenter;
	GLfloat puckBottom = 1000.0;
	forChildModels(std::list<Model*>{Models::puck},
		[&](Model &m) {
			for (auto pt : m.geometry) {
				puckPoints.outer().push_back(
						b2Vec2{pt.position[0], pt.position[2]});
				puckBottom = std::min(puckBottom, pt.position[1]);
			}
		});

	puckYStart = tableHeight - puckBottom;

	geo::convex_hull(puckPoints, puckEdge);
	geo::centroid(puckEdge, puckCenter);
	puckRadius = 0;
	for (auto pt : puckEdge.outer()) {
		puckRadius = std::max(puckRadius, b2Distance(pt, puckCenter));
	}

    b2BodyDef puckBodyDef;
    b2CircleShape puckShape;

    puckBodyDef.type = b2_dynamicBody;
    puckBodyDef.linearDamping = 0.001;
    puckBodyDef.position = tableCenter;
    phys::puck = phys::world.CreateBody(&puckBodyDef);

    puckShape.m_radius = puckRadius;

    fixtureDef.shape = &puckShape;
    fixtureDef.density = 1.0f;
    fixtureDef.friction = 0.001f;
    fixtureDef.restitution = 1.0f;
    fixtureDef.filter.categoryBits = 0x4;
    fixtureDef.filter.maskBits = 0xa;

    phys::puck->CreateFixture(&fixtureDef);

    resetPuck();

    // Table
    b2BodyDef tableBodyDef;
    phys::table = phys::world.CreateBody(&tableBodyDef);

    std::vector<b2ChainShape> tableSideShapes(2);
    std::vector<b2ChainShape> paddleBoundaryShapes(2);
    for (int i = 0; i < 2; i++) {
		tableSideShapes[i].CreateChain(tableSides[i][0].outer().data(),
				tableSides[i][0].outer().size());
        fixtureDef.shape = &tableSideShapes[i];
        fixtureDef.density = 0.0;
        fixtureDef.filter.categoryBits = 0x2;
        fixtureDef.filter.maskBits = 0xf;
		phys::table->CreateFixture(&fixtureDef);

		paddleBoundaryShapes[i].CreateChain(paddleSides[i][0].outer().data(),
				paddleSides[i][0].outer().size());
		fixtureDef.shape = &paddleBoundaryShapes[i];
        fixtureDef.density = 0.0;
        fixtureDef.filter.categoryBits = 0x1;
        fixtureDef.filter.maskBits = 0x8;
		phys::table->CreateFixture(&fixtureDef);

    }

    // Paddles
	// Find the paddle's radius by finding the point on the edge furthest from
	//  the center
	polygon paddlePoints, paddleEdge;
	b2Vec2 paddleCenter;
	GLfloat paddleBottom = 1000.0;
	forChildModels(std::list<Model*>{Models::paddle1},
		[&](Model &m) {
			for (auto pt : m.geometry) {
				paddlePoints.outer().push_back(
						b2Vec2{pt.position[0], pt.position[2]});
				paddleBottom = std::min(paddleBottom, pt.position[1]);
			}
		});
	// TODO: Save this somewhere for resetting
	paddleY = tableHeight - paddleBottom;

	geo::convex_hull(paddlePoints, paddleEdge);
	geo::centroid(paddleEdge, paddleCenter);
	paddleRadius = 0;
	for (auto pt : paddleEdge.outer()) {
		paddleRadius = std::max(paddleRadius, b2Distance(pt, paddleCenter));
	}

    b2BodyDef paddleBodyDef[2];
    b2CircleShape paddleShape;

    paddleBodyDef[0].type = b2_dynamicBody;
    paddleBodyDef[0].linearDamping = 0;
    paddleBodyDef[0].position = 0.5*(tableCenter +
    		b2Vec2{tableBounds.min_corner().x,tableCenter.y});
    phys::paddle1 = phys::world.CreateBody(&paddleBodyDef[0]);
    paddleBodyDef[1].type = b2_dynamicBody;
    paddleBodyDef[1].linearDamping = 0;
    paddleBodyDef[1].position = 0.5*(tableCenter +
    		b2Vec2{tableBounds.max_corner().x,tableCenter.y});
    phys::paddle2 = phys::world.CreateBody(&paddleBodyDef[1]);

    paddleShape.m_radius = paddleRadius;

    fixtureDef.shape = &paddleShape;
    fixtureDef.density = 0.5f;
    fixtureDef.friction = 0.001f;
    fixtureDef.restitution = 1.0f;
    fixtureDef.filter.categoryBits = 0x8;
    fixtureDef.filter.maskBits = 0xf;

    phys::paddle1->CreateFixture(&fixtureDef);
    phys::paddle2->CreateFixture(&fixtureDef);
}

void initModels () {
    // First, the table

    static const aiScene* tableScene = nullptr;
    static Assimp::Importer Importer;
    if (tableScene == nullptr) {
        tableScene = Importer.ReadFile("boardtest4.obj", aiProcess_Triangulate | aiProcess_FixInfacingNormals | aiProcess_GenSmoothNormals);
    }
    if (tableScene == nullptr) {
        std::cout << Importer.GetErrorString() << std::endl;
        throw 0;
    }
    geometryRoot.emplace_back(convertAssimpScene(*tableScene));
    Models::table = &geometryRoot.back();
    Models::table->modelMatrix = glm::scale(glm::mat4(1.0), glm::vec3(1.0,1.0,1.0));
    //Models::table->modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3{0})
    /*forChildModels(std::list<Model*>{Models::table},
        [](Model &m) {
            gil::png_read_and_convert_image("Material_boardtest.png", Models::table->texture);
            m.setMaterial(BLUE_PLASTIC);
        });*/

    //Puck
    static const aiScene* puckScene = nullptr;
    if (puckScene == nullptr) {
        puckScene = Importer.ReadFile("puck.obj", aiProcess_Triangulate | aiProcess_FindInvalidData | aiProcess_FixInfacingNormals | aiProcess_GenSmoothNormals);
    }
    if (puckScene == nullptr) {
        std::cout << Importer.GetErrorString() << std::endl;
        throw 0;
    }
    geometryRoot.emplace_back(convertAssimpScene(*puckScene));
    Models::puck = &geometryRoot.back();
    Models::puck->modelMatrix = glm::scale(glm::mat4(1.0), glm::vec3(1,1,1));
    forChildModels(std::list<Model*>{Models::puck},
    		[&](Model& m) {
    			for (auto &pt : m.geometry) {
    				pt.position[0] *= 0.1;
    				pt.position[1] *= 0.1;
    				pt.position[2] *= 0.1;
    				pt = makeVertex(BLUE_PLASTIC,
    						glm::vec3{pt.position[0],
    					              pt.position[1],
    					              pt.position[2]},
							glm::vec3{pt.normal[0],
									  pt.normal[1],
									  pt.normal[2]});
    			}
    		});

    //Paddles
    static const aiScene* paddleScene = nullptr;
    if (paddleScene == nullptr) {
    	paddleScene = Importer.ReadFile("paddle.obj", aiProcess_Triangulate | aiProcess_FindInvalidData | aiProcess_FixInfacingNormals | aiProcess_GenSmoothNormals);
    }
    if (paddleScene == nullptr) {
        std::cout << Importer.GetErrorString() << std::endl;
        throw 0;
    }
    geometryRoot.emplace_back(convertAssimpScene(*paddleScene));
    Models::paddle1 = &geometryRoot.back();
    geometryRoot.emplace_back(convertAssimpScene(*paddleScene));
    Models::paddle2 = &geometryRoot.back();
    forChildModels(std::list<Model*>{Models::paddle1, Models::paddle2},
    		[&](Model& m) {
    			for (auto &pt : m.geometry) {
    				pt.position[0] *= 0.1;
    				pt.position[1] *= 0.1;
    				pt.position[2] *= 0.1;
    				pt = makeVertex(BLUE_PLASTIC,
    						glm::vec3{pt.position[0],
    					              pt.position[1],
    					              pt.position[2]},
							glm::vec3{pt.normal[0],
									  pt.normal[1],
									  pt.normal[2]});
    			}
    		});

    // Create a Vertex Buffer object to store these vertex infos on the GPU
    forAllModels([&](Model& current) {
        glGenBuffers(1, &current.geometryVBO);
        attachModelToBuffer(current.geometryVBO, current.geometry);
    });
}

void initLightsAndCamera() {
    // Initialize the lights and camera positions
    cameraPosition = glm::vec3{0.0, 10, -10.0};
    lightPosition[0] =
            glm::vec3{1, 10, 1};
    lightColors[0] = glm::vec3{1.0, 1.0, 1.0};
}

void initGLFlags() {
    //enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Antialiasing
    glEnable(GL_MULTISAMPLE);

    // Allow better keyboard control
    glutIgnoreKeyRepeat(1);
}

bool initialize()
{
    initKeys();

    initModels();

    if (!initShaders()) {
        return false;
    }
    initShaderInputLocations();
    initTextures();
    initLightsAndCamera();
    updateVPMatrices();
    initPhysics();



    initGLFlags();


    //and its done
    return true;
}

void cleanUp()
{
    // Reset stuff
    mouseX = 0;
    mouseY = 0;
    // Clean up, Clean up
    glDeleteProgram(program);
    forAllModels([&](Model& current) {
        glDeleteBuffers(1, &current.geometryVBO);
        glDeleteBuffers(1, &current.textureVBO);
    });
    geometryRoot.clear();
}


void glutPrint(float x, float y, void* font, const char* text, float r, float g,
        float b, float a)
{
    if(!text || !strlen(text)) return;
    bool blending = false;
    if(glIsEnabled(GL_BLEND)) blending = true;
    glEnable(GL_BLEND);
    glColor4f(r,g,b,a);
    int width = glutBitmapLength(font, (unsigned char*) text);
    //printf("%d\n", width);
    glRasterPos2f(x-float(width)/float(w),y);
    while (*text) {
        glutBitmapCharacter(font, *text);
        text++;
    }
    if(!blending) glDisable(GL_BLEND);
}

//--Implementations
void render()
{
    //--Render the scene

    //clear the screen
    glClearColor(0.0, 0.0, 0.2, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);

    //enable the shader program
    glUseProgram(program);

    std::function<void(Model&, glm::mat4)> renderSceneGraph =
        [&](Model& current, glm::mat4 previousTransforms) {
            // Construct matrices
            glm::mat4 modelToWorld = previousTransforms * current.modelMatrix;
            glm::mat4 mvpMatrix = mats::projection * mats::view * modelToWorld;

            // Copy uniforms
            glUniformMatrix4fv(loc_mvpmat, 1, GL_FALSE, glm::value_ptr(mvpMatrix));
            glUniformMatrix4fv(loc_to_world_mat, 1, GL_FALSE,
                    glm::value_ptr(mats::toWorld));
            glUniform3fv(loc_light_colors, LIGHT_COUNT, glm::value_ptr(lightColors[0]));
            glUniform3fv(loc_light_positions, LIGHT_COUNT, glm::value_ptr(lightPosition[0]));
            glUniform3fv(loc_cam_position, 1, glm::value_ptr(cameraPosition));

            // Bind attributes
            glEnableVertexAttribArray(loc_position);
            glEnableVertexAttribArray(loc_color);
            glEnableVertexAttribArray(loc_tex_coord);
            glEnableVertexAttribArray(loc_tex_opacity);
            glEnableVertexAttribArray(loc_normal);
            glEnableVertexAttribArray(loc_ambient);
            glEnableVertexAttribArray(loc_diffuse);
            glEnableVertexAttribArray(loc_specular);
            glEnableVertexAttribArray(loc_shininess);

            glBindBuffer(GL_ARRAY_BUFFER, current.geometryVBO);
            glBindTexture(GL_TEXTURE_2D, current.textureVBO);
            //set pointers into the vbo for each of the attributes(position and color)
            glVertexAttribPointer( loc_position,//location of attribute
                                   3,//number of elements
                                   GL_FLOAT,//type
                                   GL_FALSE,//normalized?
                                   sizeof(Vertex),//stride
                                   0);//offset

            glVertexAttribPointer( loc_color,
                                   3,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,color));

            glVertexAttribPointer( loc_tex_coord,
                                   2,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,tex_coord));

            glVertexAttribPointer( loc_tex_opacity,
                                   1,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,tex_opacity));

            glVertexAttribPointer( loc_normal,
                                   3,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,normal));

            glVertexAttribPointer( loc_ambient,
                                   3,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,ambient));

            glVertexAttribPointer( loc_diffuse,
                                   3,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,diffuse));

            glVertexAttribPointer( loc_specular,
                                   3,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,specular));

            glVertexAttribPointer( loc_shininess,
                                   1,
                                   GL_FLOAT,
                                   GL_FALSE,
                                   sizeof(Vertex),
                                   (void*)offsetof(Vertex,shininess));

            glDrawArrays(current.drawMode, 0, current.geometry.size());

            for (auto &child : current.children) {

                renderSceneGraph(child, modelToWorld);
            }
        };

    for (auto &model : geometryRoot)
        renderSceneGraph(model, glm::mat4(1.0f));

    //clean up
    glDisableVertexAttribArray(loc_position);
    glDisableVertexAttribArray(loc_color);
    glDisableVertexAttribArray(loc_tex_coord);
    glDisableVertexAttribArray(loc_tex_opacity);
    glDisableVertexAttribArray(loc_normal);
    glDisableVertexAttribArray(loc_ambient);
    glDisableVertexAttribArray(loc_diffuse);
    glDisableVertexAttribArray(loc_specular);
    glDisableVertexAttribArray(loc_shininess);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);
    glutPrint(-0.9f, -0.9f, GLUT_BITMAP_TIMES_ROMAN_24, std::to_string(score[1]).c_str(),
            1.0f, 1.0f, 1.0f, 0.5f);
    glutPrint(0.9f, -0.9f, GLUT_BITMAP_TIMES_ROMAN_24,
            std::to_string(score[0]).c_str(),
            1.0f, 1.0f, 1.0f, 0.5f);

    //swap the buffers
    glutSwapBuffers();
}

void updateMatrices() {
    auto puckPosition =
    Models::puck->modelMatrix =
    		glm::translate(glm::mat4(1.0),
    				glm::vec3(phys::puck->GetPosition().x,
    						puckY,
    						phys::puck->GetPosition().y)) *
    		glm::rotate(glm::mat4(1.0),
    		        float(atanf(puckYVel)/M_PI*180.0f),
    		        glm::normalize(
    		                glm::cross(
    		                        glm::vec3(
    		                                phys::puck->GetLinearVelocity().x,
    		                                puckYVel,
    		                                phys::puck->GetLinearVelocity().y),
    		                        glm::vec3(0.0,1.0,0.0))));
    Models::paddle1->modelMatrix =
    		glm::translate(glm::mat4(1.0),
    				glm::vec3(phys::paddle1->GetPosition().x,
    						paddleY,
    						phys::paddle1->GetPosition().y));
    Models::paddle2->modelMatrix =
    		glm::translate(glm::mat4(1.0),
    				glm::vec3(phys::paddle2->GetPosition().x,
    						paddleY,
    						phys::paddle2->GetPosition().y));
}

void updateControls(GLfloat dt) {
    static int previousMouseX = 0, previousMouseY = 0;

    b2Vec2 paddleVelocities[2] = {b2Vec2_zero, b2Vec2_zero};
    if (specialKeys[GLUT_KEY_LEFT])
        paddleVelocities[0] += keyboardSensitivity*b2Vec2(1, 0);
    if (specialKeys[GLUT_KEY_RIGHT])
        paddleVelocities[0] += keyboardSensitivity*b2Vec2(-1, 0);
    if (specialKeys[GLUT_KEY_UP])
        paddleVelocities[0] += keyboardSensitivity*b2Vec2(0, 1);
    if (specialKeys[GLUT_KEY_DOWN])
        paddleVelocities[0] += keyboardSensitivity*b2Vec2(0, -1);

    if (keys['a'])
        paddleVelocities[1] += keyboardSensitivity*b2Vec2(1, 0);
    if (keys['d'])
        paddleVelocities[1] += keyboardSensitivity*b2Vec2(-1, 0);
    if (keys['w'])
        paddleVelocities[1] += keyboardSensitivity*b2Vec2(0, 1);
    if (keys['s'])
        paddleVelocities[1] += keyboardSensitivity*b2Vec2(0, -1);

    if (leftMouseButton) {
        previousMouseX = mouseX;
        previousMouseY = mouseY;
        captureMouse = !captureMouse;
        leftMouseButton = false;
    }
    b2Vec2 mouseMove = b2Vec2_zero;
    if (captureMouse) {
        paddleVelocities[1] +=
                mouseSensitivity/dt*b2Vec2(mouseX-previousMouseX,
                                           mouseY-previousMouseY);
        previousMouseX = mouseX;
        previousMouseY = mouseY;
    }

    phys::paddle1->SetLinearVelocity(paddleVelocities[0]);
    phys::paddle2->SetLinearVelocity(paddleVelocities[1]);

    if (keys['i'])
    	phi += dt*2.0;
    if (keys['k'])
    	phi -= dt*2.0;
    if (keys['j'])
    	theta += dt*2.0;
    if (keys['l'])
    	theta -= dt*2.0;
}

void checkForGoal() {
    if (!matchOver) {
        if (phys::puck->GetPosition().x < tableLeft) {
            score[1]++;
            matchOver = true;
            printf("%d/%d\n", score[1], score[0]);
        }
        if (phys::puck->GetPosition().x > tableRight) {
            score[0]++;
            matchOver = true;
            printf("%d/%d\n", score[1], score[0]);
        }
    } else {
        if (puckY < -10.0) {
            resetPuck();
        }
    }
}

void updateVerticalPhysics(float dt) {
    if (matchOver) {
        puckY += dt*puckYVel/2;
        puckYVel -= 9.8*dt;
        puckY += dt*puckYVel/2;
    }
}

void update() {
    float dt = getDT();// if you have anything moving, use dt.

    updateMatrices();
    updateControls(dt);
    updateVPMatrices();
    checkForGoal();

    // Call functions that update based on what's going on
    updateVerticalPhysics(dt);
    phys::world.Step(dt, 8, 3);

    glutPostRedisplay();
}


void reshape(int n_w, int n_h)
{
    w = n_w;
    h = n_h;
    //Change the viewport to be correct
    glViewport( 0, 0, w, h);
    //Update the projection matrix as well
    //See the init function for an explaination
    mats::projection = glm::perspective(45.0f, float(w)/float(h), 0.01f, 100.0f);
    mats::toWorld = glm::inverse(mats::projection * mats::view);
}

void keyboard(unsigned char key, int x_pos, int y_pos)
{
    keys[key] = true;
}

void keyboardUp(unsigned char key, int x_pos, int y_pos)
{
    keys[key] = false;
}

void specialKey(int key, int x, int y)
{
    specialKeys[key] = true;
}

void specialKeyUp(int key, int x, int y)
{
    specialKeys[key] = false;
}

void passiveMotion(int x, int y)
{
    int centerX = glutGet(GLUT_WINDOW_WIDTH) / 2;
    int centerY = glutGet(GLUT_WINDOW_HEIGHT) / 2;
    static bool mouseOn = false;
    static bool checkingForMotion = false;

    if (mouseOn)
        if (captureMouse) {
            mouseX += centerX-x;
            mouseY += centerY-y;

            if (checkingForMotion)
            {
                glutWarpPointer(centerX, centerY);
            }

            checkingForMotion = !checkingForMotion;
        } else {
            glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
            mouseOn = false;
        }
    else
        if (captureMouse) {
            glutSetCursor(GLUT_CURSOR_NONE);
            checkingForMotion = false;
            glutWarpPointer(centerX, centerY);
            mouseOn = true;
            mouseX -= centerX-x;
            mouseY -= centerY-y;
        }
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN)
            leftMouseButton = true;
        else
            leftMouseButton = false;
    } else if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_DOWN)
            rightMouseButton = true;
        else
            rightMouseButton = false;
    }
}

//returns the time delta
float getDT()
{
    float ret;
    t2 = std::chrono::high_resolution_clock::now();
    ret = std::chrono::duration_cast< std::chrono::duration<float> >(t2-t1).count();
    t1 = std::chrono::high_resolution_clock::now();
    return ret;
}

Vertex makeVertex(Material m, glm::vec3 pos, glm::vec3 normal) {
    switch (m) {
    case BLUE_PLASTIC:
        return Vertex{
            {pos.x, pos.y, pos.z},
            BALL_COLOR,
            {0,0},
            0,
            {normal.x, normal.y, normal.z},
            {0.4, 0.4, 0.4},
            {0.7, 0.7, 0.7},
            {.7, .7, .7},
            {0, 0, 0},
            15
        };
        break;
    case TEXTURED_PLASTIC:
        return Vertex{
            {pos.x, pos.y, pos.z},
            {1.0, 1.0, 1.0},
            {0,0},
            1,
            {normal.x, normal.y, normal.z},
            {0.4, 0.4, 0.4},
            {0.7, 0.7, 0.7},
            {.7, .7, .7},
            {0, 0, 0},
            15
        };
        break;
    }
    return Vertex{};
}

void addTriangle(std::vector<Vertex> &geometry,
        Material m1, glm::vec3 pos1,
        Material m2, glm::vec3 pos2,
        Material m3, glm::vec3 pos3) {
    glm::vec3 normal = glm::normalize(glm::cross(pos2-pos1, pos3-pos2));

    geometry.push_back(makeVertex(m1, pos1, normal));
    geometry.push_back(makeVertex(m2, pos2, normal));
    geometry.push_back(makeVertex(m3, pos3, normal));
}

GLint getGLLocation(const std::string &name, bool isUniform) {
    GLint result;
    if (isUniform)
        result = glGetUniformLocation(program,
                            const_cast<const char*>(name.c_str()));
    else
        result = glGetAttribLocation(program,
                        const_cast<const char*>(name.c_str()));
    if(result == -1)
    {
        std::cerr << "[F] " << boost::to_upper_copy(name) << " NOT FOUND" <<
                std::endl;
    }
    return result;
}

Vertex getMaterial(const aiMaterial &mtl) {
    Vertex result;
    float c[4];

    GLenum fill_mode;
    int ret1, ret2;
    aiColor4D diffuse;
    aiColor4D specular;
    aiColor4D ambient;
    aiColor4D emission;
    float shininess, strength;
    int two_sided;
    int wireframe;
    unsigned int max;



    for (int i = 0; i < 3; i++)
      result.diffuse[i] = 0.8f;
    if(AI_SUCCESS == mtl.Get( AI_MATKEY_COLOR_DIFFUSE, diffuse))
      for (int i = 0; i < 3; i++)
        result.diffuse[i] = diffuse[i];


    for (int i = 0; i < 3; i++)
      result.specular[i] = 0.0f;
    if(AI_SUCCESS == mtl.Get(AI_MATKEY_COLOR_SPECULAR, specular))
      for (int i = 0; i < 3; i++)
        result.specular[i] = specular[i];

    for (int i = 0; i < 3; i++)
      result.ambient[i] = 0.2f;
    if(AI_SUCCESS == mtl.Get(AI_MATKEY_COLOR_AMBIENT, ambient))
      for (int i = 0; i < 3; i++)
        result.ambient[i] = ambient[i];

    for (int i = 0; i < 3; i++)
      result.emission[i] = 0.0f;
    if(AI_SUCCESS == mtl.Get(AI_MATKEY_COLOR_EMISSIVE, emission))
      for (int i = 0; i < 3; i++)
        result.emission[i] = emission[i];

    ret1 = mtl.Get( AI_MATKEY_SHININESS, shininess );
    if(ret1 == AI_SUCCESS) {
      max = 1;
      ret2 = mtl.Get(AI_MATKEY_SHININESS_STRENGTH, strength);
      if(ret2 == AI_SUCCESS)
        result.shininess = shininess * strength;
      else
        result.shininess = shininess;
    } else {
      result.shininess = 0;
    }

    return result;
}

gil::rgba8_image_t getMaterialTexture(const aiMaterial &mtl) {
    aiString path;

    if (mtl.GetTextureCount(aiTextureType_DIFFUSE) > 0) {
        int w, h;
        mtl.GetTexture(aiTextureType_DIFFUSE,
                       0,
                       &path);

        //std::cout << path.C_Str() << std::endl;

        stbi_uc *rawImage = stbi_load(path.C_Str(), &w, &h, nullptr, 4);

        gil::rgba8_view_t converter =
            gil::interleaved_view(w, h, (gil::rgba8_pixel_t*) rawImage, w*4);
        gil::rgba8_image_t result(converter.dimensions());
        copy_pixels(converter, gil::view(result));

        stbi_image_free(rawImage);

        return result;
    } else {
        gil::rgba8_image_t result(1,1);
        gil::view(result)(0,0) = gil::rgba8_pixel_t(0,0,0,0);
        return result;
    }
}

Model convertAssimpScene (const aiScene &sc) {
    Model model;
    std::vector<aiNode*> nodeStack{sc.mRootNode};
    std::vector<aiMatrix4x4> matrixStack{sc.mRootNode->mTransformation};
    std::vector<Model*> modelStack{&model};
    aiMatrix4x4 matrix;

    while (!matrixStack.empty()) {
        aiNode* node = nodeStack.back();
        nodeStack.pop_back();

        matrix = matrixStack.back() * node->mTransformation;
        matrixStack.pop_back();

        Model *nodeModel = modelStack.back();
        modelStack.pop_back();

        for (int n = 0; n < node->mNumMeshes; n++) {
            Model *meshModel = nullptr;

            if (n == 0) {
                meshModel = nodeModel;
            } else {
                nodeModel->children.emplace_back();
                meshModel = &nodeModel->children.back();
            }

            aiMesh* mesh = sc.mMeshes[node->mMeshes[n]];

            Vertex prototype = getMaterial(*sc.mMaterials[mesh->mMaterialIndex]);
            //Vertex prototype = makeVertex(BALL, glm::vec3(), glm::vec3());

            meshModel->texture =
                    getMaterialTexture(*sc.mMaterials[mesh->mMaterialIndex]);

            for (int t = 0; t < mesh->mNumFaces; t++) {
                const struct aiFace* face = &mesh->mFaces[t];

                switch(face->mNumIndices) {
                    case 1: meshModel->drawMode = GL_POINTS; break;
                    case 2: meshModel->drawMode = GL_LINES; break;
                    case 3: meshModel->drawMode = GL_TRIANGLES; break;
                    default: meshModel->drawMode = GL_POLYGON; break;
                }


                for(int i = 0; i < face->mNumIndices; i++) {
                    Vertex v = prototype;
                    int index = face->mIndices[i];
                    if(mesh->mColors[0] != nullptr) {
                        v.color[0] = mesh->mColors[0][index].r;
                        v.color[1] = mesh->mColors[0][index].g;
                        v.color[2] = mesh->mColors[0][index].b;
                    } else {
                    	v.color[0] = 0.8;
                    	v.color[1] = 0.8;
                    	v.color[2] = 0.8;
                    }
                    if(mesh->mNormals != nullptr) {
                        v.normal[0] = mesh->mNormals[index].x;
                        v.normal[1] = mesh->mNormals[index].y;
                        v.normal[2] = mesh->mNormals[index].z;
                    }
                    v.position[0] = mesh->mVertices[index].x;
                    v.position[1] = mesh->mVertices[index].y;
                    v.position[2] = mesh->mVertices[index].z;

                    if (mesh->HasTextureCoords(0)) {
                        v.tex_coord[0] = mesh->mTextureCoords[0][index].x;
                        v.tex_coord[1] = 1-mesh->mTextureCoords[0][index].y;
                        v.tex_opacity = 1;
                    } else {
                    	v.tex_coord[0] = 0;
                    	v.tex_coord[1] = 0;
                    	v.tex_opacity = 0;
                    }

                    meshModel->geometry.push_back(v);
                }
            }

        }

        // draw all children
        for (int n = 0; n < node->mNumChildren; ++n) {
            nodeStack.push_back(node->mChildren[n]);
            matrixStack.push_back(matrix);

            nodeModel->children.emplace_back();
            modelStack.push_back(&nodeModel->children.back());
        }
    }

    return model;
}

void forAllModels(std::function<void (Model &)> f) {
    std::list<Model*> rootRef;
    for (auto &model : geometryRoot) {
        rootRef.push_back(&model);
    }
    forChildModels(rootRef, f);
}

void forChildModels(std::list<Model*> root, std::function<void (Model &)> f) {
    std::stack<Model*> modelStack;
    for (auto model : root) {
        modelStack.push(model);
    }

    while (!modelStack.empty()) {
        Model* m = modelStack.top();
        modelStack.pop();

		f(*m);

		for (auto &child : m->children) {
			modelStack.push(&child);
		}
    }
}

Model::Model() {
    texture = gil::rgba8_image_t(1,1);
    gil::view(texture)(0,0) = gil::rgba8_pixel_t(0,0,0,0);

    modelMatrix = glm::mat4(1);
}

void Model::setMaterial(Material m) {
    for (auto &v : geometry) {
        auto oldV = v;
        v = makeVertex(m,
                       glm::vec3{v.position[0], v.position[1], v.position[2]},
                       glm::vec3{v.normal[0], v.normal[1], v.normal[2]});
        v.tex_coord[0] = oldV.tex_coord[0];
        v.tex_coord[1] = oldV.tex_coord[1];
    }
}
