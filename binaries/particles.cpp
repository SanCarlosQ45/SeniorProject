/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"
#include "particles_kernel.cuh"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64

using namespace std;

const uint width = 640, height = 480;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 2000;

enum { M_VIEW = 0, M_MOVE };

uint numParticles = 0;

vector<int> group_start_index;

int numIterations = 0; // run until exit
uint numOfSteps;

// simulation parameters
float timestep = 0.1f;

float damping = 1.0f;
float gravity = 0.0003f;
int iterations = 1;
int ballr = 10;

Type int_type = EULER;
kernel ker_type = GAUSSIAN;

float collideSpring = 0.5f;
float collideDamping = 0.02f;
float collideShear = 0.1f;
float collideAttraction = 0.0f;


ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

ParticleRenderer *renderer = 0;

float modelView[16];

ParamListGL *params;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
char        *g_refFile = NULL;

const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

vector<group> groupvec;
int* particle_group_number;
vector<GroupParams> group_params;

GroupParams* gparams_array;
GroupParams* d_gparams_array;
uint flag = 0;

bool dataC;
bool gfxEn;

/**
* void initialize(group&)
*
* Method that initializes the input group with default values. This is used in 
* case the init file is missing a parameter + value.
*
* @param g group containing the parameters to be changed
*/
void initialize(group& g){
	string name = "default";
	g.color = "green";
	g.type = "square";
	g.rad = 5;
	g.height = 5;
	g.length = 5;
	g.width = 5;
	g.coord[0] = 0;
	g.coord[1] = 0;
	g.coord[2] = 0;
	g.vel[0] = 0;
	g.vel[1] = 0;
	g.vel[2] = 0;
	g.particle_count = 125;
}

/**
* void initialize(GroupParams&)
*
* Method that initializes the input GroupParam with default values. This is used in 
* case the init file is missing a parameter + value.
*
* @param gp GroupParam to be initialized
*/
void initialize_params(GroupParams& gp){
	gp.spring = 0.5f;
	gp.damping = 0.02f;
	gp.shear = 0.1f;
	gp.attraction = 0.0f;
}

/**
* void print_group(const group&, const GroupParam&)
*
* Method that prints the data of the input group as well as the 
* corresponding GroupParams
*
* @param g group to be printed
* @param gp GroupParam to be printed
*/
void print_group(const group& g, const GroupParams& gp){
	cout << "Group Name: " << g.name << endl;
	cout << "Color: " << g.color << endl;
	cout << "Type: " << g.type << endl;
	cout << "Radius: " << g.rad << endl;
	cout << "Height: " << g.height << endl;
	cout << "Length: " << g.length << endl;
	cout << "Width: " << g.width << endl;
	cout << "X coordinate: " << g.coord[0] << endl;
	cout << "Y coordinate: " << g.coord[1] << endl;
	cout << "Z coordinate: " << g.coord[2] << endl;
	cout << "X vel: " << g.vel[0] << endl;
	cout << "Y vel: " << g.vel[1] << endl;
	cout << "Z vel: " << g.vel[2] << endl;
	cout << "Particle Count: " << g.particle_count << endl;

	cout << "Damping: " << gp.damping << endl;
	cout << "Spring: " << gp.spring << endl;
	cout << "Shear: " << gp.shear << endl;
	cout << "Attraction: " << gp.attraction << endl;
}

/**
* vector<string> split (const string&, const string&)
*
* Function that takes a line and a delimintor(point at which to split) and
* returns a pair of strings.
*
* @param line string that was read from the init file
* @param deliminator a char that is looked for in the line which is where it will split
*/
vector<string> split (const string& line, const string& delimiters) {
   vector<string> words;
   size_t end = 0;
   if(line.find_first_of(delimiters, 0) == string::npos){
	   words.push_back(line);
   }
   for (;;) {
      size_t start = line.find_first_not_of (delimiters, end);
      if (start == string::npos) break;
      end = line.find_first_of (delimiters, start);
      words.push_back (line.substr (start, end - start));
   }
   return words;
}

/**
* void remove_spaces(string&)
*
* Method that removes spaces from a line. This will remove spaces at the beginning, middle or end
*
* @param s line containing spaces to be removed
*/
void remove_spaces(string& s){
	string result = "";
	for(int i=0; i<s.length(); ++i){
		if(s.at(i) != ' ' && s.at(i) != 'TAB')
			result.append(1,s.at(i));	
	}
	s = result;
}

/**
* void remove_comment(string&)
*
* Method that removes comments at the end of a line, it removes all
* chars after a '#'
*
* @param s line that may or may not contain a comment
*/
void remove_comment(string& s){
	vector<string> values = split(s, "#");
	if(values.size() != 1){
		s = values.at(0);
	}
}

string get_name(const string& s){
	if(s.at(1) == ']'){
		return "None";
	}
	else{
		return s.substr(1,s.length()-2);
	}
}

kernel get_kernel_type(string input){
	if(input == "Gaussian"){
		return GAUSSIAN;
	}
	return GAUSSIAN;
}
Type get_integration_type(string input){
	if(input == "Euler"){
		return EULER;
	}
	else if(input == "Verlet"){
		return VERLET;
	}
	return EULER;
}

/**
* void set_key_value(group&, GroupParams&, vector<string>&, SimParams)
*
* Function that takes the parameter/value pair and sets their corresponding value
*
* @param g group containing the parameters to be changed
* @param gp current group parameter if a [group] has been reached 
* @param pair string pair that contains the parameter at 0 and value to be set at 1
* @param m_params SimParams struct containing the global constants
*/

void set_key_value(group& g, GroupParams& gp, vector<string>& pair, SimParams* m_params){
	string key=pair.at(0);
	string value=pair.at(1);

		if(key == "Type"){
			g.type = value;
		}
		else if(key == "Color"){
			g.color = value;
		}
		else if(key == "Radius"){
			g.rad = atoi(value.c_str());
		}
		else if(key == "Height"){
			g.height = atoi(value.c_str());
		}
		else if(key == "Length"){
			g.length = atoi(value.c_str());
		}
		else if(key == "Width"){
			g.width = atoi(value.c_str());
		}
		else if(key == "Position"){
			vector<string> cor = split(value, ",");
			g.coord[0] = (float)atof(cor.at(0).c_str());
			g.coord[1] = (float)atof(cor.at(1).c_str());
			g.coord[2] = (float)atof(cor.at(2).c_str());
		}
		else if(key == "Velocity"){
			vector<string> vel = split(value, ",");
			g.vel[0] = (float)atof(vel.at(0).c_str());
			g.vel[1] = (float)atof(vel.at(1).c_str());
			g.vel[2] = (float)atof(vel.at(2).c_str());
		}

		else if(key == "Damping"){
			gp.damping = (float)atof(value.c_str());
		}
		else if(key == "Spring"){
			gp.spring = (float)atof(value.c_str());
		}
		else if(key == "Shear"){
			gp.shear = (float)atof(value.c_str());
		}
		else if(key == "Attraction"){
			gp.attraction = (float)atof(value.c_str());
		}
		else if(key == "DataCollection"){
			if(value == "True" || value == "true"){
				dataC = true;
			}
			else{
				dataC = false;
			}
		}
		else if(key == "Graphics"){
			if(value == "True" || value == "true"){
				gfxEn = true;
			}
			else{
				gfxEn = false;
			}
		}
		else if(key == "Timestep"){
			cout << "Timestep: " << (float)atof(value.c_str()) << endl;
			m_params->timestep = (float)atof(value.c_str());
			timestep = (float)atof(value.c_str());
		}
		else if(key == "GlobalDamping"){
			m_params->globalDamping = (float)atof(value.c_str());
		}
		else if(key == "Gravity"){
			float3 gv = make_float3(0.0f,-(float)atof(value.c_str()), 0.0f);
			m_params->gravity = gv;
		}
		else if(key == "IntegrationType"){
			m_params->integration_type = get_integration_type(value);
		}
		else if(key == "KernelType"){
			m_params->kernel_type = get_kernel_type(value);
		}
		else{
			cout << "Arguement " << key << " doesn't exist" << endl;
		}
}

/**
* int get_sphere_partcount(int)
*
* Calculates the particle count for a given sphere radius
*
* @param r radius value for sphere given in init file
*/
int get_sphere_partcount(int r){
    int index = 0;
	float particleRadius = 1.0f / 64.0f;
	float spacing = particleRadius*2.0f;

    for (int z=-r; z<=r; z++){
        for (int y=-r; y<=r; y++){
            for (int x=-r; x<=r; x++){

                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = particleRadius*0.01f;

                if (l <= particleRadius*2.0f*r)
                {
					++index;
                }
            }
        }
    }
	return index;
}

/**
* void fill_array(const int&)
*
* Method that fills the array containing the particles group value. This is necissary
* because the program needs this data when the particle system is initialized. 
*
* @param part_count total number of particles in system, calculated from the initfile
*/
void fill_array(const int& part_count){

	int group_index = 0;
	int cur_limit = groupvec.at(group_index).particle_count;

	for(int i=0; i < part_count; ++i){
		if(i == cur_limit){
			++group_index;
			cur_limit += groupvec.at(group_index).particle_count;
		}
		particle_group_number[i] = group_index;
	}
}

/**
* int calc_num_particles(group&)
*
* Function that calculates and returns the number of particles for the type it is
*
* @param g group containing the type, either a sphere, square or rectangle
* @return an int containing the number of particles the object contains
*/
int calc_num_particles(group& g){
	if(g.type == "Square"){
		g.particle_count = g.length*g.length*g.length;
		return g.particle_count;
	}
	else if(g.type == "Rectangle"){
		g.particle_count = g.length*g.width*g.height;
		
	}
	else if(g.type == "Sphere"){
		g.particle_count = get_sphere_partcount(g.rad);
	}

	return g.particle_count;
}

/**
* void read_particlesystem_initfile(SimParams*)
*
* Main method for reading the init file, modifies the SimParam struct to contain
* the new global constant values as well as modify the group param array and 
* vector. These values are then used by host and device code.
*
* @param SimParams global constant struct to be modified
*/
void read_particlesystem_initfile(SimParams* m_params){
	string line;
	ifstream ifile ("partinit.txt");
	int objnum = 0;
	bool gstart = false;
	gparams_array = new GroupParams[10];
	
	dataC = true;
	gfxEn = true;

	vector<string> pair;

	if(ifile.fail()){
		cerr << "partinit.txt doesnt exist, using default values!" << endl;
	}

	while(!ifile.eof()){
		getline(ifile,line);
		remove_spaces(line); //Remove any spaces

		if(line.size()){ //If not empty line			
			if(line.at(0) != '#'){ //Not a comment and line is not empty				 
				 remove_comment(line);
				 //Check for Section header, otherwise Check for keywords
				 if(line.at(0) == '['){
					//Create new object and set object number
					gstart = true;
					string n= get_name(line);
					group g;
					GroupParams gp;
					initialize(g);
					initialize_params(gp);
					g.name = n;
					groupvec.push_back(g);
					objnum = (int)groupvec.size()-1;

					gparams_array[objnum] = gp;

					group_params.push_back(gp);
									 
				 }
				 else{
					 if(gstart){
						set_key_value(groupvec.at(objnum), gparams_array[objnum], split(line, "="), m_params);
					 }
					 else{
						 group i;
						 GroupParams j;
						 set_key_value(i,j, split(line, "="), m_params);
					 }
				 }
			}
		}
	}

	numParticles = 0;

	for(int i=0; i < groupvec.size(); ++i){
		numParticles += calc_num_particles(groupvec.at(i));
		print_group(groupvec.at(i), gparams_array[i]);
		cout << endl;
	}
	
	particle_group_number = new int[numParticles];
	fill_array(numParticles);

	cout << "Paticle Count: " << numParticles << endl;
	cout << "Data Collection: " << dataC << endl;
	cout << "Graphics Enabled: " << gfxEn << endl;

}


// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL, SimParams* m_params)
{
    psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL, particle_group_number, groupvec, gparams_array, m_params);
	psystem->reset(ParticleSystem::CONFIG_TEST);
	psystem->init_sph_kernel(1, 1.0);

    if (bUseOpenGL)
    {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
}

// initialize OpenGL
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Particles");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined (_WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "CUDA Particles (%d particles): %3.1f fps", numParticles, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void display()
{
    sdkStartTimer(&timer);

    // update the simulation
    if (!bPause){
		if(dataC){
			psystem->dumpParticles(0, numParticles-1, timestep, numOfSteps);
		}

        psystem->update(timestep, d_gparams_array);

        if (renderer)
        {
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        }
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);

    // collider
    glPushMatrix();
    float3 p = psystem->getColliderPos();
    glTranslatef(p.x, p.y, p.z);
    glColor3f(1.0, 0.0, 0.0);
    glutSolidSphere(psystem->getColliderRadius(), 20, 10);
    glPopMatrix();

    if (renderer && displayEnabled)
    {
        renderer->display(displayMode);
    }

    if (displaySliders)
    {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    sdkStopTimer(&timer);

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void addSphere()
{
    // inject a sphere of particles
    float pr = psystem->getParticleRadius();
    float tr = pr+(pr*2.0f)*ballr;
    float pos[4], vel[4];
    pos[0] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[1] = 1.0f - tr;
    pos[2] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[3] = 0.0f;
    vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
    psystem->addSphere(0, numParticles, pos, vel, ballr, pr*2.0f, 0);
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer)
    {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    if (displaySliders)
    {
        if (params->Mouse(x, y, button, state))
        {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}

// transfrom vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
    r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
    r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
    r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
    r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (displaySliders)
    {
        if (params->Motion(x, y))
        {
            ox = x;
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

    switch (mode)
    {
        case M_VIEW:
            if (buttonState == 3)
            {
                // left+middle = zoom
                camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
            }
            else if (buttonState & 2)
            {
                // middle = translate
                camera_trans[0] += dx / 100.0f;
                camera_trans[1] -= dy / 100.0f;
            }
            else if (buttonState & 1)
            {
                // left = rotate
                camera_rot[0] += dy / 5.0f;
                camera_rot[1] += dx / 5.0f;
            }

            break;

        case M_MOVE:
            {
                float translateSpeed = 0.003f;
                float3 p = psystem->getColliderPos();

                if (buttonState==1)
                {
                    float v[3], r[3];
                    v[0] = dx*translateSpeed;
                    v[1] = -dy*translateSpeed;
                    v[2] = 0.0f;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }
                else if (buttonState==2)
                {
                    float v[3], r[3];
                    v[0] = 0.0f;
                    v[1] = 0.0f;
                    v[2] = dy*translateSpeed;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }

                psystem->setColliderPos(p);
            }
            break;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case ' ':
            bPause = !bPause;
            break;

        case 13:
            psystem->update(timestep, d_gparams_array);

            if (renderer)
            {
                renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
            }

            break;

        case '\033':
        case 'q':
            exit(EXIT_SUCCESS);
            break;

        case 'v':
            mode = M_VIEW;
            break;

        case 'm':
            mode = M_MOVE;
            break;

        case 'p':
            displayMode = (ParticleRenderer::DisplayMode)
                          ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
            break;

        case 'd':
            psystem->dumpGrid();
            break;

        case 'u':
            psystem->dumpParticles(0, numParticles-1, timestep, numOfSteps);
            break;

        case 'r':
            displayEnabled = !displayEnabled;
            break;

        case '1':
            psystem->reset(ParticleSystem::CONFIG_GRID);
            break;

        case '2':
            psystem->reset(ParticleSystem::CONFIG_RANDOM);
            break;

        case '3':
            addSphere();
            break;

        case '4':
            {
                // shoot ball from camera
                float pr = psystem->getParticleRadius();
                float vel[4], velw[4], pos[4], posw[4];
                vel[0] = 0.0f;
                vel[1] = 0.0f;
                vel[2] = -0.05f;
                vel[3] = 0.0f;
                ixform(vel, velw, modelView);

                pos[0] = 0.0f;
                pos[1] = 0.0f;
                pos[2] = -2.5f;
                pos[3] = 1.0;
                ixformPoint(pos, posw, modelView);
                posw[3] = 0.0f;

                psystem->addSphere(0, numParticles, posw, velw, ballr, pr*2.0f, 0);
            }
            break;

        case 'w':
            wireframe = !wireframe;
            break;

        case 'h':
            displaySliders = !displaySliders;
            break;
    }

    demoMode = false;
    idleCounter = 0;
    glutPostRedisplay();
}

void special(int k, int x, int y)
{
    if (displaySliders)
    {
        params->Special(k, x, y);
    }

    demoMode = false;
    idleCounter = 0;
}

void idle(void)
{
    glutPostRedisplay();
}

void initParams()
{
    if (g_refFile)
    {
        timestep = 0.0f;
        damping = 0.0f;
        gravity = 0.0f;
        ballr = 1;
        collideSpring = 0.0f;
        collideDamping = 0.0f;
        collideShear = 0.0f;
        collideAttraction = 0.0f;

    }
    else
    {

        // create a new parameter list
        params = new ParamListGL("misc");
        params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.01f, &timestep));
        params->AddParam(new Param<float>("damping"  , damping , 0.0f, 10.0f, 0.1f, &damping)); //Original 1.0f, 0.01f
        params->AddParam(new Param<float>("gravity"  , gravity , 0.0f, 0.01f, 0.001f, &gravity)); //Original 0.001f, 0.0001f
        params->AddParam(new Param<int> ("ball radius", ballr , 1, 20, 1, &ballr));

        params->AddParam(new Param<float>("collide spring" , collideSpring , 0.0f, 10.0f, 0.01f, &collideSpring)); //Original 1.0f, 0.001f
        params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 1.0f, 0.01f, &collideDamping)); //original 0.1f, 0.001f
        params->AddParam(new Param<float>("collide shear"  , collideShear  , 0.0f, 1.0f, 0.01f, &collideShear)); //original 0.1f, 0.001f
        params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 1.0f, 0.01f, &collideAttraction)); //original 0.1f, 0.001f
    }
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
    glutAddMenuEntry("Add sphere [3]", '3');
    glutAddMenuEntry("View mode [v]", 'v');
    glutAddMenuEntry("Move cursor mode [m]", 'm');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    printf("%s Starting...\n\n", sSDKsample);
	uint3 gridSize;
	SimParams m_params;

//	uint gridDim = GRID_SIZE;
    numIterations = 0;
	numOfSteps = 0;

    gridSize.x = gridSize.y = gridSize.z = GRID_SIZE;
    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
    printf("particles: %d\n", numParticles);

	read_particlesystem_initfile(&m_params);

    initGL(&argc, argv);
    cudaGLInit(argc, argv);

    initParticleSystem(numParticles, gridSize, g_refFile==NULL, &m_params);
    initParams();

	initMenus();

	//if(flag == 0){
			cudaMalloc((void**)&d_gparams_array, 10 * sizeof(GroupParams));
			checkCudaErrors(cudaMemcpy(d_gparams_array, gparams_array, 10 * sizeof(GroupParams), cudaMemcpyHostToDevice));
			flag = 1;
	//}

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutSpecialFunc(special);
    glutIdleFunc(idle);

    atexit(cleanup);

    glutMainLoop();

    if (psystem)
    {
        delete psystem;
    }

    cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

