#ifndef READINIT_H
#define READINIT_H

#include <vector>

using namespace std;


struct rdat{
	int numParticles;
	bool dataC;
	bool gfxEn;
	int timestep;
	vector<group> groupvec; 
};

void initialize(group& g);

void initialize_params(GroupParams& gp);

void print_group(const group& g, const GroupParams& gp);

vector<string> split (const string& line, const string& delimiters);

void remove_spaces(string& s);

void remove_comment(string& s);

string get_name(const string& s);

kernel get_kernel_type(string input);

Type get_integration_type(string input);

void set_key_value(group& g, GroupParams& gp, vector<string>& pair, SimParams* m_params, rdat* rtrn);

int get_sphere_partcount(int r);

void fill_array(const int& part_count, int* particle_group_number, vector<group>* groupvec);

int calc_num_particles(group& g);

rdat read_particlesystem_initfile(SimParams* m_params, GroupParams* gparams_array, int* particle_group_number);

#endif