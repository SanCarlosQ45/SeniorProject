#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "particleSystem.h"
#include "readinit.h"

using namespace std;

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

void initialize_params(GroupParams& gp){
	gp.spring = 0.5f;
	gp.damping = 0.02f;
	gp.shear = 0.1f;
	gp.attraction = 0.0f;
}

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

void remove_spaces(string& s){
	string result = "";
	for(int i=0; i<s.length(); ++i){
		if(s.at(i) != ' ' && s.at(i) != 'TAB')
			result.append(1,s.at(i));	
	}
	s = result;
}

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

void set_key_value(group& g, GroupParams& gp, vector<string>& pair, SimParams* m_params, rdat* rtrn){
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
				rtrn->dataC = true;
			}
			else{
				rtrn->dataC = false;
			}
		}
		else if(key == "Graphics"){
			if(value == "True" || value == "true"){
				rtrn->gfxEn = true;
			}
			else{
				rtrn->gfxEn = false;
			}
		}
		else if(key == "Timestep"){
			cout << "Timestep: " << (float)atof(value.c_str()) << endl;
			m_params->timestep = (float)atof(value.c_str());
			rtrn->timestep = (float)atof(value.c_str());
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

void fill_array(const int& part_count, int* particle_group_number, vector<group>* groupvec){

	int group_index = 0;
	int cur_limit = groupvec->at(group_index).particle_count;

	particle_group_number = new int[part_count];

	for(int i=0; i < part_count; ++i){
		if(i == cur_limit){
			++group_index;
			cur_limit += groupvec->at(group_index).particle_count;
		}
		particle_group_number[i] = group_index;
	}
}

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



rdat read_particlesystem_initfile(SimParams* m_params, GroupParams* gparams_array, int* particle_group_number){
	string line;
	ifstream ifile ("partinit.txt");
	int objnum = 0;
	bool gstart = false;

	gparams_array = new GroupParams[10];
	vector<group> groupvec;
	vector<GroupParams> group_params;

	rdat rtrn;

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
				 /*Check for Section header, otherwise Check for keywords*/
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
						set_key_value(groupvec.at(objnum), gparams_array[objnum], split(line, "="), m_params, &rtrn);
					 }
					 else{
						 group i;
						 GroupParams j;
						 set_key_value(i,j, split(line, "="), m_params, &rtrn);
					 }
				 }
			}
		}
	}

	rtrn.numParticles = 0;

	for(int i=0; i < groupvec.size(); ++i){
		rtrn.numParticles += calc_num_particles(groupvec.at(i));
		print_group(groupvec.at(i), gparams_array[i]);
		cout << endl;
	}
	
	particle_group_number = new int[rtrn.numParticles];
	fill_array(rtrn.numParticles, particle_group_number, &groupvec);
	rtrn.groupvec = groupvec;

	cout << "Paticle Count: " << rtrn.numParticles << endl;
	cout << "Data Collection: " << rtrn.dataC << endl;
	cout << "Graphics Enabled: " << rtrn.gfxEn << endl;

	return rtrn;
}
