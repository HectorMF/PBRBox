#pragma once

#include <string>
#include <vector>
#include <gli/gli.hpp>
#include <fstream>

class Volume
{
	int size;
public:
	Volume(std::string file)
	{
		std::ifstream ifs(file, std::ios::binary | std::ios::ate);
		std::ifstream::pos_type pos = ifs.tellg();

		std::vector<char> result(pos);

		ifs.seekg(0, std::ios::beg);
		ifs.read(&result[0], pos);

		size = 768;
	}
};