#include <string>
#include <fstream>
#include <streambuf>

#include "read_file.h"

// Thanks to Tyler McHenry on Stack Overflow
std::string read_file(const std::string &fName) {
	std::ifstream t(fName);
	std::string str;

	t.seekg(0, std::ios::end);   
	str.reserve(t.tellg());
	t.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(t)),
			            std::istreambuf_iterator<char>());

	return str;
}

