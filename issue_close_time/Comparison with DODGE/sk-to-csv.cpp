#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <limits>

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " FILENAME METRIC\n";
		return -1;
	}

	std::ifstream fin(argv[1]);
	std::ofstream fout(argv[2]);

	fout << "rank,model,data,median\n";

	std::string line;
	bool looking = false;
	while (std::getline(fin, line))
	{
		if (line.length() == 0)
		{
			looking = false;
			continue;
		}
		if (line[0] == '=') continue;
		if (line.find(argv[2]) != std::string::npos)
		{
			// We found the metric
			looking = true;
			continue;
		}
		
		if (looking) {
		// We're in the Scott-Knott results, parse.
		std::istringstream iss(line);
		int rank;
		iss >> rank;

		char name[8];
		iss.ignore(1);
		iss.get(name, std::numeric_limits<int>::max(), '-');

		char data[30];
		iss.ignore(1);
		iss.get(data, std::numeric_limits<int>::max(), '-');
		iss.ignore(1);
		data[std::strlen(data) - 1] = '-';
		iss.get(data + std::strlen(data), std::numeric_limits<int>::max(), '-');
		iss.ignore(1);
		char treatment[15];
		iss >> treatment;

		for (int i = 0; i < 3; i++)
			iss.ignore(std::numeric_limits<int>::max(), ',');

		double median;
		iss >> median;

		fout << rank << "," << name << "-" << treatment << "," << data << "," << median << "\n";
		}
	}

	fin.close();
	fout.close();

	return 0;
}
