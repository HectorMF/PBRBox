#pragma once
#include <string>
#include <vector>
#include "Geometry.h"
#include <memory>
#include <map>
#include <algorithm> 

std::vector<std::string> stressVariables;
std::vector<glm::vec2> stressVariableMinMax;

glm::vec3 getHeatMapColor(const float & pressure)
{
	const int NUM_COLORS = 5;
	float value = pressure;
	float color[NUM_COLORS][3] = { { 0, 0, 1 },{ 0, 1, 1 },{ 0, 1, 0 },{ 1, 1, 0 },{ 1, 0, 0 } };
	// A static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each.

	int idx1;        // |-- Our desired color will be between these two indexes in "color".
	int idx2;        // |
	float fractBetween = 0;  // Fraction between "idx1" and "idx2" where our value is.

	if (value <= 0) { idx1 = idx2 = 0; }    // accounts for an input <=0
	else if (value >= 1) { idx1 = idx2 = NUM_COLORS - 1; }    // accounts for an input >=0
	else
	{
		value = value * (NUM_COLORS - 1);        // Will multiply value by 3.
		idx1 = floor(value);                  // Our desired color will be after this index.
		idx2 = idx1 + 1;                        // ... and before this index (inclusive).
		fractBetween = value - float(idx1);    // Distance between the two indexes (0-1).
	}

	glm::vec3 hmColor;
	hmColor[0] = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
	hmColor[1] = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
	hmColor[2] = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
	return hmColor;
}

void removeExtraCharacters(std::string& string)
{
	string.erase(remove(string.begin(), string.end(), ' '), string.end());
	string.erase(remove(string.begin(), string.end(), '"'), string.end());
}

void removeSubstring(std::string& string, const std::string substring)
{
	int foundpos = string.find(substring);
	if (foundpos != std::string::npos)
		string.erase(string.begin() + foundpos, string.begin() + foundpos + substring.length());
}

void parseLine(const std::string &line, std::vector<std::string> *elements, const std::string &delimiter)
{
	std::string input = line;

	while (input != "")
	{
		if (input[0] == ' ')
		{
			input.erase(input.begin());
			continue;
		}
		int offset = input.find_first_of(delimiter);
		int quote = input.find_first_of('"');

		std::string ret = "";

		if (offset != std::string::npos)
		{
			//check if we have a quote for the next set...
			if (quote == std::string::npos && offset < quote)
			{
				ret = input.substr(0, offset);

				//erase from our string, keep going
				input.erase(0, offset + 1);
			}
			//we have a quoted type, lets scan it.
			else if (quote != std::string::npos && offset > quote)
			{
				int quote2 = input.find_first_of('"', quote + 1);

				//read off quote
				ret = input.substr(0, quote2 + 1);

				//remove comma
				input.erase(0, quote2 + 2);
			}
			else
			{
				// handle normally...
				ret = input.substr(0, offset);

				//erase from our string, keep going
				input.erase(0, offset + 1);
			}
		}
		else
		{
			//no more strings, this is it.
			ret = input;
			input.clear();
		}

		elements->push_back(ret);
	}
}

void readLines(const std::string &file, std::vector <std::string> *list)
{
	//if (!gb::FileTools::fileExists(file))
	//	return;

	//clear the list?
	list->clear();

	//open file
	FILE *f = fopen(file.c_str(), "r");

	//go line by line
	while (!feof(f))
	{
		char tmp[2048];
		memset(tmp, 0, 2048);

		fgets(tmp, 2048, f);

		std::string str(tmp);

		//do we have a new line?
		if (str.find('\n') != std::string::npos)
			str.erase(str.find('\n'));

		//add without \n
		list->push_back(str);
	}

	fclose(f);
}

std::vector<Geometry> loadModel(const std::string &file)
{
	//minPoint.set(FLT_MAX, FLT_MAX, FLT_MAX);
	//maxPoint.set(FLT_MIN, FLT_MIN, FLT_MIN);
	std::vector<Geometry> m_meshes;
	m_meshes.clear();
	std::vector <std::string> lines;
	readLines(file, &lines);

	//insert file parsing here:
	std::string title = lines[0];
	std::vector<std::string> variables;
	std::map<std::string, std::string> auxData;

	int totalVertices = 0;
	int totalFaces = 0;

	int lineOffset = 1;
	//parse the variables from the file
	while (lineOffset < lines.size())
	{
		if (lines[lineOffset].find("DATASETAUXDATA") != std::string::npos ||
			lines[lineOffset].find("ZONE") != std::string::npos)
		{
			break;
		}

		removeExtraCharacters(lines[lineOffset]);

		if (lineOffset == 1)
		{
			removeSubstring(lines[lineOffset], "VARIABLES=");
		}

		variables.push_back(lines[lineOffset]);

		lineOffset++;
	}

	while (lineOffset < lines.size())
	{
		if (lines[lineOffset].find("DATASETAUXDATA") == std::string::npos)
			break;

		removeSubstring(lines[lineOffset], "DATASETAUXDATA");
		removeExtraCharacters(lines[lineOffset]);

		int pos = lines[lineOffset].find("=");
		auxData[lines[lineOffset].substr(0, pos)] = lines[lineOffset].substr(pos + 1);;

		lineOffset++;
	}

	//load every node
	while (lineOffset < lines.size())
	{
		while (lineOffset < lines.size())
		{
			if (lines[lineOffset].find("ZONE") != std::string::npos)
				break;
			lineOffset++;
		}
		if (lineOffset >= lines.size()) break;

		std::vector<int> faceVariables;
		std::map <std::string, std::vector<float>> variableMap;

		Geometry mesh;

		removeSubstring(lines[lineOffset], "ZONE T=");
		removeExtraCharacters(lines[lineOffset]);
		//mesh->m_nodeName = lines[lineOffset];
		lineOffset += 2;
		//parse fileInfo to get positions
		std::vector<std::string> list;
		parseLine(lines[lineOffset], &list, ",");
		int iNodes = 0;
		sscanf(list[0].c_str(), "Nodes=%i", &iNodes);
		int iElements = 0;
		sscanf(list[1].c_str(), "Elements=%i", &iElements);
		char tmp[256];
		std::string zoneType;
		sscanf(list[2].c_str(), "ZONETYPE=%s", &tmp);
		zoneType = tmp;
		//clear string
		memset(&tmp, 0, 256);

		lineOffset++;

		//fetch our packing method:
		removeExtraCharacters(lines[lineOffset]);
		removeSubstring(lines[lineOffset], "DATAPACKING=");
		std::string dataPacking = lines[lineOffset];

		lineOffset++;
		std::string varLocation;
		sscanf(lines[lineOffset].c_str(), "VARLOCATION=%s", &tmp);
		varLocation = tmp;

		//parse range
		{
			while (stressVariableMinMax.size() < 3)
				stressVariableMinMax.push_back(glm::vec2(FLT_MAX, FLT_MIN));

			//mesh->m_numFaces = iElements;
			//mesh->m_numVertices = iNodes;

			totalVertices += iNodes;
			totalFaces += iElements;
			faceVariables.push_back(3);
			faceVariables.push_back(4);
			faceVariables.push_back(5);

			//for(first to last) push
			if (stressVariables.size() == 0) {
				stressVariables.push_back(variables[3]);
				stressVariables.push_back(variables[4]);
				stressVariables.push_back(variables[5]);
			}
		}

		while (lineOffset < lines.size())
		{
			if (lines[lineOffset].find("DT") != std::string::npos)
				break;
			lineOffset++;
		}
		lineOffset++;

		if (dataPacking == "POINT")
		{

		}
		else if (dataPacking == "BLOCK")
		{

			for (int i = 0; i < variables.size(); i++)
			{
				bool faceVariable = std::find(faceVariables.begin(), faceVariables.end(), i) != faceVariables.end();

				std::vector<float> values;
				int maxPoints = faceVariable ? iElements : iNodes;
				int points = 0;
				while (points < maxPoints)
				{
					std::vector <std::string> elements;
					parseLine(lines[lineOffset], &elements, " ");

					for (int k = 0; k < elements.size(); k++)
					{
						float v = 0.0;
						sscanf(elements[k].c_str(), "%f", &v);

						values.push_back(v);
						points++;
					}

					lineOffset++;
				}

				variableMap[variables[i]] = values;
			}

			//read in the faces
			int faceOffset = lineOffset;
			for (int i = 0; i < iElements; i++)
			{
				int a = 0, b = 0, c = 0;
				std::string line = lines[lineOffset + i];
				sscanf(line.c_str(), " %i %i %i", &a, &b, &c);
				mesh.addTriangle(a - 1, b - 1, c - 1);
			}

			for (int i = 0; i < variableMap["X"].size(); i++)
			{
				glm::vec3 point = glm::vec3(variableMap["X"][i], variableMap["Y"][i], variableMap["Z"][i]) * 1000.0f;
				//m//inPoint = Min(point, minPoint);
				//maxPoint = Max(point, maxPoint);
				Vertex v;
				v.position = point;
				mesh.addVertex(v);
			}

			for (int fv = 0; fv < 1; fv++)
			{
				std::vector<float> data = variableMap[variables[faceVariables[fv]]];

				std::vector<float> accumulatedValues;
				std::vector<int>   sharedFaces;

				for (int i = 0; i < mesh.getNumVertices(); i++)
				{
					accumulatedValues.push_back(0);
					sharedFaces.push_back(0);
				}

				for (int i = 0; i < mesh.getNumTriangles(); i++)
				{
					for (int v = 0; v < 3; v++) {
						int index = mesh.getIndices()[i*3 + v];

						accumulatedValues[index] += data[i];
						sharedFaces[index] += 1;
					}
				}

				for (int i = 0; i < mesh.getNumVertices(); i++)
				{
					accumulatedValues[i] /= sharedFaces[i];
				}

				//mesh->m_vertExtraData.push_back(accumulatedValues);
				float maxData = *std::max_element(std::begin(accumulatedValues), std::end(accumulatedValues));
				float minData = *std::min_element(std::begin(accumulatedValues), std::end(accumulatedValues));

				for (int i = 0; i < mesh.getNumVertices(); i++)
				{
					mesh.setVertexColor(i, glm::vec4(getHeatMapColor((accumulatedValues[i] - 0) / (12 - 0)),1));			
				}

				//int index = mesh->m_vertExtraData.size() - 1;
				//gb::Vec2f minMax = stressVariableMinMax[index];
				//if (minMax.x > minData)
				//	minMax.x = minData;
				//if (minMax.y < maxData)
				//	minMax.y = maxData;
				//stressVariableMinMax[index] = minMax;
			}

			m_meshes.push_back(mesh);
		}
	}
	return m_meshes;

}