#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class SceneObject
{
public:
	bool m_enabled;
	std::string name;
	unsigned int uID;


	glm::mat4 transform;

	void setPosition(glm::vec3);
	glm::vec3 getPosition();

	void setRotationEuler(glm::vec3);
	glm::vec3 getRotationEuler();

	void setRotation(glm::quat);
	glm::quat getRotation();


};