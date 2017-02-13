#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>

#include "Tween.h"
#include "TweenSequence.h"


struct ActiveTween{
	std::shared_ptr<ITween> tween;
	std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<float>> endTime;
};

struct TweenComparator
{
	static bool sortbyEndTime(const ActiveTween &lhs, const ActiveTween &rhs)
	{
		return lhs.endTime < rhs.endTime;
	}
};

class TweenManager
{
public:
	TweenManager(){}
	~TweenManager(){ m_activeTweens.clear(); }

	void update(float dt){
		for (int i = 0; i < m_activeTweens.size(); i++)
		{
			m_activeTweens[i].tween->update(dt);
		}

		for (int i = m_activeTweens.size() - 1; i >= 0; i--)
		{
			if (m_activeTweens[i].tween->isComplete()){
				m_activeTweens[i].tween.reset();
				m_activeTweens.pop_back();
			}
			else
				break;
		}
	}

	template <class TweenObject>
	void start(const TweenObject tween)
	{
			
		//We want to be able to add any subclass of ITween to this manager, but we need the concrete class to call make_shared();
		//This function is templated allowing any class, but we do a compile time check below to make sure it is valid. 
		static_assert(std::is_base_of<ITween, TweenObject>::value, "Trying to add Tween to TweenManager that does not inherit from the ITween interface");

		ActiveTween activeTween;
		activeTween.tween = std::shared_ptr<ITween>(tween.clone());
		activeTween.endTime = std::chrono::system_clock::now() + std::chrono::duration<float>(tween.getFullDuration());

		m_activeTweens.push_back(activeTween);
		std::sort(m_activeTweens.begin(), m_activeTweens.end(), TweenComparator::sortbyEndTime);
	}

private:
	std::vector<ActiveTween> m_activeTweens;
};
