#pragma once

#include <functional>
#include "EasingFunctions.h"
#include "ITween.h"

class TweenDelay : public ITween
{
public:
	TweenDelay(float time) : ITween()
	{
		m_duration = time;
		m_invDuration = 1/time;
		m_timer = 0;
	}

	~TweenDelay(){}

	TweenDelay& loop(int times, LoopType type = LoopType::Restart)
	{
		ITween::loop(times, type);
		return *this;
	}

	void update(float dt)
	{
		if(isComplete()) return;

		m_timer += dt;

		if (m_timer >= m_duration)
		{
			m_playCount++;
			if (m_maxLoops > 0 && m_playCount >= m_maxLoops)
			{
				m_complete = true;
			}
			else
			{
				m_timer = m_timer - m_duration;
			}
		}
		
	}

	void reset()
	{
		ITween::reset();
		m_timer = 0;
	}

	float getTimeOverflow() const
	{
		if (isComplete())
			return m_timer - m_duration;
		return 0;
	}

	float getFullDuration() const
	{
		return m_duration * m_maxLoops;
	}

	TweenDelay* clone() const {
		TweenDelay* td = new TweenDelay(m_duration);
		td->m_maxLoops = m_maxLoops;
		td->m_loopType = m_loopType;
		return td;
	}

private:
	float m_timer;
	float m_duration;
	float m_invDuration;
};
