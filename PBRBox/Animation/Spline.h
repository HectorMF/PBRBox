#pragma once

#include <functional>
#include <vector>

#include "ITween.h"

template <class T>
class Spline : public ITween
{
public:
	Spline() : ITween()
	{
		m_target = NULL;
		m_duration = 1;
		m_invDuration = 1;
		m_delay = 0;
		m_delayComplete = false;
		m_timer = 0;
		m_delayTimer = 0;
	}

	Spline& target(T& target)
	{
		m_target = &target;
		return *this;
	}

	Spline& add(const T point)
	{
		m_points.push_back(point);
		return *this;
	}

	Spline& duration(float time)
	{
		m_duration = time;
		m_invDuration = 1.0f / time;
		return *this;
	}

	Spline& loop(int times, LoopType type = LoopType::Restart)
	{
		ITween::loop(times, type);
		return *this;
	}

	Spline& delay(float time)
	{
		m_delay = time;
		return *this;
	}

	Spline& onStart(std::function<void(Spline<T>&)> func)
	{
		m_startCallback = func;
		return *this;
	}

	Spline& onEnd(std::function<void(Spline<T>&)> func)
	{
		m_endCallback = func;
		return *this;
	}

	Spline& onUpdate(std::function<void(Spline<T>&)> func)
	{
		m_updateCallback = func;
		return *this;
	}

	void update(float dt)
	{
		if (isComplete()) return;

		if (!m_started){
			if (m_startCallback)
				m_startCallback(*this);
			m_started = true;
		}

		if (m_delayTimer < m_delay)
		{
			m_delayTimer += dt;
			return;
		}
		else if (!m_delayComplete)
		{
			m_delayComplete = true;
			//account for any overdelay by moving any extra time over to the timer
			m_timer += (m_delayTimer - m_delay);
		}

		m_timer += dt;

		float clampedTime = CLAMP(m_timer * m_invDuration, 0, 1);

		if (isReversed())
			clampedTime = 1 - clampedTime;
			
		m_current = compute(m_points, clampedTime);

		if (m_target)
			*m_target = m_current;

		if (m_updateCallback)
			m_updateCallback(*this);

		if (m_timer >= m_duration){
			m_playCount++;
			if (m_maxLoops > 0 && m_playCount >= m_maxLoops)
			{
				m_complete = true;
				if (m_endCallback)
					m_endCallback(*this);
			}
			else
			{
				if (m_loopType == LoopType::YoYo)
					reverse();
				m_timer = m_timer - m_duration;
			}
		}
	}

	void reset(){
		ITween::reset();
		m_delayComplete = false;
		m_timer = 0;
		m_delayTimer = 0;
	}

	float getFullDuration() const
	{
		return (m_delay + m_duration) * m_maxLoops;
	}

	float getTimeOverflow() const
	{
		if (isComplete())
			return m_timer - m_duration;
		return 0;
	}

	T getValue() const
	{
		return m_current;
	}

	virtual Spline* clone() const = 0;
		
protected:
	T* m_target;
	T m_current;
	virtual T compute(const std::vector<T> points, const float t) = 0;
		
	std::vector<T> m_points;

	float m_duration;
	float m_invDuration;
	float m_delay;

	float m_timer;
	float m_delayTimer;
	bool m_delayComplete;

	std::function<void(Spline<T>&)> m_startCallback;
	std::function<void(Spline<T>&)> m_endCallback;
	std::function<void(Spline<T>&)> m_updateCallback;
};