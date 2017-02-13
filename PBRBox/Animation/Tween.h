#pragma once

#include <functional>
#include "EasingFunctions.h"
#include "ITween.h"
#include "../MathUtil.h"

template <class T>
class Tween : public ITween
{
public:
	Tween() : ITween()
	{
		m_target = NULL; 
		m_ease = Easing::Linear;
		m_duration = 1;
		m_invDuration = 1;
		m_delay = 0;
		m_delayComplete = false;
		m_timer = 0;
		m_delayTimer = 0;
	}

	Tween& target(T& target)
	{
		m_target = &target;
		return *this;
	}

	Tween& start(const T start)
	{
		m_start = start;
		return *this;
	}

	Tween& end(const T end)
	{
		m_end = end;
		return *this;
	}

	Tween& ease(std::function<float(float)> func)
	{
		m_ease = func;
		return *this;
	}

	Tween& duration(float time)
	{
		m_duration = time;
		m_invDuration = 1.0f / time;
		return *this;
	}

	Tween& loop(int times, LoopType type = LoopType::Restart)
	{
		ITween::loop(times, type);
		return *this;
	}

	Tween& delay(float time)
	{
		m_delay = time;
		return *this;
	}

	Tween& onStart(std::function<void(Tween<T>&)> func)
	{
		m_startCallback = func;
		return *this;
	}

	Tween& onEnd(std::function<void(Tween<T>&)> func)
	{
		m_endCallback = func;
		return *this;
	}

	Tween& onUpdate(std::function<void(Tween<T>&)> func)
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

		m_current = LERP(m_start, m_end, m_ease(clampedTime));

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

	Tween* clone() const
	{
		Tween<T>* t = new Tween<T>(*this);
		return t;
	}

	T getValue() const
	{
		return m_current;
	}
		
	T getStartValue() const
	{
		return m_start;
	}

	T getEndValue() const
	{
		return m_end;
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


private:
	//The tween target can be a reference to an object, or NULL
	T* m_target;

	T m_start;
	T m_end;
	T m_current;

	std::function<float(float)> m_ease;
	float m_duration;
	float m_invDuration;
	float m_delay;

	float m_timer;
	float m_delayTimer;
	bool m_delayComplete;

	//callback functions, the tween object is passed to the callback since tweens can be cloned
	std::function<void(Tween<T>&)> m_startCallback;
	std::function<void(Tween<T>&)> m_endCallback;
	std::function<void(Tween<T>&)> m_updateCallback;
};