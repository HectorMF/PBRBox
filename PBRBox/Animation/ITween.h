#pragma once

#include <functional>

enum class LoopType { Restart, YoYo };

class ITween
{
public:
	ITween() : m_loopType(LoopType::Restart), m_maxLoops(1), m_playCount(0), m_started(false), m_complete(false), m_reversed(false){ }

	virtual void update(float dt) = 0;

	virtual void reset()
	{
		m_complete = false;
		m_started = false;
		m_playCount = 0;
	}

	virtual void reverse()
	{
		m_reversed = !m_reversed;
	}

	virtual float getFullDuration() const = 0;
	virtual float getTimeOverflow() const = 0;
	virtual ITween* clone() const = 0;


	virtual ITween& loop(int times, LoopType type = LoopType::Restart)
	{
		m_maxLoops = times;
		m_loopType = type;
		return *this;
	}

	bool hasStarted() const { return m_started; }
	bool isReversed() const { return m_reversed; }
	bool isComplete() const { return m_complete; }
	LoopType getLoopType() const { return m_loopType; }
	
protected:
	LoopType m_loopType;
	int m_maxLoops;
	int m_playCount;

	bool m_started;
	bool m_complete;
	bool m_reversed;
};
