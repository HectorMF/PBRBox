#pragma once

#include <vector>
#include <memory>

#include <functional>
#include "EasingFunctions.h"
#include "ITween.h"
#include "Tween.h"
#include "TweenDelay.h"

enum class SequenceType { Parallel, Serial };

class TweenSequence : public ITween
{
		
public:
	TweenSequence():TweenSequence(SequenceType::Serial){}

	TweenSequence(SequenceType stype) : ITween()
	{
		parent = NULL;
		m_currentStep = 0;
		m_sequenceType = stype;
	}

	TweenSequence& TweenSequence::loop(int times, LoopType type)
	{
		ITween::loop(times, type);
		return *this;
	}

	TweenSequence& begin(SequenceType type)
	{
		add(TweenSequence(type));
		TweenSequence* ts = static_cast<TweenSequence*> (m_tweens[m_tweens.size() - 1].get());
		ts->parent = this;
		return *ts;
	}

	TweenSequence& end()
	{
		return parent?*parent:*this;
	}

	template <class T>
	TweenSequence& add(T tween)
	{
		//We want to be able to add any subclass of ITween to this sequence, but we need a concrete class to make a shared pointer
		//compile time check to make sure this is only being used in valid ways
		static_assert(std::is_base_of<ITween, T>::value, "Add(T tween): Trying to add Object to TweenSequence that does not inherit from the ITween interface");
		m_tweens.push_back(std::shared_ptr<ITween>(tween.clone()));
		return *this;
	}

	TweenSequence* clone() const
	{
		TweenSequence* ts = new TweenSequence(m_sequenceType);
		ts->m_maxLoops = m_maxLoops;
		ts->m_loopType = m_loopType;
		for (int i = 0; i < m_tweens.size(); i++)
			ts->m_tweens.push_back(std::shared_ptr<ITween>(m_tweens[i]->clone()));
		ts->build();
		return ts;
	}

	TweenSequence& pause(float time)
	{
		add(TweenDelay(time));
		return *this;
	}

	void update(float dt)
	{
		if (isComplete()) return;

		if (m_sequenceType == SequenceType::Serial)
		{
			ITween* currentTween;

			currentTween = getNextTween();
				
			currentTween->update(dt);

			if (currentTween->isComplete())
			{
				float overflow = currentTween->getTimeOverflow();
				if (m_currentStep < m_tweens.size() - 1)
				{
					m_currentStep++;

					getNextTween()->update(overflow);
				}
				else
				{
					m_playCount++;
					if (m_maxLoops >= 0 && m_playCount >= m_maxLoops)
							m_complete = true;
					else
					{
						if (m_loopType == LoopType::YoYo)
							reverse();
						PartialReset();
						getNextTween()->update(overflow);
					}
				}
			}
		}

		if (m_sequenceType == SequenceType::Parallel)
		{
			m_complete = true;
			for (int i = 0; i < m_tweens.size(); i++)
			{
				m_tweens[i]->update(dt);
				if (!m_tweens[i]->isComplete())
					m_complete = false;
			}

			if (m_complete)
			{
				m_playCount++;
				if (m_maxLoops < 0 || m_playCount < m_maxLoops)
				{
					reset();
					if (m_loopType == LoopType::YoYo)
						reverse();
				}
			}
		}
	}

	float getTimeOverflow() const
	{
		float overflow = 0;
		if (m_complete)
		{
			if (m_sequenceType == SequenceType::Parallel)
			{
				if (m_tweens.size() > 0){
					overflow = m_tweens[0]->getTimeOverflow();
					for (int i = 1; i < m_tweens.size(); i++)
					{
						overflow = fminf(m_tweens[i]->getTimeOverflow(), overflow);
					}
				}
			}

			if (m_sequenceType == SequenceType::Serial)
			{
				if (m_tweens.size() > 0)
					overflow = m_tweens[m_tweens.size() - 1]->getTimeOverflow();
			}
		}
		return overflow;
	}

	void reverse()
	{
		ITween::reverse();
		for (int i = 0; i < m_tweens.size(); i++)
		{
			m_tweens[i]->reverse();
		}
	}

	virtual void reset(){
		ITween::reset();

		m_currentStep = 0;
		for (int i = 0; i < m_tweens.size(); i++)
		{
			m_tweens[i]->reset();
		}
	}

	float getFullDuration() const
	{
		float maxDuration = 0;
		if (m_sequenceType == SequenceType::Parallel)
		{
			for (int i = 0; i < m_tweens.size(); i++)
			{
				maxDuration = fmaxf(m_tweens[i]->getFullDuration(), maxDuration);
			}
		}

		if (m_sequenceType == SequenceType::Serial)
		{
			for (int i = 0; i < m_tweens.size(); i++)
			{
				maxDuration += m_tweens[i]->getFullDuration();
			}
		}

		return maxDuration * m_maxLoops;
	}

	private:
		TweenSequence* parent;
		SequenceType m_sequenceType;
		int m_currentStep;
		std::vector<std::shared_ptr<ITween>> m_tweens;

		void build()
		{
			for (int i = 0; i < m_tweens.size(); i++)
			{
				if (TweenSequence* ts = dynamic_cast<TweenSequence*>(m_tweens[i].get()))
				{
					ts->parent = this;
					ts->build();
				}
			}
		}

		void PartialReset()
		{
			m_complete = false;
			if (m_maxLoops < 0)
				m_playCount = 0;
			m_currentStep = 0;

			for (int i = 0; i < m_tweens.size(); i++)
			{
				m_tweens[i]->reset();
			}
		}

		ITween* getNextTween()
		{
			if (isReversed())
				return (m_tweens.rbegin() + m_currentStep)->get();
			else
				return (m_tweens.begin() + m_currentStep)->get();
		}
};