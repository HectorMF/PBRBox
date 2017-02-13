/***************************************************************
*  Copyright (C) 2013 Ohio Supercomputer Center, Ohio State University
*
* This file and its content is protected by a software license.
* You should have received a copy of this license with this file.
* If not, please contact the Ohio Supercomputer Center immediately:
* Attn: Brad Hittle Re: 1224 Kinnear Rd, Columbus, Ohio 43212
*            bhittle@osc.edu
***************************************************************/
#ifndef _Bezier_Spline_h_
#define _Bezier_Spline_h_

#include <Animation/Spline.h>
/*
namespace gearbox{

	template <class T>
	class BezierSpline : public Spline<T>
	{
	public:
		BezierSpline(T& target) : Spline(target) { }

		BezierSpline* clone() const
		{
			BezierSpline<T>* spline = new BezierSpline<T>(m_target);
			spline->m_maxLoops = m_maxLoops;
			spline->m_loopType = m_loopType;
			spline->m_duration = m_duration;
			spline->m_invDuration = m_invDuration;

			for (int i = 0; i < m_points.size(); i++)
				spline->m_points.push_back(m_points[i]);

			return spline;
		}

	protected:
		T compute(T a, T b, T c, T d, float t)
		{
			T t1 = (c - a) * .5f;
			T t2 = (d - b) * .5f;

			float h1 = +2 * t * t * t - 3 * t * t + 1;
			float h2 = -2 * t * t * t + 3 * t * t;
			float h3 = t * t * t - 2 * t * t + t;
			float h4 = t * t * t - t * t;

			return b * h1 + c * h2 + t1 * h3 + t2 * h4;
		}
	};
}*/
#endif
